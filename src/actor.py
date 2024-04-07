"""Multi-agent actor including post-processing. All computations are made across a mini-batch and agents in parallel."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Multiply, Reshape
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import Model
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
from environment import Environment

class Actor(tf.keras.Model):
    def __init__(self, args, env: Environment):
        super().__init__(name="Actor")
        self.env = env
        self.args = args
        
        self.request_embedding = RequestEmbedding(args)
        self.vehicle_embedding = VehicleEmbedding(args)
        self.requests_context = RequestsContext(args)
        self.vehicles_context = VehiclesContext(args)

        self.n_veh = args["veh_count"]
        self.n_req_max = args["max_req_count"]
        self.n_req_total = self.env.n_req_total
        self.n_req_total_const = self.env.n_req_total_const
        self.n_req_rebalancing = self.env.n_req_rebalancing
        self.n_actions = self.n_req_total_const + 1 # added reject actions
        self.max_horizontal_idx = self.env.max_horizontal_idx
        self.max_vertical_idx = self.env.max_vertical_idx
        self.nodes_count = self.env.nodes_count

        self.vehicle_embedding_dim = args["veh_embedding_dim"]
        self.request_embedding_dim = args["req_embedding_dim"]
        self.request_context_dim = args["req_context_dim"]
        self.vehicle_context_dim = args["veh_context_dim"]
        self.reg_coef = args["regularization_coefficient"]
        self.noise_steps = args["noise_steps"]
        self.rebalancing_bool = tf.constant(args["rebalancing_bool"], dtype=tf.bool)
        if args["rebalancing_request_generation"] == "origin_destination_neighbours" and self.rebalancing_bool:
            self.rebalancing_neighbours_bool = tf.constant(True, dtype=tf.bool)
        else:
            self.rebalancing_neighbours_bool = tf.constant(False, dtype=tf.bool)
        
        self.noise_factor = tf.constant(1, dtype=tf.float32)
        self.veh_dim = self.env.veh_dim
        self.req_dim = self.env.req_dim
        self.mis_dim = self.env.mis_dim
        self.feature_size = self.request_embedding_dim + self.vehicle_embedding_dim + self.veh_dim + self.req_dim + self.mis_dim +1

        self.req_indices = tf.constant([i for i in range(self.n_req_max)], dtype=tf.int32)
        self.actor_model = self.init_model()

    def init_model(self):
        '''Create Parallel Actor Model with inner and outer layers'''
        input_features = tf.keras.Input(shape=(self.n_veh, self.n_actions, self.feature_size))

        # create the inner layers
        for i in range(len(self.args["inner_units"])):
            layer_size = self.args["inner_units"][i]
            if i == 0:
                dense_layer = Dense(layer_size, activation="relu", kernel_initializer='he_uniform', kernel_regularizer=L2(self.args["regularization_coefficient"]))
                features = dense_layer(input_features)
            else:
                dense_layer = Dense(layer_size, activation="relu", kernel_initializer='he_uniform', kernel_regularizer=L2(self.args["regularization_coefficient"]))
                features = dense_layer(features)
        
        __batch, __veh, index_req, index_feat = features.shape
        
        # add reshape layer
        reshape_layer = Reshape((-1,index_req*index_feat))
        features = reshape_layer(features)

        # add layers after flattening
        for i in range(len(self.args["outer_units"])):
            layer_size = self.args["outer_units"][i] #* self.n_req_max
            dense_layer = Dense(layer_size, activation="relu", kernel_initializer='he_uniform', kernel_regularizer=L2(self.args["regularization_coefficient"]))
            features = dense_layer(features)

        output_layer = Dense(self.n_actions, kernel_initializer='glorot_uniform', kernel_regularizer=L2(self.args["regularization_coefficient"]))
        output_features = output_layer(features)
        activation_layer = Activation('softmax', dtype='float32')
        activation_features = activation_layer(output_features)

        model = Model(inputs=input_features, outputs=activation_features, name=self.name)
        
        return model

    def __call__(self, state, hvs, test, request_masks, rebalancing_requests):
        """returns action for agent after computing probabilities"""
        probs = self.compute_prob(state, request_masks, rebalancing_requests)
        act, rejects = self.post_process(probs, test, hvs, request_masks["l1"], state["vehicles_state"], state["requests_state"], rebalancing_requests)
        # calculate the percentage of rejected requests
        zeros = tf.reduce_sum(tf.cast(tf.equal(request_masks["s"], 1), tf.int32))
        rejected = tf.reduce_sum(tf.cast(tf.equal(act, -1), tf.int32)) - tf.reduce_sum(tf.cast(tf.equal(request_masks["s"], 1), tf.int32))
        rejected_percentage = tf.cast(rejected, tf.float32) / tf.cast(self.n_req_total - zeros, tf.float32)
        return tf.squeeze(act, axis=[0]), rejected_percentage, tf.squeeze(rejects, axis=[0])

    def get_random_action(self, state, hvs, request_mask, rebalancing_requests):
        """return random action"""
        probs = tf.random.uniform((1, self.n_veh, self.n_actions), minval=0, maxval=1, dtype=tf.float32)
        probs /= tf.reduce_sum(probs, axis=2, keepdims=True)
        act, rejects = self.post_process(probs, tf.constant(False), tf.expand_dims(hvs, axis=0), request_mask["l1"], tf.expand_dims(state["vehicles_state"], axis=0),tf.expand_dims(state["requests_state"], axis=0), rebalancing_requests, tf.constant(True))
        return tf.squeeze(act, axis=[0]), tf.squeeze(rejects, axis=[0])

    @tf.function
    def compute_prob(self, state, request_masks, rebalancing_requests):
        requests_state_raw = state["requests_state"]
        vehicles_state_raw = state["vehicles_state"]

        # compute the embeddings and context
        request_embedding = self.request_embedding(requests_state_raw)
        vehicle_embedding = self.vehicle_embedding(vehicles_state_raw)
        requests_context = self.requests_context(request_embedding, request_masks["s"])
        vehicles_context = self.vehicles_context(vehicle_embedding)

        # create feature vectors
        request_state = tf.expand_dims(requests_state_raw, axis=1)
        request_state = tf.repeat(request_state, repeats=self.n_veh, axis=1)
        vehicles_state = tf.expand_dims(vehicles_state_raw, axis=2)
        vehicles_state = tf.repeat(vehicles_state, repeats=self.n_actions, axis=2)
        vehicles_state = tf.reshape(vehicles_state, [-1,self.n_veh, self.n_actions, self.veh_dim])
        reject_state = self.get_reject_state(vehicles_state_raw)
        if self.rebalancing_neighbours_bool:
            request_state = tf.concat([request_state, rebalancing_requests], axis=2)
        request_state = tf.concat([request_state, reject_state], axis=2)
        # combine context, embedding and misc state
        context = tf.concat([requests_context, vehicles_context], axis=1)
        context = tf.expand_dims(context, axis=1)
        misc_state =tf.expand_dims(tf.cast(state["misc_state"], tf.float16), axis=1)
        combined_input = tf.concat([context, misc_state], axis=2)
        combined_input = tf.expand_dims(combined_input, axis=1)
        combined_input = tf.repeat(combined_input, repeats=self.n_veh, axis=1)
        combined_input = tf.repeat(combined_input, repeats=self.n_actions, axis=2)
        approach_distance = self.env.get_approach_dist_norm(vehicles_state_raw, requests_state_raw)
        # combine all features
        request_state = tf.cast(request_state, tf.float16)
        vehicles_state = tf.cast(vehicles_state, tf.float16)
        features = tf.concat([combined_input, approach_distance, request_state, vehicles_state ], axis=3)
        
        # mask features
        mask = tf.repeat(tf.expand_dims(request_masks["l1"], axis=3),  repeats=features.shape[3], axis=3)
        features = features * tf.cast(mask, tf.float16)
        features = tf.where(tf.math.is_nan(features), tf.zeros_like(features), features)
        
        # shuffle the indices and features to avoid overfitting the end nodes of customer requests
        shuffeld_indices = tf.random.shuffle(self.req_indices)
        shuffeld_indices = tf.concat([shuffeld_indices, tf.cast(tf.range(self.n_req_max, self.n_actions),dtype=tf.int32)], axis=0)
        features = tf.gather(features, shuffeld_indices, axis=2)
        
        # compute the probabilities
        probs = self.actor_model(features)
        
        # unshuffle the probabilities
        unsorted_indices = tf.argsort(shuffeld_indices, direction='ASCENDING')
        probs = tf.gather(probs, unsorted_indices, axis=2)
        return probs

    def post_process(self, probs, test, hvs, request_mask_l, vehicles_state, requests_state, rebalancing_requests, random=tf.constant(False)):
        batch_size = tf.shape(probs)[0]

        # add decreasing noise to the probabilities
        if not test and self.noise_factor > 0 and not random:
            probs = self.add_noise(probs, self.noise_factor)
            self.noise_factor = tf.math.maximum(self.noise_factor - 1/(self.noise_steps+1), 0)

        probs = self.mask_all_probs(probs, hvs, request_mask_l, requests_state, vehicles_state)

        #get action
        act, rejects = self.get_action_from_probs(probs, rebalancing_requests)
        act = act.numpy()
        # parallelize the weighted matching, which is done by linear sum assignment for each batch
        action_list = Parallel(n_jobs=2, prefer="threads")(delayed(self.weighted_matching)(act[i,:,:]) for i in range(batch_size))
        act = tf.constant(action_list)
        matched_actions, _ = tf.split(act, num_or_size_splits=[act.shape[1]-self.n_veh, self.n_veh], axis=1)

        return matched_actions, rejects

    @tf.function
    def mask_all_probs(self, probs, hvs, request_mask_l, requests_state, vehicles_state):
        '''mask all the probabilities according to the request mask'''
        request_mask , _ = tf.split(request_mask_l, num_or_size_splits=[-1, 1], axis=2)
        if self.rebalancing_bool:
            # access the last n requests from the request state
            _, rebalancing_requests = tf.split(requests_state, num_or_size_splits=[self.n_req_max, self.n_req_total-self.n_req_max], axis=1)
            request_mask_r2 , request_mask_r1 = tf.split(request_mask, num_or_size_splits=[self.n_req_max, self.n_req_total_const-self.n_req_max], axis=2)
            probs_r2 , probs_r1, probs_reject = tf.split(probs, num_or_size_splits=[self.n_req_max, self.n_req_total_const-self.n_req_max, 1], axis=2)
            probs_r2 = self.mask_probs(probs_r2, hvs, request_mask_r2)
            probs_r1 = self.mask_probs_rebalancing(probs_r1, hvs, request_mask_r1, vehicles_state, rebalancing_requests)
            probs_reject = self.mask_probs_reject(hvs, probs_reject)
            probs = tf.concat([probs_r2, probs_r1, probs_reject], axis=2)
        else:
            probs_r2 , probs_reject = tf.split(probs, num_or_size_splits=[-1, 1], axis=2)
            probs_r2 = self.mask_probs(probs_r2, hvs, request_mask)
            probs_reject = self.mask_probs_reject(hvs, probs_reject)
            probs = tf.concat([probs_r2, probs_reject], axis=2)
        return probs
        
    @tf.function
    def mask_probs(self, probs, hvs, request_mask):
        '''mask the probabilities for regular requests according to the request mask'''
        mask = hvs[:,:,6] == -1
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.tile(mask, multiples=[1, request_mask.shape[2], 1])
        mask = tf.transpose(mask, perm=[0,2,1])
        mask = mask & tf.cast(request_mask, tf.bool)
        probs = probs * tf.cast(mask, tf.float32)
        return probs
    
    @tf.function
    def mask_probs_reject(self, hvs, probs):
        '''mask the reject action'''
        mask = hvs[:,:,6] == -1
        mask = tf.expand_dims(mask, axis=2)
        probs = probs * tf.cast(mask, tf.float32)
        return probs

    @tf.function
    def mask_probs_rebalancing(self, probs, hvs, request_mask, vehicles_state, rebalancing_requests):
        '''mask the probabilities for rebalancing requests according to the request mask'''
        mask = hvs[:,:,3] == -1
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.tile(mask, multiples=[1, request_mask.shape[2], 1])
        mask = tf.transpose(mask, perm=[0,2,1]) 
        
        if self.rebalancing_neighbours_bool: 
            mask = mask & tf.cast(request_mask, tf.bool)
        else:
            mask_rebalancing = self.check_origin_position(self.n_req_rebalancing, vehicles_state, rebalancing_requests)
            mask = mask & mask_rebalancing & tf.cast(request_mask, tf.bool)
 
        probs = probs * tf.cast(mask, tf.float32)
        return probs

    @tf.function
    def get_reject_state(self, vehicles_state):
        '''append one state with zeros as reject state'''
        position = tf.expand_dims(vehicles_state[:,:,:2], axis=2)
        zeros = tf.expand_dims(tf.expand_dims(tf.zeros_like(vehicles_state[:,:,0]), axis=2), axis=2)
        if self.rebalancing_bool:
            reject_state = tf.concat([position, position, zeros, zeros, zeros, zeros], axis=3)
        else:
            reject_state = tf.concat([position,position, zeros, zeros, zeros], axis=3)
        return reject_state

    @tf.function
    def check_origin_position(self, size_req, vehicles_state, requests_state):
        '''check if the origin position of the rebalancing requests is the same as the current position of the vehicles'''
        position_horizontal = vehicles_state[:,:,0]
        position_horizontal = tf.expand_dims(position_horizontal, axis=1)
        position_horizontal = tf.tile(position_horizontal, multiples=[1, size_req, 1])
        position_horizontal = tf.transpose(position_horizontal, perm=[0,2,1]) 
        
        position_vertical = vehicles_state[:,:,1]
        position_vertical = tf.expand_dims(position_vertical, axis=1)
        position_vertical = tf.tile(position_vertical, multiples=[1, size_req, 1])
        position_vertical = tf.transpose(position_vertical, perm=[0,2,1]) 
       
        origin_horizontal = requests_state[:,:,0]
        origin_horizontal = tf.expand_dims(origin_horizontal, axis=1)
        origin_horizontal = tf.tile(origin_horizontal, multiples=[1, self.n_veh, 1])
        
        origin_vertical = requests_state[:,:,1]
        origin_vertical = tf.expand_dims(origin_vertical, axis=1)
        origin_vertical = tf.tile(origin_vertical, multiples=[1, self.n_veh, 1])

        mask = tf.math.logical_or(position_horizontal != origin_horizontal, position_vertical != origin_vertical)

        return mask

    
    @tf.function
    def get_action_from_probs(self, probs, rebalancing_requests):
        '''get action from probabilities'''
        #sample all actions with probability higher than 1/n_actions (treshhold)
        sampled_actions = tf.where(probs > 1/self.n_actions, probs, tf.zeros_like(probs))
        #retrieve the active reject action
        sampled_actions, rejects = tf.split(sampled_actions, num_or_size_splits=[self.n_req_total_const, 1], axis=2)
        reject_all = tf.where(tf.reduce_sum(sampled_actions, axis=2) == 0, 1, 0)
        rejects = tf.repeat(rejects, repeats=self.n_veh, axis=2)
        mask = tf.one_hot(tf.range(self.n_veh), depth=self.n_veh, dtype=tf.float32)
        rejects = rejects * mask

        if self.rebalancing_neighbours_bool:
            #accumulate the 6 neighbouring rebalancing actions to one vector with the size of the nodes_count
            sampled_cust_req, sampled_reb_req = tf.split(sampled_actions, num_or_size_splits=[self.n_req_max, self.n_req_total_const-self.n_req_max], axis=2)
            hor = tf.cast(tf.math.round(rebalancing_requests[:,:,:,0] * self.max_horizontal_idx), tf.int32)
            ver = tf.cast(tf.math.round(rebalancing_requests[:,:,:,1]* self.max_vertical_idx), tf.int32)
            zones = self.env.zone_mapping_table[self.env.get_keys(hor, ver)]
            mask = tf.one_hot(zones, depth=self.nodes_count, dtype=tf.float32)
            sampled_reb_req = tf.repeat(tf.expand_dims(sampled_reb_req, axis=3), repeats=(self.nodes_count), axis=3)
            sampled_reb_req = sampled_reb_req * mask
            sampled_reb_req = tf.reduce_sum(sampled_reb_req, axis=2)
            extended_actions = tf.concat([sampled_cust_req, sampled_reb_req, rejects], axis=2)
        else:
            extended_actions = tf.concat([sampled_actions, rejects], axis=2)#
        return extended_actions, reject_all
    
    @tf.function
    def add_noise(self, probs, noise_factor):
        '''get noise from normal distribution'''
        noise = tf.random.normal(shape=tf.shape(probs), mean=0.0, stddev=tf.cast(tf.math.pow(1/self.n_actions, 2), tf.float32), dtype=tf.float32)
        noise *= noise_factor
        probs = tf.clip_by_value(probs + noise, 0, 1)
        probs /= tf.reduce_sum(probs, axis=2, keepdims=True)
        probs = tf.where(tf.math.is_nan(probs), tf.zeros_like(probs), probs)
        return probs
 
    def matching(self, x):
        '''matching according to bipartite graph'''
        return maximum_bipartite_matching(csr_matrix(x))
    
    def weighted_matching(self, x):
        '''weighted matching according to linear sum assignment'''
        matched_veh, matched_req = linear_sum_assignment(x, maximize=True)  # weighted matching

        # correct matching decision to reject decision if weight is zero
        matched_weights = x[matched_veh, matched_req]
        matched_veh = np.where(matched_weights == 0., -1, matched_veh)  # if weight is zero, correct matching decision to reject decision
        
        if self.rebalancing_neighbours_bool:
            action = -np.ones(self.n_req_max + self.nodes_count + self.n_veh, int)
        else:
            action = -np.ones(self.env.n_req_total + self.n_veh, int)

        action[matched_req] = matched_veh
        
        return action

# create an embedding layer for requests
class RequestEmbedding(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="RequestEmbedding")
        
        self.embedding_layer = Dense(args["req_embedding_dim"], activation="relu", kernel_initializer='he_uniform', kernel_regularizer=L2(args["regularization_coefficient"]))
    
    # call the embedding layer with padding
    @tf.function
    def call(self, requests_state):
        paddings = tf.constant([[0,0],[0,0],[0,3]])
        features = tf.pad(requests_state, paddings, constant_values=0.)
        
        return self.embedding_layer(features)

# create an embedding layer for vehicles
class VehicleEmbedding(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="VehicleEmbedding")
        
        self.embedding_layer = Dense(args["veh_embedding_dim"], activation="relu", kernel_initializer='he_uniform', kernel_regularizer=L2(args["regularization_coefficient"]))
    # call the embedding layer with padding
    @tf.function
    def call(self, vehicles_state):
        paddings = tf.constant([[0,0],[0,0],[0,4]])
        features = tf.pad(vehicles_state, paddings, constant_values=0.)
        
        return self.embedding_layer(features)

# create a context layer for requests
class RequestsContext(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="RequestsContext")

        self.attention = tf.constant(args["attention"], dtype=tf.bool)
        reg_coef = args["regularization_coefficient"]
        
        # if attention is true, then initialize the attention layers
        if self.attention:
            self.w = Dense(1, activation="sigmoid", use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))
            self.W = Dense(args["req_context_dim"], activation="tanh", use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))

    @tf.function
    def call(self, requests_embeddings, request_mask_s):
        # if attention is true, then compute the attention weights
        if self.attention:
            betas = Multiply()([self.w(self.W(requests_embeddings)), tf.expand_dims(request_mask_s, axis=2)])
        # else, set the attention weights to the mask
        else:
            betas = tf.expand_dims(tf.cast(request_mask_s, tf.float16), axis=2)
        
        # compute the context vector?
        return tf.reduce_sum(betas * requests_embeddings, axis=1) / tf.reduce_sum(tf.cast(request_mask_s, tf.float16), axis=1, keepdims=True)

# create a context layer for vehicles
class VehiclesContext(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="VehiclesContext")
        
        self.attention = tf.constant(args["attention"], dtype=tf.bool)
        reg_coef = args["regularization_coefficient"]

        # if attention is true, then initialize the attention layers
        if self.attention:
            self.w = Dense(1, activation="sigmoid", use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))
            self.W = Dense(args["veh_context_dim"], activation="tanh", use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))

    @tf.function
    def call(self, vehicles_embeddings):
        # if attention is true, then compute the attention weights?
        if self.attention:
            betas = self.w(self.W(vehicles_embeddings))
            return tf.reduce_mean(betas * vehicles_embeddings, axis=1)
        else:
            return tf.reduce_mean(vehicles_embeddings, axis=1)
