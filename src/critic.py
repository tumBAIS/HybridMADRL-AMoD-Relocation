"""Multi-agent critic: takes state and all agents' actions as input but ignores action of agent under consideration, computes Q-values 
   for all possible actions of this agent. All computations are made across a mini-batch and agents in parallel."""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Multiply, Reshape
from tensorflow.keras.regularizers import L2
from environment import Environment


class Critic(tf.keras.Model):
    def __init__(self, args, env: Environment, name="Critic"):
        super().__init__(name=name) 
        self.env = env
        self.args = args
        
        self.request_embedding = RequestEmbedding(args)
        self.vehicle_embedding = VehicleEmbedding(args)
        self.vehicle_embedding2 = VehicleEmbedding(args)
        self.requests_context = RequestsContext(args, env)
        self.vehicles_context = VehiclesContext(args)

        self.n_veh = args["veh_count"]
        self.n_req_max = args["max_req_count"]
        self.n_req_total = self.env.n_req_total
        self.n_req_total_const = self.env.n_req_total_const
        self.n_req_rebalancing = self.env.n_req_rebalancing
        self.rebalancing_neighbours_bool = env.rebalancing_neighbours_bool
        self.rebalancing_request_generation = env.rebalancing_request_generation
        self.n_actions = self.n_req_total_const + 1
        self.nodes_count = self.env.nodes_count
        self.max_horizontal_idx = self.env.max_horizontal_idx
        self.max_vertical_idx = self.env.max_vertical_idx

        self.reg_coef = args["regularization_coefficient"]
        self.rebalancing_bool = args["rebalancing_bool"]
    
        self.request_embedding_dim = args["req_embedding_dim"]
        self.vehicle_embedding_dim = args["veh_embedding_dim"]
        self.vehicle_context_dim = args["veh_context_dim"]
        self.request_context_dim = args["req_context_dim"]

        self.veh_dim = self.env.veh_dim
        self.req_dim = self.env.req_dim
        self.mis_dim = self.env.mis_dim
        self.vehicle_input_shape = self.vehicle_embedding_dim + self.veh_dim + self.mis_dim
        self.feature_size = self.request_embedding_dim + self.req_dim + 1 + 1 + self.vehicle_input_shape

        self.req_indices = tf.constant([i for i in range(self.n_req_max)], dtype=tf.int32)
        self.critic_model = self.init_model()

    def init_model(self):
        '''Create Parallel Critic Model with inner and outer layers'''
        # create the input layer
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
        activation_layer = Activation('linear', dtype='float32')
        activation_features = activation_layer(output_features)

        model = tf.keras.models.Model(inputs=input_features, outputs=activation_features, name=self.name)
        
        return model    
       
    @tf.function
    def call(self, state, act, hvs, request_masks, rebalancing_requests):   
        requests_state = state["requests_state"]
        vehicles_state = state["vehicles_state"]
        misc_state = state["misc_state"]
        requests_state = tf.cast(requests_state, dtype=tf.float16)
        vehicles_state = tf.cast(vehicles_state, dtype=tf.float16)
        misc_state = tf.cast(misc_state, dtype=tf.float16)
        
        # add action to request state and prepare next step
        if self.rebalancing_request_generation == "origin_destination_neighbours" and self.rebalancing_bool:
            act, act_reb = tf.split(act, [self.n_req_max, -1], axis=1)
            hor = tf.cast(tf.math.round(rebalancing_requests[:,:,:,0] * self.max_horizontal_idx), tf.int32)
            ver = tf.cast(tf.math.round(rebalancing_requests[:,:,:,1]* self.max_vertical_idx), tf.int32)
            zones = self.env.zone_mapping_table[self.env.get_keys(hor, ver)]
            count =  tf.expand_dims(tf.range(1, self.nodes_count+1, dtype=tf.float32), axis=0)
            act_mask = tf.repeat(count, rebalancing_requests.shape[0], 0)
            act_mask = tf.repeat(tf.expand_dims(act_mask, axis=2), repeats=self.n_veh, axis=2)
            veh_mask = tf.one_hot(act_reb, depth=self.n_veh, dtype=tf.float32)
            mask_reb = veh_mask * act_mask
            mask_reb = tf.reduce_sum(mask_reb, axis=1) - 1
            mask_reb = tf.repeat(tf.expand_dims(mask_reb, axis=2), repeats=6, axis=2)
            mask_reb = tf.where(tf.logical_and(mask_reb == tf.cast(zones, dtype=tf.float32), zones != -1) , tf.ones_like(mask_reb, dtype=tf.float32), tf.zeros_like(mask_reb, dtype=tf.float32))
            mask_reb = tf.expand_dims(mask_reb, axis=3)
            rebalancing_input = tf.concat([rebalancing_requests, mask_reb], axis=3)
            mask = tf.repeat(mask_reb, repeats=5, axis=3)
            vehicles_input_reb = mask_reb * rebalancing_requests[:,:,:,:5]
            vehicles_input_reb = tf.cast(tf.reduce_sum(vehicles_input_reb, axis=2),dtype=tf.float16)
        else:
            vehicles_input_reb = tf.zeros([requests_state.shape[0], self.n_veh, 5], dtype=tf.float16)

        requests_input = tf.cast(tf.expand_dims(tf.where(act==-1, 0., 1.), axis=2), tf.float16)
        requests_input = tf.concat([requests_state, requests_input], axis=2)

        #add info to vehicle state of those requests incl rebalancing which have been executed
        act = tf.one_hot(act, depth=self.n_veh, axis=-1, dtype=tf.float16)

        if self.rebalancing_bool:
            vehicles_input1 = tf.matmul(act, requests_state[:,:,:5], transpose_a=True)

            if self.rebalancing_request_generation == "origin_destination_neighbours" and self.rebalancing_bool:
                vehicles_input1 = vehicles_input1 + vehicles_input_reb
            vehicles_input_reb = tf.concat([vehicles_state, vehicles_input1], axis=2)
            request_embedding = self.request_embedding(requests_input)
            vehicle_embedding = self.vehicle_embedding2(vehicles_input_reb)

        else:
            vehicles_input = tf.matmul(act, requests_state[:,:,:-1], transpose_a=True)
            vehicles_input = tf.concat([vehicles_state, vehicles_input], axis=2)
            request_embedding = self.request_embedding(requests_input)
            vehicle_embedding = self.vehicle_embedding(vehicles_input)

        requests_context_raw, requests_context = self.requests_context(request_embedding, request_masks["s"])
        vehicles_context = self.vehicles_context(vehicle_embedding)

        #combine the context and state of the requests
        requests_input = tf.concat([requests_context, requests_state], axis=2)
        requests_input = tf.expand_dims(requests_input, axis=1)
        requests_input = tf.repeat(requests_input, repeats=self.n_veh, axis=1)
        reject_state = self.get_reject_state(vehicles_state, requests_context)
        if self.rebalancing_request_generation == "origin_destination_neighbours" and self.rebalancing_bool:
            requests_context = tf.expand_dims(tf.expand_dims(requests_context_raw[:,0,:], axis=1), axis=1)
            requests_context = tf.repeat(requests_context, self.n_veh, axis=1)
            requests_context = tf.repeat(requests_context, self.n_req_rebalancing , axis=2)
            rebalancing_input = tf.concat([requests_context, tf.cast(rebalancing_requests, dtype=tf.float16)], axis=3)
            requests_input = tf.concat([requests_input, rebalancing_input], axis=2)
        requests_input = tf.concat([requests_input, reject_state], axis=2)

        #combine the context, misc. state and state of the vehicles
        misc_state = tf.repeat(tf.expand_dims(misc_state,axis=1), repeats=self.n_veh, axis=1)
        
        vehicles_input = tf.concat([vehicles_state, misc_state, vehicles_context], axis=2)
        vehicles_input = tf.expand_dims(vehicles_input, axis=1)
        vehicles_input = tf.repeat(vehicles_input, repeats=self.n_actions, axis=2)
        vehicles_input = tf.reshape(vehicles_input, [-1,self.n_veh, self.n_actions, self.vehicle_input_shape]) #changes to featuresize must be adapted
        within_max_waiting_time = self.env.get_flag_within_max_waiting_time(vehicles_state, hvs, requests_state)
        approach_distance = self.env.get_approach_dist_norm(vehicles_state, requests_state)

        # combine all features
        features = tf.concat([vehicles_input, within_max_waiting_time, approach_distance, requests_input], axis=3)

        # mask features
        mask = tf.repeat(tf.expand_dims(request_masks["l1"], axis=3),  repeats=features.shape[3], axis=3)
        features = features * tf.cast(mask, tf.float16)
        features = tf.where(tf.math.is_nan(features), tf.zeros_like(features), features)
        # shuffle the indices and features to avoid overfitting the end nodes
        shuffeld_indices = tf.random.shuffle(self.req_indices)
        shuffeld_indices = tf.concat([shuffeld_indices, tf.cast(tf.range(self.n_req_max, self.n_actions),dtype=tf.int32)], axis=0)
        features = tf.gather(features, shuffeld_indices, axis=2)
        # compute the probabilities
        q_values = self.critic_model(features)

        # unshuffle the probabilities
        unsorted_indices = tf.argsort(shuffeld_indices, direction='ASCENDING')
        q_values = tf.gather(q_values, unsorted_indices, axis=2)

        return q_values
    
    @tf.function
    def get_reject_state(self, vehicles_state, requests_context):
        ''' append one state with zeros as reject state'''
        position = tf.expand_dims(vehicles_state[:,:,:2], axis=2)
        zeros = tf.expand_dims(tf.expand_dims(tf.zeros_like(vehicles_state[:,:,0]), axis=2), axis=2)
        if self.rebalancing_bool:
            reject_state = tf.concat([position, position, zeros, zeros, zeros, zeros], axis=3)
        else:
            reject_state = tf.concat([position,position, zeros,  zeros, zeros], axis=3)
        reject_state = tf.concat([reject_state, tf.zeros([requests_context.shape[0], self.n_veh, 1, requests_context.shape[2]], dtype=tf.float16)], axis=3)
        return reject_state

#create a layer for embedding the requests
class RequestEmbedding(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="RequestEmbedding")
        
        self.embedding_layer = Dense(args["req_embedding_dim"], activation="relu", kernel_initializer='he_uniform', kernel_regularizer=L2(args["regularization_coefficient"]))

    # embed the requests
    @tf.function
    def call(self, requests_inputs):
        paddings = tf.constant([[0,0],[0,0],[0,3]])
        features = tf.pad(requests_inputs, paddings, constant_values=0.)
        
        return self.embedding_layer(features)

    
#create a layer for embedding the vehicles
class VehicleEmbedding(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="VehicleEmbedding")
        
        self.embedding_layer = Dense(args["veh_embedding_dim"], activation="relu", kernel_initializer='he_uniform', kernel_regularizer=L2(args["regularization_coefficient"]))

    # embed the vehicles
    @tf.function
    def call(self, vehicles_inputs):
        paddings = tf.constant([[0,0],[0,0],[0,4]])
        features = tf.pad(vehicles_inputs, paddings, constant_values=0.)
        
        return self.embedding_layer(features)

# create a layer for computing the context of requests
class RequestsContext(tf.keras.layers.Layer):
    def __init__(self, args, env):
        super().__init__(name="RequestsContext")

        self.n_req_total = env.n_req_total
        self.attention = tf.constant(args["attention"], dtype=tf.bool)
        reg_coef = args["regularization_coefficient"]

        # if attention is true, then initialize the attention layers
        if self.attention:
            self.w = Dense(1, activation="sigmoid", use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))
            self.W = Dense(args["req_context_dim"], activation="tanh", use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))

    @tf.function
    def call(self, requests_embeddings, request_mask_s):
        # if attention is true, then compute the betas
        if self.attention:
            betas = Multiply()([self.w(self.W(requests_embeddings)), tf.expand_dims(request_mask_s, axis=2)])
        # else, set betas to 1
        else:
            betas = tf.expand_dims(tf.cast(request_mask_s, tf.float16), axis=2)
        
        # compute the context of the requests
        requests_context = tf.reduce_sum(betas * requests_embeddings, axis=1) / tf.reduce_sum(tf.cast(request_mask_s, tf.float16), axis=1, keepdims=True)
        requests_context = tf.repeat(tf.expand_dims(requests_context, axis=1), repeats=self.n_req_total, axis=1)
        return requests_context, requests_context - betas * requests_embeddings / tf.expand_dims(tf.reduce_sum(tf.cast(request_mask_s, tf.float16), axis=1, keepdims=True), axis=2) # exclude contribution of individual requests

# create a layer for computing the context of vehicles
class VehiclesContext(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="VehiclesContext")

        self.n_veh = args["veh_count"]
        self.attention = tf.constant(args["attention"], dtype=tf.bool)
        reg_coef = args["regularization_coefficient"]
        self.batch_size = args["batch_size"]
        # if attention is true, then initialize the attention layers
        if self.attention:
            self.w = Dense(1, activation="sigmoid", use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))
            self.W = Dense(args["veh_context_dim"], activation="tanh", use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))

    @tf.function
    def call(self, vehicles_embeddings):
        # if attention is true, then predict the betas        
        if self.attention:
            betas = self.w(self.W(vehicles_embeddings))
        # else, set betas to 1
        else:
            betas = tf.ones([self.batch_size, self.n_veh, 1], tf.float16)
        
        # compute the context of the vehicles
        vehicles_context = tf.reduce_mean(betas * vehicles_embeddings, axis=1)
        vehicles_context = tf.repeat(tf.expand_dims(vehicles_context, axis=1), repeats=self.n_veh, axis=1) 
        return vehicles_context - betas * vehicles_embeddings / tf.cast(self.n_veh, tf.float16) # exclude contribution of individual vehicles
