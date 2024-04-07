"""Set up networks and define one training iteration"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import LossScaleOptimizer

from actor import Actor
from critic import Critic
from environment import Environment


class SACDiscrete(tf.keras.Model):
    def __init__(self, args, env: Environment):
        super().__init__()
        self.env = env
        # set up all parameters
        self.n_veh = args["veh_count"]
        self.n_req_max = args["max_req_count"]
        self.batch_size = args["batch_size"]
        self.alpha = tf.exp(tf.constant(args["log_alpha"]))
        self.tau = tf.constant(args["tau"])
        self.huber_delta = tf.constant(args["huber_delta"])
        self.gradient_clipping = tf.constant(args["gradient_clipping"])
        self.clip_norm = tf.constant(args["clip_norm"])
        self.discount = tf.Variable(args["discount"], dtype=tf.float32)
        self.n_actions = env.n_req_total + 1
        self.n_req_total = env.n_req_total
        self.n_req_total_const = env.n_req_total_const
        self.n_req_rebalancing = env.n_req_rebalancing
        self.rebalancing_bool = args["rebalancing_bool"]
        self.rebalancing_request_generation = args["rebalancing_request_generation"]
        if args["rebalancing_request_generation"] == "origin_destination_neighbours" and self.rebalancing_bool:
            self.rebalancing_neighbours_bool = tf.constant(True, dtype=tf.bool)
        else:
            self.rebalancing_neighbours_bool = tf.constant(False, dtype=tf.bool)
        self.record_bool = args["record_bool"]
        self.max_horizontal_idx = env.max_horizontal_idx
        self.max_vertical_idx = env.max_vertical_idx
        self.nodes_count = env.nodes_count
        self.adjusted_loss_bool = args["adjusted_loss_bool"]

        # set up the networks
        self.actor = Actor(args, env)
        self.qf1 = Critic(args, env, name="qf1")
        self.qf2 = Critic(args, env, name="qf2")
        self.qf1_target = Critic(args, env, name="qf1_target")
        self.qf2_target = Critic(args, env, name="qf2_target")
        
        # set up the optimizers
        lr = args["lr"]
        self.actor_optimizer = LossScaleOptimizer(Adam(lr))
        self.qf1_optimizer = LossScaleOptimizer(Adam(lr))
        self.qf2_optimizer = LossScaleOptimizer(Adam(lr))
        
        # set up the update functions
        self.q1_update = tf.function(self.q_update) 
        self.q2_update = tf.function(self.q_update)

    def get_action(self, state, hvs, rebalancing_requests, test=tf.constant(False)):
        '''get action from actor network for state input without batch dim'''
        state, request_masks = self.get_action_body(state, rebalancing_requests)
        return self.actor(state, tf.expand_dims(hvs, axis=0), test, request_masks, rebalancing_requests)
    
    @tf.function
    def get_action_body(self, state, rebalancing_requests):
        '''get body by combining state and request masks'''
        requests_state = state["requests_state"]
        vehicles_state = state["vehicles_state"]
        misc_state = state["misc_state"]
        
        requests_state = tf.expand_dims(requests_state, axis=0)
        vehicles_state = tf.expand_dims(vehicles_state, axis=0)
        misc_state = tf.expand_dims(misc_state, axis=0)
        
        state = {"requests_state": requests_state,
                 "vehicles_state": vehicles_state,
                 "misc_state": misc_state}
                
        return state, self.get_masks(requests_state, rebalancing_requests)

    @tf.function
    def get_masks(self, requests_state, rebalancing_requests=None):
        '''request masks of shape (batch size, n_req_max) and (batch size, n_req_max * n_veh)'''
        request_mask_s = tf.cast(tf.reduce_sum(requests_state, axis=2) > 0, tf.float32)
            
        if self.rebalancing_neighbours_bool:
            request_mask_l_cust = tf.expand_dims(request_mask_s, axis=1)
            request_mask_l_cust = tf.repeat(request_mask_l_cust, repeats=self.n_veh, axis=1)
            request_mask_reb = tf.cast(tf.reduce_sum(rebalancing_requests, axis=3) > 0, tf.float32)
            request_mask_l = tf.concat([request_mask_l_cust, request_mask_reb], axis=2)
            request_mask_l1 = tf.concat([request_mask_l, tf.ones((request_mask_s.shape[0], self.n_veh ,1), dtype=tf.float32)], axis=2)
        else:
            request_mask_l = tf.expand_dims(request_mask_s, axis=1)
            request_mask_l = tf.repeat(request_mask_l, repeats=self.n_veh, axis=1)
            request_mask_l1 = tf.expand_dims(tf.concat([request_mask_s, tf.ones((request_mask_s.shape[0], 1), dtype=tf.float32)], axis=1), axis=1)
            request_mask_l1 = tf.repeat(request_mask_l1, repeats=self.n_veh, axis=1)

        # s1 and l1 include the reject action, depends on whether tensors are used before or after output
        request_masks = {"s": tf.stop_gradient(request_mask_s),
                         "l1": tf.stop_gradient(request_mask_l1)}
        
        return request_masks

    # define one training iteration for a batch of experience
    def train(self, states, hvses, actions, rejects, rewards, next_states, next_hvses):
        if self.rebalancing_neighbours_bool:
            rebalancing_requests = self.env.get_rebalancing_requests(states["vehicles_state"])
            next_rebalancing_requests = self.env.get_rebalancing_requests(next_states["vehicles_state"])
        elif self.rebalancing_bool:
            rebalancing_requests = tf.zeros([states["vehicles_state"].shape[0],self.n_veh, self.n_req_total - self.n_req_max, 8])
            next_rebalancing_requests = tf.zeros([states["vehicles_state"].shape[0],self.n_veh, self.n_req_total - self.n_req_max, 8])
        else:
            rebalancing_requests = tf.zeros([states["vehicles_state"].shape[0],self.n_veh, self.n_req_total - self.n_req_max, 7])
            next_rebalancing_requests = tf.zeros([states["vehicles_state"].shape[0],self.n_veh, self.n_req_total - self.n_req_max,7])
            
        request_masks = self.get_masks(states["requests_state"], rebalancing_requests)
        #get current action probabilities and actions
        cur_act_prob = self.actor.compute_prob(states, request_masks, rebalancing_requests)
        actions_current_policy, _  = self.actor.post_process(cur_act_prob, tf.constant(True), hvses, request_masks["l1"], states["vehicles_state"], states["requests_state"], rebalancing_requests)
        cur_act_prob = self.actor.mask_all_probs(cur_act_prob, hvses, request_masks["l1"], states["requests_state"], states["vehicles_state"])
        
        #get target Q values
        next_request_masks = self.get_masks(next_states["requests_state"], next_rebalancing_requests)
        target_q = self.target_Qs(rewards, next_states, next_request_masks, next_hvses, next_rebalancing_requests)
        #train the networks
        q1_loss, q2_loss, policy_loss, mean_ent, cur_act_logp = self.train_body(states, hvses, actions, rejects, target_q, request_masks, actions_current_policy, cur_act_prob, rebalancing_requests)
        if self.record_bool:
            tf.summary.scalar(name="critic_loss", data=(q1_loss + q2_loss) / 2.)
            tf.summary.scalar(name="actor_loss", data=policy_loss)
            tf.summary.scalar(name="mean_ent", data=mean_ent)
            tf.summary.scalar(name="logp_mean", data=tf.reduce_mean(cur_act_logp))

    @tf.function
    def train_body(self, states, hvses, actions, rejects, target_q, request_masks, actions_current_policy, cur_act_prob, rebalancing_requests):
        #get current Q values
        cur_q1 = self.qf1(states, actions_current_policy, hvses, request_masks, rebalancing_requests)
        cur_q2 = self.qf2(states, actions_current_policy, hvses, request_masks, rebalancing_requests)
        mask = self.get_mask_from_actions(actions, rejects, rebalancing_requests)

        #update the Q networks
        q1_loss = self.q1_update(states, hvses, actions, mask, target_q, self.qf1, self.qf1_optimizer, self.qf1_target, request_masks, rebalancing_requests)
        q2_loss = self.q2_update(states, hvses, actions, mask, target_q, self.qf2, self.qf2_optimizer, self.qf2_target, request_masks, rebalancing_requests)

        policy_loss, cur_act_prob, cur_act_logp = self.actor_update(states, hvses, request_masks, cur_q1, cur_q2, rebalancing_requests)
        mean_ent = self.compute_mean_ent(cur_act_prob, cur_act_logp, request_masks["s"]) # mean entropy (info for summary output, not needed for algorithm)
        
        return q1_loss, q2_loss, policy_loss, mean_ent, cur_act_logp

    def target_Qs(self, rewards, next_states, next_request_masks, next_hvses, next_rebalancing_requests):
        '''get target Q values for a batch of experience'''
        next_act_prob = self.actor.compute_prob(next_states, next_request_masks, next_rebalancing_requests)
        next_actions, next_rejects = self.actor.post_process(next_act_prob, tf.constant(True), next_hvses, next_request_masks["l1"], next_states["vehicles_state"], next_states["requests_state"], next_rebalancing_requests)
        next_act_prob = self.actor.mask_all_probs(next_act_prob, next_hvses, next_request_masks["l1"], next_states["requests_state"], next_states["vehicles_state"])
        action_mask_q = self.get_mask_from_actions(next_actions, next_rejects, next_rebalancing_requests)
        return self.target_Qs_body(rewards, next_states, next_hvses, next_actions, next_act_prob, next_request_masks, action_mask_q, next_rebalancing_requests)
    
    @tf.function
    def target_Qs_body(self, rewards, next_states, next_hvses, next_actions, next_act_prob, next_request_masks, action_mask, next_rebalancing_requests):
        #get next Q values from target networks
        next_q1_target = self.qf1_target(next_states, next_actions, next_hvses, next_request_masks, next_rebalancing_requests)
        next_q2_target = self.qf2_target(next_states, next_actions, next_hvses, next_request_masks, next_rebalancing_requests)
        
        #get the minimum of the two next Q values
        next_q = tf.minimum(next_q1_target, next_q2_target)

        if self.adjusted_loss_bool:
            # get correct Q(s,a) from Q(s) 
            next_q = tf.reduce_sum(next_q * tf.cast(action_mask, dtype=tf.float32), axis = 2)
            # get correct probs(a) from probs 
            next_act_prob = tf.reduce_sum(next_act_prob * tf.cast(action_mask, dtype=tf.float32), axis = 2)
            next_action_logp = tf.math.log(next_act_prob + 1e-8)
            target_q = tf.where(next_act_prob > 0, next_q - self.alpha * next_action_logp, 0)
        else:
            next_action_logp = tf.math.log(next_act_prob + 1e-8)
            target_q = tf.einsum('ijk,ijk->ij', next_act_prob, next_q - self.alpha * next_action_logp)

        return tf.stop_gradient(rewards + self.discount * target_q)        

    def q_update(self, states, hvses, actions, mask, target_q, qf, qf_optimizer, qf_target, request_masks, rebalancing_requests):
        with tf.GradientTape() as tape:
            cur_q = qf(states, actions, hvses, request_masks, rebalancing_requests) # gives Q(s) for all a, not Q(s,a) for one a
            cur_q_selected = tf.reduce_sum(cur_q * tf.cast(mask, dtype=tf.float32), axis = 2) # get correct Q(s,a) from Q(s)
            target_q = tf.where(cur_q_selected == 0, tf.zeros_like(target_q), target_q)
            # calculate the loss
            q_loss = self.huber_loss(target_q - cur_q_selected, self.huber_delta)
            q_loss = tf.reduce_mean(tf.reduce_sum(q_loss, axis=1)) # sum over agents and expectation over batch (mean)
            # add regularization loss
            regularization_loss = tf.reduce_sum(qf.losses)
            scaled_q_loss = qf_optimizer.get_scaled_loss(q_loss + regularization_loss)
        
        # calculate the gradients and apply them
        scaled_gradients = tape.gradient(scaled_q_loss, qf.trainable_weights)
        gradients = qf_optimizer.get_unscaled_gradients(scaled_gradients)
        if self.gradient_clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        qf_optimizer.apply_gradients(zip(gradients, qf.trainable_weights))
        
        # update the target network
        for target_var, source_var in zip(qf_target.weights, qf.weights):
            target_var.assign(self.tau * source_var + (1. - self.tau) * target_var)
        
        return q_loss

    @tf.function
    def huber_loss(self, x, delta):
        '''huber loss with delta'''
        delta = tf.ones_like(x) * delta
        less_than_max = 0.5 * tf.square(x) # MSE
        greater_than_max = delta * (tf.abs(x) - 0.5 * delta) # linear
        return tf.where(tf.abs(x)<=delta, x=less_than_max, y=greater_than_max) # MSE for -delta < x < delta, linear otherwise

    @tf.function
    def actor_update(self, states, hvses, request_masks, cur_q1, cur_q2, rebalancing_requests):
        with tf.GradientTape() as tape:
            # get the current action probabilities and log probabilities
            cur_act_prob = self.actor.compute_prob(states, request_masks, rebalancing_requests)
            cur_act_prob = self.actor.mask_all_probs(cur_act_prob, hvses, request_masks["l1"], states["requests_state"], states["vehicles_state"])
            cur_act_logp = tf.math.log(cur_act_prob + 1e-8)
            # calculate the loss via dot product of cur_act_prob and (self.alpha * cur_act_logp - tf.minimum(cur_q1, cur_q2))
            policy_loss = tf.einsum('ijk,ijk->ij', cur_act_prob, self.alpha * cur_act_logp - tf.stop_gradient(tf.minimum(cur_q1, cur_q2)))
            policy_loss = tf.reduce_mean(tf.reduce_sum(policy_loss, axis=1)) # sum over agents and expectation over batch
            # add regularization loss
            regularization_loss = tf.reduce_sum(self.actor.losses)
            scaled_loss = self.actor_optimizer.get_scaled_loss(policy_loss + regularization_loss)

        # calculate the gradients and apply them
        scaled_gradients = tape.gradient(scaled_loss, self.actor.trainable_weights)
        gradients = self.actor_optimizer.get_unscaled_gradients(scaled_gradients)
        if self.gradient_clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_weights))
        
        return policy_loss, cur_act_prob, cur_act_logp

    @tf.function
    def compute_mean_ent(self, cur_act_prob, cur_act_logp, request_mask_s):
        '''compute the mean entropy of the current action probabilities'''
        mean_ent = -tf.einsum('ijk,ijk->ij', cur_act_prob, cur_act_logp)
        mask = tf.where(tf.reduce_sum(request_mask_s, axis=1) > 0, 1, 0)
        mask = tf.repeat(tf.expand_dims(mask, axis=1), axis=1, repeats=self.n_veh)
        mean_ent = mean_ent * tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(mean_ent) / tf.reduce_sum(tf.cast(mask,dtype=tf.float32)) # mean over agents and batch

    @tf.function
    def get_mask_from_actions(self, actions, rejects, rebalancing_requests):
        '''create mask for target Q values according to the global action'''
        if self.rebalancing_request_generation == "origin_destination_neighbours" and self.rebalancing_bool:
            act_cust, act_reb = tf.split(actions, [self.n_req_max, -1], axis=1)
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
            mask = tf.one_hot(act_cust, depth=self.n_veh, dtype=tf.float32)
            mask = tf.stop_gradient(tf.transpose(mask, perm=[0, 2, 1]))
            mask = tf.concat([mask,mask_reb], axis = 2)
        else:
            mask = tf.one_hot(actions, depth=self.n_veh, dtype=tf.float32)
            mask = tf.stop_gradient(tf.transpose(mask, perm=[0, 2, 1]))

        rejects = tf.expand_dims(rejects, axis=2)
        mask = tf.concat([mask, tf.cast(rejects, dtype=tf.float32)], axis=2)

        return mask