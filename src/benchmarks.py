"""Benchmark"""

import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
from environment import Environment

class Benchmark(tf.keras.Model):
    def __init__(self, args, env: Environment):
        super().__init__(name="Actor")

        self.env = env
        self.args = args

        self.n_veh = args["veh_count"]
        self.n_req_max = args["max_req_count"]
        self.n_req_total = self.env.n_req_total
        self.n_req_total_const = self.env.n_req_total_const
        self.n_req_rebalancing = self.env.n_req_rebalancing
        self.n_actions = self.n_req_total_const + 1 # added reject actions
        self.max_horizontal_idx = self.env.max_horizontal_idx
        self.max_vertical_idx = self.env.max_vertical_idx
        self.nodes_count = self.env.nodes_count

        self.rebalancing_bool = args["rebalancing_bool"]
        self.rebalancing_request_generation = args["rebalancing_request_generation"]
        if args["rebalancing_request_generation"] == "origin_destination_neighbours" and self.rebalancing_bool:
            self.rebalancing_neighbours_bool = tf.constant(True, dtype=tf.bool)
        else:
            self.rebalancing_neighbours_bool = tf.constant(False, dtype=tf.bool)
        self.cost_parameter = abs(args["cost_parameter"])
        self.max_time = self.env.max_time
        self.model = args["model"]
        

    
    def get_action(self, state, hvs, rebalancing_requests, test):  # TODO clean up or check whether it works
        """returns action for agent after computing probabilities"""
        state, request_masks = self.get_action_body(state, rebalancing_requests)
        hvs = tf.expand_dims(hvs, axis=0)
        probs = self.compute_greedy(state, hvs, rebalancing_requests)
        act = self.post_process(probs, test, hvs, request_masks["l1"], state["vehicles_state"], state["requests_state"], rebalancing_requests)
        return tf.squeeze(act, axis=[0])
    
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

    #@tf.function
    def compute_greedy(self, state, hvs, rebalancing_requests):
        requests_state = state["requests_state"]
        vehicles_state = state["vehicles_state"]
        # get distance of the request
        distance1 = requests_state[:,:,4] * self.env.max_distance
        distance1 = tf.expand_dims(distance1, axis=1)
        distance1 = tf.repeat(distance1, repeats=self.n_veh, axis=1)
        # check if request is within max waiting time
        within_max_waiting_time = self.env.get_flag_within_max_waiting_time(vehicles_state, hvs, requests_state)
        within_max_waiting_time = tf.squeeze(within_max_waiting_time, axis=3)
        if self.rebalancing_neighbours_bool:
            within_max_waiting_time, _ = tf.split(within_max_waiting_time, num_or_size_splits=[self.n_req_max,-1], axis=2)
        else:
            within_max_waiting_time, _ = tf.split(within_max_waiting_time, num_or_size_splits=[-1,1], axis=2)
        # get location of each vehicle (which is the position = end of last request)
        location_raw = tf.where(hvs[:,:,7] != -1, hvs[:,:,7], tf.where(hvs[:,:,4] != -1, hvs[:,:,4], hvs[:,:,0]))
        location_raw = tf.expand_dims(location_raw, axis=1)
        location = tf.tile(location_raw, multiples=tf.constant([1,self.n_req_total,1]))
        location = tf.transpose(location, perm=[0,2,1])
        
        # get pickup location for each request
        pickup_horizontal = tf.cast(tf.math.round(requests_state[:,:,0] * self.env.max_horizontal_idx), tf.int32)
        pickup_vertical = tf.cast(tf.math.round(requests_state[:,:,1] * self.env.max_vertical_idx), tf.int32)
        pickup = self.env.zone_mapping_table[self.env.get_keys(pickup_horizontal, pickup_vertical)]
        # get time from location to pickup
        pickup = tf.expand_dims(pickup, axis=1)
        pickup = tf.repeat(pickup, repeats=self.n_veh, axis=1)
        idx = tf.where(location == pickup, (pickup + 1) % (self.env.nodes_count - 1), location)
        
        # get destination location for each request
        destination_horizontal = tf.cast(tf.math.round(requests_state[:,:,2] * self.env.max_horizontal_idx), tf.int32)
        destination_vertical = tf.cast(tf.math.round(requests_state[:,:,3] * self.env.max_vertical_idx), tf.int32)
        destination = self.env.zone_mapping_table[self.env.get_keys(destination_horizontal, destination_vertical)]
        destination = tf.repeat(destination, repeats=self.env.n_veh, axis=1)
        # get time from location to pickup
        destination = tf.expand_dims(destination, axis=1)
        destination = tf.repeat(destination, repeats=self.n_veh, axis=1)

        if self.rebalancing_request_generation == "origin_destination_neighbours" and self.rebalancing_bool:
            destination_distribution = rebalancing_requests[:,:,:,6]
        else:
            destination_distribution = requests_state[:,:,6]
            destination_distribution = tf.expand_dims(destination_distribution, axis=1)
            destination_distribution = tf.repeat(destination_distribution, repeats=self.n_veh, axis=1)

        zone_distribution = self.env.get_zone_distribution_rebalancing(vehicles_state)
        origin_distribution = tf.gather(zone_distribution, tf.math.abs(location), axis=1, batch_dims=1)
        origin_distribution = tf.cast(origin_distribution / self.n_veh, tf.float32)


        #get the distance from the vehicle to the pickup location
        distance2 = tf.where(location != pickup, self.env.distance_table[self.env.get_keys(idx, pickup)], 0)

        # calculate the delta between the two distances to get the profibality
        cost_factor = (0.005-self.cost_parameter)/self.cost_parameter
        delta =  tf.cast(cost_factor*distance1, dtype=tf.float32) - tf.cast(distance2, dtype=tf.float32) #for our cost parameter
        probs_greedy = tf.where(delta <= 0, 
                tf.cast(0, dtype=tf.float32),
                tf.cast(0.75 + (0.24 * tf.cast(delta, dtype=tf.float32) / self.env.max_distance), dtype=tf.float32)
                )
      
        probs = tf.where(within_max_waiting_time==1, probs_greedy, 0)
        if self.model == "dispatching_rebalancing":
            probs =tf.where(distance2==0, probs, tf.cast(0, dtype=tf.float32))
        if self.rebalancing_bool:
            if self.rebalancing_request_generation == "origin_destination_neighbours":
                probs_greedy = probs
                location = tf.tile(location_raw, multiples=tf.constant([1,6,1]))
                location = tf.transpose(location, perm=[0,2,1])
                origin_distribution = tf.gather(zone_distribution, tf.math.abs(location), axis=1, batch_dims=1)
                origin_distribution = tf.cast(origin_distribution / self.n_veh, tf.float32)
                delta =  tf.ones([1,self.n_veh,6], dtype=tf.float32) * (self.env.max_distance / self.nodes_count)
            else:
                probs_greedy , _ = tf.split(probs, num_or_size_splits=[self.n_req_max, self.n_req_total_const-self.n_req_max], axis=2)
                
            probs_reb = tf.where(origin_distribution*self.n_veh<=tf.math.ceil(self.n_veh/self.nodes_count),
                                tf.cast(0, dtype=tf.float32),
                                tf.where(destination_distribution*self.n_veh>=tf.math.floor(self.n_veh/self.nodes_count),
                                        tf.cast(0, dtype=tf.float32),
                                        tf.cast(0.5 + 0.25*(1-abs(delta) / self.env.max_distance), tf.float32)))
            if self.rebalancing_request_generation != "origin_destination_neighbours":

                _ , probs_reb = tf.split(probs_reb, num_or_size_splits=[self.n_req_max, self.n_req_total_const-self.n_req_max], axis=2)
            
            #append zeros for reject action
            probs = tf.concat([probs_greedy, probs_reb, tf.ones([1,self.n_veh,1])/2], axis=2)
        else:
            probs = tf.concat([probs, tf.ones([1,self.n_veh,1])/2], axis=2)

        return probs
    

    def post_process(self, probs, test, hvs, request_mask_l, vehicles_state, requests_state, rebalancing_requests, random=tf.constant(False)):
        batch_size = tf.shape(probs)[0]

        probs = self.mask_all_probs(probs, hvs, request_mask_l, requests_state, vehicles_state)

        #get actionn
        if self.rebalancing_bool:
            act = self.get_action_from_probs(probs, rebalancing_requests)
        else:
            act = probs

        act = act.numpy()
        # parallelize the weighted matching, which is done by linear sum assignment for each batch
        action_list = Parallel(n_jobs=2, prefer="threads")(delayed(self.weighted_matching)(act[i,:,:]) for i in range(batch_size))
        act = tf.constant(action_list)
        matched_actions, _ = tf.split(act, num_or_size_splits=[act.shape[1]-self.n_veh, self.n_veh], axis=1)
        return matched_actions

    @tf.function
    def mask_all_probs(self, probs, hvs, request_mask_l, requests_state, vehicles_state):
        '''mask all the probabilities according to the request mask'''
        request_mask , _ = tf.split(request_mask_l, num_or_size_splits=[-1, 1], axis=2)
        if self.rebalancing_bool:
            # access the last n requests from the request stat
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
        ''' append one state with zeros as reject state'''
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
            extended_actions = tf.concat([sampled_actions, rejects], axis=2)
        return extended_actions
    
    
    def weighted_matching(self, x):
        '''weighted matching according to linear sum assignment'''
        matched_veh, matched_req = linear_sum_assignment(x, maximize=True)  # weighted matching

        # correct matching decision to reject decision if weight is zero
        matched_weights = x[matched_veh, matched_req]
        matched_veh = np.where(matched_weights == 0., -1, matched_veh)  # if weight is zero, correct matching decision to reject decision
        
        if self.rebalancing_request_generation == "origin_destination_neighbours" and self.rebalancing_bool:
            action = -np.ones(self.n_req_max + self.nodes_count + self.n_veh, int)
        else:
            action = -np.ones(self.env.n_req_total + self.n_veh, int)

        action[matched_req] = matched_veh
        
        return action