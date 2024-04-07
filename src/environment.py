""" Episode initialization, transition and reward function to get next states and rewards

Gives state encodings as needed for the neural networks and uses hvs (hidden vehicles state) to track vehicles' state internally

Definition of states, actions, rewards:


Hidden vehicles state: array of shape (vehicles count, 8 (+1 if rebalancing enabled))
-------------------------------------------------------------------------------
Entries per vehicle:
    v: int in (0,...,nodes count-1)
    tau: int >= 0
    omega(r1): int >= 0 or -1 (-1 if no request assigned or already picked up)
    o(r1): int in (0,...,nodes count-1) or -1 (-1 if no request assigned)
    d(r1): int in (0,...,nodes count-1) or -1 (-1 if no request assigned)
    omega(r2): int >= 0 or -1 (-1 if no second request assigned)
    o(r2): int in (0,...,nodes count-1) or -1 (-1 if no request assigned)
    d(r2): int in (0,...,nodes count-1) or -1 (-1 if no request assigned)


State: Dictionary with entries "requests_state", "vehicles_state", "misc_state"
-------------------------------------------------------------------------------
Requests: padded tensor of shape (requests count (R_t), 7 (+1 if rebalancing enabled))
    Origin:
        horizontal idx, normalized to [0,1]
        vertical idx, normalized to [0,1]
    Destination:
        horizontal idx, normalized to [0,1]
        vertical idx, normalized to [0,1]
    Distance between origin and destination, divided by max. distance between any two nodes in graph (scalar in [0,1])
    Zone Distribution:
        number of vehicles with target in current zone, normalized to [0,1]
        number of vehicles with target in neighbouring zone, normalized to [0,1]
    [optional rebalancing_requests]
    if rebalancing enabled:
    request type: 1 if request is rebalancing request, 0 otherwise (flag)

Vehicles: tensor of shape (vehicles count, 6 (+1 if rebalancing enabled))
    Position: current node if no request assigned, otherwise destination of assigned request that will be served last
        horizontal idx, normalized to [0,1]
        vertical idx, normalized to [0,1]
    Time steps to reach position, divided by max. time between any two nodes in graph (scalar >= 0, usually < 1)
    Zone Distribution:
        number of vehicles with target in current location, normalized to [0,1]
        number of vehicles with target in neighbouring zones, normalized to [0,1]
    Number of assigned requests, divided by 2 (scalar in [0,1])
    if rebalancing enabled: 1 if vehicle is rebalancing, 0 otherwise (flag)

Misc: tensor of shape (3 (+1 if rebalancing enabled),)
    Time step, divided by no. of time steps in one episode (scalar in [0,1])
    Count of requests placed since start of current episode, divided by count of requests placed on average until current time step (scalar >= 0, usually close to 1)
    Sum over all vehicles of time steps to reach position, divided by number of vehicles x max. time between any two nodes in graph x 4 (scalar in [0,1])
    if rebalancing enabled: Number of rebalancing vehicles in previous time step, divided by vehicles count (scalar in [0,1])


Action: padded tensor of shape (R_t)
-------------------------------------------------------------------------------
Each request gets a single vehicle index if assigned, -1 if rejected


Reward: padded tensor of shape (vehicles count,)
-------------------------------------------------------------------------------
Each entry is a float representing the reward for the respective agent
"""

import copy
import random
import pandas as pd
import numpy as np
import tensorflow as tf


class Environment(object):
    def __init__(self, args):
        # intialize environment parameters
        self.episode_length = args["episode_length"]
        self.dt = args["time_step_size"]
        self.time_steps_count = tf.constant(int(self.episode_length/self.dt))
        self.n_veh = args["veh_count"]
        self.n_req_max = args["max_req_count"]
        self.discount = args["discount"]
        self.max_waiting_time = int(args["max_waiting_time"] / self.dt) # convert from seconds to time steps
        self.cost_parameter = args["cost_parameter"]
        self.data_dir = args["data_dir"]
        self.model = args["model"]

        # initialize rebalancing
        self.rebalancing_bool = args["rebalancing_bool"]
        self.rebalancing_mode = args["rebalancing_mode"]
        self.rebalancing_request_generation = args["rebalancing_request_generation"]
        if args["rebalancing_request_generation"] == "origin_destination_neighbours" and self.rebalancing_bool:
            self.rebalancing_neighbours_bool = tf.constant(True, dtype=tf.bool)
        else:
            self.rebalancing_neighbours_bool = tf.constant(False, dtype=tf.bool)

        # initialize dates
        training_dates = pd.read_csv(self.data_dir + '/training_dates.csv')
        validation_dates = pd.read_csv(self.data_dir + '/validation_dates.csv')
        test_dates = pd.read_csv(self.data_dir + '/test_dates.csv')
        self.training_dates = training_dates.training_dates.tolist()
        self.validation_dates = validation_dates.validation_dates.tolist()
        self.test_dates = test_dates.test_dates.tolist()
        self.remaining_training_dates = copy.deepcopy(self.training_dates)
        self.remaining_validation_dates = copy.deepcopy(self.validation_dates)
        
        #intitalize grid, zones and indices
        self.zones = pd.read_csv(self.data_dir + '/zones.csv', header=0, index_col=0)
        self.max_horizontal_idx = self.zones.horizontal_idx.max()
        self.max_vertical_idx = self.zones.vertical_idx.max()
        self.horizontal_idx_table = self.get_lookup_table(tf.constant(self.zones.index, dtype=tf.int32), tf.constant(self.zones.horizontal_idx, dtype=tf.int32))
        self.vertical_idx_table = self.get_lookup_table(tf.constant(self.zones.index, dtype=tf.int32), tf.constant(self.zones.vertical_idx, dtype=tf.int32))
        keys = self.get_keys(self.zones.horizontal_idx, self.zones.vertical_idx)
        self.zone_mapping_table = self.get_lookup_table(keys, tf.constant(self.zones.index, dtype=tf.int32)) # lookup table for mapping from horizontal/vertical idx to zone ID
        
        # initialize graph
        self.graph = pd.read_csv(self.data_dir + f'/graph(dt_{self.dt}s).csv', index_col=[0,1])
        self.graph.route = self.graph.route.apply(lambda x: list(map(int, x[1:-1].split(', '))))
        self.graph["next_node"] = np.array([self.graph.route.tolist()[i][1] for i in range(len(self.graph.index))])
        self.nodes_count = len(self.graph.index.unique(level=0))
        self.max_distance = self.graph.distance.max()
        self.min_distance = self.graph.distance.min()
        self.max_time = self.graph.travel_time.max()
        self.base_o = tf.math.ceil(self.n_veh/self.nodes_count)
        
        # initialize lookup tables
        distances = tf.constant(self.graph["distance"].reset_index(), tf.int32)
        travel_times = tf.constant(self.graph["travel_time"].reset_index(), tf.int32)
        fares = tf.constant(self.graph["fare"].reset_index(), tf.float32)
        next_nodes = tf.constant(self.graph["next_node"].reset_index(), tf.int32)
        keys = self.get_keys(distances[:,0], distances[:,1])
        self.distance_table = self.get_lookup_table(keys, distances[:,2])
        self.travel_time_table = self.get_lookup_table(keys, travel_times[:,2])
        self.fare_table = self.get_lookup_table(keys, fares[:,2])
        self.next_node_table = self.get_lookup_table(keys, next_nodes[:,2])
        
        self.avg_request_count = pd.read_csv(self.data_dir + f'/avg_request_count_per_timestep(dt_{self.dt}s).csv', header=None, names=["avg_count"])
        self.avg_request_count = tf.constant(self.avg_request_count.avg_count.tolist())

        self.neighbour_graph = self.get_neighbour_graph()

        #define dimensions and setup constants depending on rebalancing
        if self.rebalancing_bool:
            self.req_dim = 7 + 1
            self.veh_dim = 6 + 1
            self.hvs_dim = 8 + 1
            self.mis_dim = 3 + 1
            if self.rebalancing_neighbours_bool:
                self.init_rebalancing_requests()
                self.n_req_rebalancing = 6
                self.n_req_total = self.n_req_max
                self.n_req_total_const = self.n_req_max + self.n_req_rebalancing
            else:
                self.rebalancing_requests_tensor = self.init_rebalancing_requests()
                self.n_req_rebalancing = tf.shape(self.rebalancing_requests_tensor)[0].numpy()
                self.n_req_total = self.n_req_max + self.n_req_rebalancing
                self.n_req_total_const = self.n_req_total
        else:
            self.rebalancing_requests = []
            self.n_req_rebalancing = 0
            self.n_req_total_const = self.n_req_max
            self.n_req_total = self.n_req_max
            self.req_dim = 7
            self.veh_dim = 6
            self.hvs_dim = 8
            self.mis_dim = 3

        self.n_actions = self.n_req_total_const + 1 #add 1 for reject action

    # combine x and y to a key
    def get_keys(self, x, y):
        return tf.strings.join([tf.strings.as_string(x), tf.strings.as_string(y)], separator=',')

    # create hash table for lookup
    def get_lookup_table(self, keys, vals):
        return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, vals), default_value=-1)

    # reset environment to initial state
    def reset(self, validation=False, testing=False):
        self.time_step = tf.constant(0)
        self.cumulative_requests_count = tf.constant(0)
        
        # pick date and prepare trip data for episode
        if testing:
            self.date = self.test_dates[0]
            self.test_dates.remove(self.date)
        elif validation:
            self.date = random.choice(self.remaining_validation_dates)
            self.remaining_validation_dates.remove(self.date)
        else:
            if not self.remaining_training_dates:
                self.remaining_training_dates = copy.deepcopy(self.training_dates)
            self.date = random.choice(self.remaining_training_dates)
            self.remaining_training_dates.remove(self.date)
        
        # load trip data for the episode
        self.trip_data = pd.read_csv(self.data_dir + f'/trip_data/trips_{self.date}.csv', index_col=0)
        self.trip_data = self.trip_data.groupby(pd.cut(self.trip_data.pickup_time, np.arange(-1, self.episode_length, self.dt), labels=False)) # group per time step


        # vehicle distribution
        # during testing and validation, distribute vehicles evenly over zones
        if validation or testing:
            # average number of vehicles per zone
            no_veh_per_zone_all_zones = int(self.n_veh/self.nodes_count)
            remaining_vehicles = self.n_veh % self.nodes_count
            # distribute vehicles evenly over zones
            v = no_veh_per_zone_all_zones * [i for i in range(self.nodes_count)] + [i for i in range(remaining_vehicles)]
        # randomly distribute vehicles over zones when not training
        else:
            v = [random.randint(0, self.nodes_count-1) for i in range(self.n_veh)]


        # initialize hidden vehicles state
        if self.rebalancing_bool:
            tau = reb = [0 for i in range(self.n_veh)]
            o1 = o2 = d1 = d2 = omega1 = omega2 = [-1 for i in range(self.n_veh)]        
            self.hvs = np.array([i for i in zip(v, tau, omega1, o1, d1, omega2, o2, d2, reb)])
        else:
            tau = [0 for i in range(self.n_veh)]
            o1 = o2 = d1 = d2 = omega1 = omega2 = [-1 for i in range(self.n_veh)]        
            self.hvs = np.array([i for i in zip(v, tau, omega1, o1, d1, omega2, o2, d2)])
        store_hvs = tf.constant(self.hvs, tf.int32)
        vehicles_state = self.get_vehicles_state_tensor(store_hvs)

        # initiate requests and rebalancing requests state
        requests_state = self.get_requests_state_tensor()
        if self.rebalancing_bool: # append rebalancing requests to requests state
            if self.rebalancing_request_generation == "origin_destination_all":
                requests_state = tf.concat([requests_state, self.rebalancing_requests_tensor], axis=0)
        zone_distribution = self.get_zone_distribution(vehicles_state)
        requests_state = self.extend_requests_tensor_by_zone_distribution(requests_state, zone_distribution)

        # initialize misc state
        misc_state = self.get_misc_state_tensor(vehicles_state, self.time_step, self.cumulative_requests_count)

        #intitalize state dictionary
        state = {"requests_state": requests_state,
                 "vehicles_state": vehicles_state,
                 "misc_state": misc_state}

        return state, store_hvs

    # get next state and reward based on action
    def step(self, act, state):     
        self.time_step += 1
                
        # initialize reward        
        rew = np.zeros(self.n_veh)
        rew_training = np.zeros(self.n_veh)

        # get reward for accepted requests
        rew, rew_training = self.assign_accepted_requests(act, rew, rew_training, state)
        
        # get next transition state
        s_new = self.transition(tf.constant(self.hvs, tf.int32))
        self.hvs = s_new.numpy()

        # get next state
        next_requests_state = self.get_requests_state_tensor()
        next_vehicles_state = self.get_vehicles_state_tensor(s_new)
        next_misc_state = self.get_misc_state_tensor(next_vehicles_state, self.time_step, self.cumulative_requests_count)
        if self.rebalancing_bool: # append rebalancing requests to requests state
            if self.rebalancing_request_generation == "origin_destination_all":
                next_requests_state = tf.concat([next_requests_state, self.rebalancing_requests_tensor], axis=0)
        zone_distribution = self.get_zone_distribution(next_vehicles_state)
        next_requests_state = self.extend_requests_tensor_by_zone_distribution(next_requests_state, zone_distribution)
        
        # combine next state
        next_state = {"requests_state": next_requests_state,
                      "vehicles_state": next_vehicles_state,
                      "misc_state": next_misc_state}
        
        return next_state, tf.constant(rew, tf.float32), tf.constant(rew_training, tf.float32), s_new
    
    def assign_accepted_requests(self, act, rew, rew_training, state):
            # loop over all requests and assign them to vehicles
            for i in range(self.n_req_max):
                veh_ix = act[i].numpy()
                if veh_ix != -1:
                    # get origin and destination of request
                    o = self.requests.iloc[i,1]
                    d = self.requests.iloc[i,2]
                    target_position = tf.where(self.hvs[veh_ix,7] != -1, self.hvs[veh_ix,7], tf.where(self.hvs[veh_ix,4] != -1, self.hvs[veh_ix,4], self.hvs[veh_ix,0]))
                    
                    # calculate time needed to pick up customer to check if request will be served within max waiting time
                    time = self.hvs[veh_ix,1]
                    if self.hvs[veh_ix,3] == -1:
                        if self.hvs[veh_ix,0] != o:
                            time += self.graph.loc[(self.hvs[veh_ix,0], o), 'travel_time']
                    elif self.hvs[veh_ix,2] != -1:
                        if self.hvs[veh_ix,0] != self.hvs[veh_ix,3]:
                            time += self.graph.loc[(self.hvs[veh_ix,0], self.hvs[veh_ix,3]), 'travel_time']
                        time += self.graph.loc[(self.hvs[veh_ix,3], self.hvs[veh_ix,4]), 'travel_time']
                        if self.hvs[veh_ix,4] != o:
                            time += self.graph.loc[(self.hvs[veh_ix,4], o), 'travel_time']
                    else:
                        if self.hvs[veh_ix,0] != self.hvs[veh_ix,4]:
                            time += self.graph.loc[(self.hvs[veh_ix,0], self.hvs[veh_ix,4]), 'travel_time']
                        if self.hvs[veh_ix,4] != o:
                            time += self.graph.loc[(self.hvs[veh_ix,4], o), 'travel_time']
                    
                    # reward: revenue
                    if self.model == "dispatching_rebalancing": # condition for rebalancing experiment to be additionally in the zone or en route to the zone
                        reward_cond = time <= self.max_waiting_time and target_position == o
                    else:
                        reward_cond = time <= self.max_waiting_time
    
                    if reward_cond:
                        fare = self.graph.loc[(o,d),'fare']
                        fare_training = fare * self.discount ** time
                        rew[veh_ix] += fare 
                        rew_training[veh_ix] += fare_training 
                    
                    # reward: cost
                    cost = self.graph.loc[(o,d),'distance'] # distance from o to d of request
                    if self.hvs[veh_ix,3] == -1: # distance from current position to origin of new request if no other request assigned
                        if self.hvs[veh_ix,0] != o:
                            cost += self.graph.loc[(self.hvs[veh_ix,0], o), 'distance']
                    else: # distance from destination of other request to origin of new request if other request assigned
                        if self.hvs[veh_ix,4] != o:
                            cost += self.graph.loc[(self.hvs[veh_ix,4], o), 'distance']
                    cost *= self.cost_parameter
                    
                    rew[veh_ix] += cost 
                    rew_training[veh_ix] += cost 

                    route = "- {current}".format(current=self.hvs[veh_ix,0])
                    # assign request to first position if it is empty, second position otherwise
                    if self.hvs[veh_ix,3] == -1:
                        self.hvs[veh_ix,2] = 0 # omega1
                        self.hvs[veh_ix,3] = o # o1
                        self.hvs[veh_ix,4] = d # d1
                        request_count = 1
                        route += ",{o}>{d}".format(o=self.hvs[veh_ix,3],d=self.hvs[veh_ix,4])
                    else:
                        self.hvs[veh_ix,5] = 0 # omega2
                        self.hvs[veh_ix,6] = o # o2
                        self.hvs[veh_ix,7] = d # d2
                        request_count = 2
                        route += ",{o}>{d}".format(o=self.hvs[veh_ix,3],d=self.hvs[veh_ix,4]) + ",{o}>{d}".format(o=self.hvs[veh_ix,6],d=self.hvs[veh_ix,7])

                    # print("accepted request {rc} from {o} to {d} by vehicle {veh_ix} for reward {rew} {route}".format(rc=request_count, o=o, d=d, veh_ix=veh_ix, rew=rew[veh_ix], route=route))
            
            if self.rebalancing_bool:
                vehicles_state = state["vehicles_state"]
                zone_distribution = self.get_zone_distribution(vehicles_state)
                # loop over all rebalancing requests and assign them to vehicles
                for i in range(act.shape[0] - self.n_req_max):
                    veh_ix = act[(i+self.n_req_max)].numpy()
                    if veh_ix != -1: # get origin and destination of rebalancing request
                        o = i % self.nodes_count
                        d = o
                        
                        cost = 0
                        if self.hvs[veh_ix,3] == -1: # distance from current position to origin of new request if no other request assigned
                            if self.hvs[veh_ix,0] != o:
                                cost += self.graph.loc[(self.hvs[veh_ix,0], o), 'distance']
                        else: # distance from destination of other request to origin of new request if other request assigned
                            if self.hvs[veh_ix,4] != o:
                                cost += self.graph.loc[(self.hvs[veh_ix,4], o), 'distance']

                        if self.rebalancing_mode == "costs":  # normal costs for each ride
                            cost *= self.cost_parameter
                            fare = 0
                            fare_training = 0
                        elif self.rebalancing_mode == "reward_shaping":
                            fare = 0
                            cost *= self.cost_parameter

                            # calculate reward multiplier
                            origin = (vehicles_state[veh_ix,3]*self.n_veh).numpy()
                            destination = tf.cast(zone_distribution[i], dtype=tf.float32)
                            multiplier_d =  1- tf.minimum(destination/tf.math.floor(self.n_veh/self.nodes_count),1).numpy()
                            multiplier_o = tf.where(origin < self.base_o, 
                                                    -2*(self.base_o-origin+1)/self.base_o, 
                                                    tf.minimum(1, (origin-self.base_o)/self.base_o)*multiplier_d).numpy()
                            multiplier= multiplier_o/2 + multiplier_d*2
                            
                            # calculate reward based on the min distance
                            fare_training = self.min_distance * self.cost_parameter * -multiplier

                        else: 
                            ValueError("Rebalancing mode not implemented")

                        
                        rew[veh_ix] += fare + cost
                        rew_training[veh_ix] += fare_training + cost

                        route = "- {current}".format(current=self.hvs[veh_ix,0])
                        # assign request to first position as it is rebalancing
                        assert self.hvs[veh_ix,3] == -1
                        if self.hvs[veh_ix,3] == -1:
                            self.hvs[veh_ix,2] = 0 # omega1
                            self.hvs[veh_ix,3] = o # o1
                            self.hvs[veh_ix,4] = d # d1
                            request_count = 1
                            route += "-->{d}".format(d=self.hvs[veh_ix,4])
                        else:
                            assert True == False
                            self.hvs[veh_ix,5] = 0 # omega2
                            self.hvs[veh_ix,6] = o # o2
                            self.hvs[veh_ix,7] = d # d2
                            request_count = 2
                            route += "-->{d}".format(o=self.hvs[veh_ix,3],d=self.hvs[veh_ix,4]) + ",{o}>{d}".format(o=self.hvs[veh_ix,6],d=self.hvs[veh_ix,7])

                        # print("accept rebalancing request {rc} to {d} by vehicle {veh_ix} for training cost {rew} {route}".format(rc=request_count, o=o, d=d, veh_ix=veh_ix, rew=rew_training[veh_ix], route=route))

            return rew, rew_training

    @tf.function
    def transition(self, s):
        """get next node for each vehicle"""
        # set target node to destination if request is assigned and origin is not -1, otherwise if origin is not -1 set target to orin otherwise set target to current position
        target_node = tf.where(tf.reduce_all([s[:,2] == -1, s[:,3] != -1], axis=0), s[:,4], tf.where(s[:,3] != -1, s[:,3], s[:,0]))
        # if target node is reached set d to next y index otherwise set d to target node
        d = tf.where(s[:,0] == target_node, (s[:,0] + 1) % (self.nodes_count - 1), target_node)
        # get next node for d
        next_node = self.next_node_table[self.get_keys(s[:,0], d)]
        # get time to next node
        time_to_next_node = self.travel_time_table[self.get_keys(s[:,0], next_node)]
        
        # true if positioned is reached and vehicle is not at target node
        cond_new_node = tf.reduce_all([s[:,1] == 0, s[:,0] != target_node], axis=0)
        # true if origin is not -1 request assigned, vehicle is at origin node and (tau is either 0 or 1)
        cond_pickup = tf.reduce_all([s[:,3] != -1, s[:,2] != -1, s[:,0] == s[:,3], tf.reduce_any([s[:,1] == 1, s[:,1] == 0], axis=0)], axis=0)
        # true uf origin is not -1, request assigned, vehicle is at destination node and tau is 1
        cond_dropoff = tf.reduce_all([s[:,3] != -1, s[:,2] == -1, s[:,0] == s[:,4], s[:,1] == 1], axis=0)
        if self.rebalancing_bool:
            # true if rebalancing ends
            cond_rebalancing_end = tf.reduce_all([cond_pickup, s[:,3] == s[:,4]], axis=0)
            cond_rebalancing = tf.reduce_all([s[:,3] == s[:,4]], axis=0)
            cond_dropoff = cond_dropoff | cond_rebalancing_end
        else:
            cond_rebalancing = tf.constant(False, dtype=tf.bool)

        # true if origin 2 is current position and tau is 1
        cond_pickup_at_dropoff = tf.reduce_all([s[:,6] != -1, s[:,0] == s[:,6], s[:,1] == 1], axis=0)
        
        # get new v (0), tau (1), omega1 (2), o1 (3), d1 (4), omega2 (5), o2 (6), d2 (7)
        # v (position), tau (number of time steps left to reach this position), omega (-1 if assigned, ), o (origin), d (destination)
        
        # set new node to next node new node is reached otherwise keep current node
        s_new_zero = tf.where(cond_new_node, next_node, s[:,0])
        # if new node set the new time, otherwise decrease time by 1 if tau > 0
        s_new_one = tf.where(cond_new_node, time_to_next_node - 1, tf.where(s[:,1] > 0, s[:,1] - 1, s[:,1]))
        # increase omega 2 by one, if there was a dropoff some time ago
        s_new_five = tf.where(s[:,6] != -1, s[:,5] + 1, s[:,5])
        # set omega 1 to -1 if pickup, keep it at -1 if -1 and count +1 if no request assigned
        s_new_two = tf.where(cond_pickup, -1, tf.where(cond_dropoff, tf.where(cond_pickup_at_dropoff, -1, s_new_five), tf.where(s[:,2] != -1, s[:,2] + 1, s[:,2])))
        # set o1, d1 to previous o2, d2 when picked up
        s_new_three = tf.where(cond_dropoff, s[:,6], s[:,3])
        s_new_four = tf.where(cond_dropoff, s[:,7], s[:,4])
        # set omega2, o2, d2 to -1 when dropped off
        s_new_five = tf.where(cond_dropoff, -1, s_new_five)
        s_new_six = tf.where(cond_dropoff, -1, s[:,6])
        s_new_seven = tf.where(cond_dropoff, -1, s[:,7])

        if self.rebalancing_bool:
            s_new_eight = tf.where(cond_rebalancing, s[:,-1] + 1, 0)
            hvs = tf.stack((s_new_zero, s_new_one, s_new_two, s_new_three, s_new_four, s_new_five, s_new_six, s_new_seven, s_new_eight), axis=1)
        else: 
            hvs = tf.stack((s_new_zero, s_new_one, s_new_two, s_new_three, s_new_four, s_new_five, s_new_six, s_new_seven), axis=1)
        return hvs
    
    def get_requests_state_tensor(self):
        '''get tensor with requests for current time step'''
        try:
            self.requests = self.trip_data.get_group(self.time_step.numpy())
            # limit number of requests to n_req_max
            if len(self.requests.index) > self.n_req_max:
                self.requests = self.requests.iloc[:self.n_req_max,:]
            
            self.cumulative_requests_count += len(self.requests.index)

            # denormalize and convert to tensor
            origin_horizontal_idx = tf.expand_dims(tf.constant(self.requests.pickup_horizontal_idx/self.max_horizontal_idx, tf.float32), axis=1)
            origin_vertical_idx = tf.expand_dims(tf.constant(self.requests.pickup_vertical_idx/self.max_vertical_idx, tf.float32), axis=1)
            destination_horizontal_idx = tf.expand_dims(tf.constant(self.requests.dropoff_horizontal_idx/self.max_horizontal_idx, tf.float32), axis=1)
            destination_vertical_idx = tf.expand_dims(tf.constant(self.requests.dropoff_vertical_idx/self.max_vertical_idx, tf.float32), axis=1)
            distance = tf.expand_dims(tf.constant(self.requests.distance/self.max_distance, tf.float32), axis=1)

            if self.rebalancing_bool:
                # add rebalancing_flag = 0 (false) for all normal requests
                rebalancing_flag = tf.zeros([len(self.requests.index), 1])
                requests_tensor = tf.concat([origin_horizontal_idx,origin_vertical_idx,destination_horizontal_idx,destination_vertical_idx,distance, rebalancing_flag], axis=1)
            else:
                requests_tensor = tf.concat([origin_horizontal_idx,origin_vertical_idx,destination_horizontal_idx,destination_vertical_idx,distance], axis=1)

            # pad tensor with zeros if less than n_req_max requests
            paddings = tf.constant([[0, self.n_req_max - tf.shape(requests_tensor)[0].numpy()], [0, 0]])
            requests_tensor = tf.pad(requests_tensor, paddings, constant_values=-1)

        except KeyError: # if no requests for current time step
            self.requests = []
            requests_tensor = tf.ones([self.n_req_max, self.req_dim-2]) * -1
        
        return requests_tensor
    
    def init_rebalancing_requests(self):
        '''create rebalancing requests initially'''
        #create rebalancing requests once, but differentiate between modes
        graph = pd.DataFrame(columns=["origin_ID", "destination_ID", "distance", "travel_time", "fare", "route", "next_node"])
        for i in range(len(self.zones.index)): # create all zones once
            new_route = pd.DataFrame({"origin_ID":self.zones.index[i], "destination_ID":self.zones.index[i], "distance":0, "travel_time":0, "fare":0, "route":[self.zones.index[i]], "next_node":self.zones.index[i]})
            graph = pd.concat([graph, new_route], ignore_index=True)
        
        # set origin_ID and destination_ID as index
        graph = graph.set_index(["origin_ID", "destination_ID"])
        self.graph = pd.concat([self.graph, graph])

        if self.rebalancing_request_generation == "origin_destination_neighbours":
            rebalancing_requests_tensor = tf.ones([6, self.req_dim]) * -1 #placeholder

        else:
            # append idxs
            graph = pd.merge(graph, self.zones, left_on="origin_ID",right_index=True, how="left")
            graph = pd.merge(graph, self.zones, left_on="destination_ID",right_index=True, how="left", suffixes=("_origin","_destination"))
    
            # denormalize and convert to tensor
            origin_horizontal_idx = tf.expand_dims(tf.constant(graph.horizontal_idx_origin  /self.max_horizontal_idx, tf.float32), axis=1)
            origin_vertical_idx = tf.expand_dims(tf.constant(graph.vertical_idx_origin/self.max_vertical_idx, tf.float32), axis=1)
            destination_horizontal_idx = tf.expand_dims(tf.constant(graph.horizontal_idx_destination/self.max_horizontal_idx, tf.float32), axis=1)
            destination_vertical_idx = tf.expand_dims(tf.constant(graph.vertical_idx_destination/self.max_vertical_idx, tf.float32), axis=1)
            distance = tf.expand_dims(tf.constant(graph.distance/self.max_distance, tf.float32), axis=1)
            rebalancing_flag = tf.ones([len(graph.index), 1])

            rebalancing_requests_tensor = tf.concat([origin_horizontal_idx, origin_vertical_idx, destination_horizontal_idx, destination_vertical_idx, distance, rebalancing_flag], axis=1)
            
        return rebalancing_requests_tensor

    @tf.function
    def get_rebalancing_requests(self, vehicles_state):
        '''get rebalancing requests for current time step'''
        # create rebalancing requests every time step based on the neighbourh graph
        veh_horizontal = tf.cast(tf.math.round(vehicles_state[:,:,0] * self.max_horizontal_idx), tf.int32)
        veh_vertical = tf.cast(tf.math.round(vehicles_state[:,:,1] * self.max_vertical_idx), tf.int32)
        veh_location = self.zone_mapping_table[self.get_keys(veh_horizontal, veh_vertical)]
        graph = tf.repeat(self.neighbour_graph, vehicles_state.shape[0] , axis=0)
        rebalancing_requests_tensor = tf.gather(graph, veh_location, axis=1, batch_dims=1)
        rebalancing_requests_tensor = tf.cast(rebalancing_requests_tensor, tf.float32)
        zone_distribution = self.get_zone_distribution_rebalancing(vehicles_state)
        rebalancing_requests_tensor = self.extend_rebalancing_requests_tensor_by_zone_distribution(rebalancing_requests_tensor, zone_distribution)

        return rebalancing_requests_tensor
    
    def get_neighbour_graph(self):
        '''create a tensor with shape [zones, neighbours, req_dim]'''
        # setup indice changes
        change_tensor = tf.constant([[1,1],[2,0],[1,-1],[-1,-1],[-2,0],[-1,1]])
        max_tensor = tf.constant([[self.max_horizontal_idx, self.max_vertical_idx]])
        max_tensor = tf.repeat(max_tensor, 6, axis=0)

        neighbours = self.zones.iloc[:,4:]
        indices = self.zones.iloc[:,2:4]
        neighbours = tf.constant(neighbours, tf.bool)
        indices = tf.constant(indices, tf.int32)

        # get neighbours through change indices
        rebalancing_list = []
        for i in self.zones.index:
            neigbh_bool = tf.expand_dims(neighbours[i,:], axis=1)
            index = tf.repeat(tf.expand_dims(indices[i,:], axis=0), 6, axis=0)
            zones = (index + change_tensor)/max_tensor
            rebalancing_zones = tf.concat([zones, zones, tf.zeros([6, 1], dtype=tf.dtypes.double), tf.ones([6, 1], dtype=tf.dtypes.double)], axis=1)
            rebalancing_zones = tf.where(neigbh_bool, rebalancing_zones, -1)
            rebalancing_list.append(rebalancing_zones)

        neighbours_graph = tf.stack(rebalancing_list)
        neighbours_graph = tf.expand_dims(neighbours_graph, axis=0)

        return neighbours_graph


    @tf.function
    def get_vehicles_state_tensor(self, s):
        '''get tensor with vehicles state for current time step'''
        # when vehicle has a destination 1 or 2, set destionation as position, otherwise set current node as position
        position = tf.where(s[:,7] != -1, s[:,7], tf.where(s[:,4] != -1, s[:,4], s[:,0]))
        # denormalize position
        position_horizontal_idx = tf.expand_dims(tf.cast(self.horizontal_idx_table[position] / self.max_horizontal_idx, tf.float32), axis=1)
        position_vertical_idx = tf.expand_dims(tf.cast(self.vertical_idx_table[position] / self.max_vertical_idx, tf.float32), axis=1)

        #steps = tau
        steps_to_position = s[:,1]
        # if origin is -1 origni is 0 else origin is origin
        o = tf.where(s[:,3] == -1, 0, s[:,3])
        # if destination is -1 destination is 1 else destination is destination
        d = tf.where(s[:,4] == -1, 1, s[:,4])
        #if desination or origin is current position, set to next node, otherwise keep curren o / d
        idx1 = tf.where(d == s[:,0], (s[:,0] + 1) % (self.nodes_count - 1), d)
        idx2 = tf.where(o == s[:,0], (s[:,0] + 1) % (self.nodes_count - 1), o)
        # steps to position
        # always: tau, if first request exists + ...
        #  ... if already picked up: time from node to destination
        #  ... otherwise: time from node to origin + time from origin to destination
        steps_to_position += tf.where(s[:,3] != -1,
                                      tf.where(s[:,2] == -1,
                                               tf.where(s[:,0] != s[:,4], self.travel_time_table[self.get_keys(s[:,0], idx1)], 0),
                                               tf.where(s[:,0] != s[:,3], self.travel_time_table[self.get_keys(s[:,0], idx2)], 0) + self.travel_time_table[self.get_keys(o, d)]),
                                      0)
        #set o2 d2 to 0 and 1 if o2 and d2 are -1
        o2 = tf.where(s[:,6] == -1, 0, s[:,6])
        d2 = tf.where(s[:,7] == -1, 1, s[:,7])
        #get next index
        idx3 = tf.where(d == o2, (o2 + 1) % (self.nodes_count - 1), d)
        #  if second request exists:
        #    + time from destination of first request to origin of second request
        #    + time from origin to destination of second request
        steps_to_position += tf.where(s[:,6] != -1,
                                      tf.where(s[:,4] != s[:,6], self.travel_time_table[self.get_keys(idx3, o2)], 0) + self.travel_time_table[self.get_keys(o2, d2)],
                                      0)
        # normalize steps to position
        steps_to_position = steps_to_position / self.max_time
        # set new data type
        steps_to_position = tf.expand_dims(tf.cast(steps_to_position, tf.float32), axis=1)
        
        # count assigned requests
        count_assigned_requests = tf.expand_dims(tf.where(s[:,7] != -1, 1., tf.where(s[:,4] != -1, 0.5, 0.)), axis=1)
        
        # get location and neighbours distribution of vehicles
        location = s[:,0]
        #count number of vehilces in each zone
        zone_distribution = tf.math.bincount(location, minlength=self.nodes_count+1, maxlength=self.nodes_count)
        location_distribution = tf.gather(zone_distribution, tf.math.abs(location))
        location_distribution = tf.expand_dims(tf.cast(location_distribution / self.n_veh, tf.float32), axis=1)
        #create a graph of neighbours for each vehicle
        graph = tf.repeat(self.neighbour_graph, location.shape[0] , axis=0)
        location_neighbours = tf.gather(graph, location, axis=1, batch_dims=1)
        neighbours_horizontal = tf.cast(tf.math.round(location_neighbours[:,:,0] * self.max_horizontal_idx), tf.int32)
        neighbours_vertical = tf.cast(tf.math.round(location_neighbours[:,:,1] * self.max_vertical_idx), tf.int32)
        neighbours = self.zone_mapping_table[self.get_keys(neighbours_horizontal, neighbours_vertical)]
        #check if neighbour exists and get distribution
        empty_neighbours = tf.cast(neighbours == -1, tf.bool)
        zone_distribution = tf.expand_dims(zone_distribution, axis=0)
        neighbours_distribution = tf.gather(zone_distribution, tf.math.abs(neighbours), axis=1)
        neighbours_distribution = tf.where(empty_neighbours, tf.zeros_like(neighbours_distribution), neighbours_distribution)
        neighbours_distribution = tf.reduce_sum(neighbours_distribution, axis=2)
        neighbours_distribution = tf.cast(neighbours_distribution / self.n_veh, tf.float32)
        neighbours_distribution = tf.expand_dims(tf.squeeze(neighbours_distribution, axis=0), axis=1)
        
        if self.rebalancing_bool:
            rebalancing = tf.expand_dims(tf.cast(s[:,-1], tf.float32), axis=1)  / self.max_time
            state = tf.concat([position_horizontal_idx, position_vertical_idx, steps_to_position, location_distribution, neighbours_distribution, count_assigned_requests, rebalancing], axis=1)
        else:
            state = tf.concat([position_horizontal_idx, position_vertical_idx, steps_to_position, location_distribution, neighbours_distribution, count_assigned_requests], axis=1)
        
        return state

    @tf.function
    def get_misc_state_tensor(self, vehicles_state, time_step, cumulative_requests_count):
        '''get tensor with misc state for current time step'''
        t = tf.cast(time_step / self.time_steps_count, tf.float32)
        
        if time_step == self.time_steps_count:
            r = tf.cast(cumulative_requests_count, tf.float32) / self.avg_request_count[time_step-1]
        else:
            r = tf.cast(cumulative_requests_count, tf.float32) / self.avg_request_count[time_step]
        
        tp = tf.reduce_sum(vehicles_state[:,2]) / (self.n_veh * 4)

        if self.rebalancing_bool:
            reb = tf.where(vehicles_state[:,-1] > 0, 1., 0.)
            reb = tf.reduce_sum(reb) / self.n_veh
            tensor =  tf.stack([t,r,tp,reb])
        else:
            tensor =  tf.stack([t,r,tp])
        return tensor

    @tf.function
    def get_flag_within_max_waiting_time(self, vehicles_state, hvs, requests_state):
        '''compute time it takes until request will be picked up for each request/vehicle combination if it is assigned to the respective vehicle and 
        based on this, get flag if request would be served within the maximum waiting time (used by actor and critic to compute additional feature)'''
        hvs = tf.cast(hvs, tf.int32)
        
        # get time to location for each vehicle
        time_to_location = tf.cast(tf.math.round(vehicles_state[:,:,2] * self.max_time), tf.int32)
        time_to_location = tf.expand_dims(time_to_location, axis=1)
        time_to_location = tf.tile(time_to_location, multiples=tf.constant([1,self.n_req_total,1]))
        time_to_location = tf.transpose(time_to_location, perm=[0,2,1])
        # get location of each vehicle
        location = tf.where(hvs[:,:,7] != -1, hvs[:,:,7], tf.where(hvs[:,:,4] != -1, hvs[:,:,4], hvs[:,:,0]))
        location = tf.expand_dims(location, axis=1)
        location = tf.tile(location, multiples=tf.constant([1,self.n_req_total,1]))
        location = tf.transpose(location, perm=[0,2,1])
        # get pickup location for each request
        pickup_horizontal = tf.cast(tf.math.round(requests_state[:,:,0] * self.max_horizontal_idx), tf.int32)
        pickup_vertical = tf.cast(tf.math.round(requests_state[:,:,1] * self.max_vertical_idx), tf.int32)
        pickup = self.zone_mapping_table[self.get_keys(pickup_horizontal, pickup_vertical)]
        
        # get time from location to pickup
        pickup = tf.expand_dims(pickup, axis=1)
        pickup = tf.repeat(pickup, repeats=self.n_veh, axis=1)
        idx = tf.where(location == pickup, (pickup + 1) % (self.nodes_count - 1), location)
        time_location_to_pickup = tf.where(location != pickup, self.travel_time_table[self.get_keys(idx, pickup)], 0)
        
        # check if time to pickup is within max waiting time
        time_to_pickup = time_to_location + time_location_to_pickup
        within_max_waiting_time = tf.cast(time_to_pickup <= self.max_waiting_time, tf.float16)
        extra_zeros = 1
        if self.rebalancing_bool:
            rebalancing = tf.expand_dims(tf.cast(requests_state[:,:,5], tf.float16), axis=1)
            rebalancing = tf.repeat(rebalancing, repeats=self.n_veh, axis=1)
            within_max_waiting_time = tf.where(rebalancing == 1, tf.zeros_like(within_max_waiting_time), within_max_waiting_time)
            if self.rebalancing_neighbours_bool:
                extra_zeros = 7

        within_max_waiting_time = tf.concat([within_max_waiting_time, tf.zeros([within_max_waiting_time.shape[0], self.n_veh, extra_zeros], dtype=tf.float16)], axis=2)
        return tf.expand_dims(within_max_waiting_time, axis=3)

    @tf.function
    def get_approach_dist_norm(self, vehicles_state, requests_state):
        '''calculate the normalized approach distance for every vehicle-request combination'''
        # get final location after handling all open requests in vehicle_buffer
        veh_horizontal = tf.cast(tf.math.round(vehicles_state[:,:,0] * self.max_horizontal_idx), tf.int32)
        veh_vertical = tf.cast(tf.math.round(vehicles_state[:,:,1] * self.max_vertical_idx), tf.int32)
        veh_location = self.zone_mapping_table[self.get_keys(veh_horizontal, veh_vertical)]
        veh_location = tf.expand_dims(veh_location, axis=1)
        veh_location = tf.tile(veh_location, multiples=tf.constant([1,self.n_req_total,1]))
        veh_location = tf.transpose(veh_location, perm=[0,2,1])

        # get pickup location for each request
        pickup_horizontal = tf.cast(tf.math.round(requests_state[:,:,0] * self.max_horizontal_idx), tf.int32)
        pickup_vertical = tf.cast(tf.math.round(requests_state[:,:,1] * self.max_vertical_idx), tf.int32)
        pickup = self.zone_mapping_table[self.get_keys(pickup_horizontal, pickup_vertical)]
        pickup = tf.expand_dims(pickup, axis=1)
        pickup = tf.repeat(pickup, repeats=self.n_veh, axis=1)

        # get distance for approach only and normalize to be between [0,1]
        approach_dist = tf.where(veh_location != pickup, self.distance_table[self.get_keys(veh_location, pickup)], 0)
        approach_dist /= self.max_distance
        approach_dist = tf.cast(approach_dist, dtype=tf.float16)
        if self.rebalancing_neighbours_bool:
            extra_zeros = 7
        else:
            extra_zeros = 1
       
        approach_dist = tf.concat([approach_dist, tf.zeros([approach_dist.shape[0], self.n_veh, extra_zeros], dtype=tf.float16)], axis=2)

        return tf.expand_dims(approach_dist, axis=3)

    @tf.function
    def get_zone_distribution_rebalancing(self, vehicle_state):
        '''get zone distribution for rebalancing requests'''
        position_horizontal = tf.cast(tf.math.round(vehicle_state[:,:,0] * self.max_horizontal_idx), tf.int32)
        poisition_vertical = tf.cast(tf.math.round(vehicle_state[:,:,1] * self.max_vertical_idx), tf.int32)
        position = self.zone_mapping_table[self.get_keys(position_horizontal, poisition_vertical)]
        # count number of vehicles in each zone
        zone_distribution = tf.math.bincount(position, minlength=self.nodes_count+1, maxlength=self.nodes_count, axis = -1)
        return zone_distribution

    @tf.function
    def get_zone_distribution(self, vehicle_state):
        '''get zone distribution for requests'''
        position_horizontal = tf.cast(tf.math.round(vehicle_state[:,0] * self.max_horizontal_idx), tf.int32)
        poisition_vertical = tf.cast(tf.math.round(vehicle_state[:,1] * self.max_vertical_idx), tf.int32)
        position = self.zone_mapping_table[self.get_keys(position_horizontal, poisition_vertical)]
        # count number of vehicles in each zone
        zone_distribution = tf.math.bincount(position, minlength=self.nodes_count+1, maxlength=self.nodes_count)
        return zone_distribution
        
    @tf.function
    def extend_rebalancing_requests_tensor_by_zone_distribution(self, requests_state, zone_distribution):
        zone_distribution = tf.expand_dims(zone_distribution, axis=1)
        empty_requests = tf.expand_dims(tf.reduce_sum(requests_state, axis=3) < 0, axis=1)
        destination_horizontal = tf.cast(tf.math.round(requests_state[:,:,:,0] * self.max_horizontal_idx), tf.int32)
        destination_vertical = tf.cast(tf.math.round(requests_state[:,:,:,1] * self.max_vertical_idx), tf.int32)
        destination = self.zone_mapping_table[self.get_keys(destination_horizontal, destination_vertical)]
        #use origin as key for the arg of the zone distribution where there is a request(origin and destination is zero), otherwise set to zero 
        destination_distribution = tf.gather(zone_distribution, tf.math.abs(destination), axis=2, batch_dims=1)
        # normalize origin distribution
        destination_distribution = tf.cast(destination_distribution / self.n_veh, tf.float32)
        destination_distribution = tf.where(empty_requests, tf.ones_like(destination_distribution) * -1, destination_distribution)

        #use destination as key for the arg of the zone distribution where there is a request(origin and destination is zero), otherwise set to zero
        destination_distribution = tf.expand_dims(tf.squeeze(destination_distribution, axis=1), axis=3)
        #create a graph of neighbours for each vehicle
        graph = tf.repeat(self.neighbour_graph, destination.shape[0] , axis=0)
        destination = tf.where(destination == -1, 0, destination)  # fix for MacOS as destination_neighbours cannot work with empty requests (gather cannot work with -1 as index)
        
        destination_neighbours = tf.gather(graph, destination, axis=1, batch_dims=1)
        neighbours_horizontal = tf.cast(tf.math.round(destination_neighbours[:,:,:,:,0] * self.max_horizontal_idx), tf.int32)
        neighbours_vertical = tf.cast(tf.math.round(destination_neighbours[:,:,:,:,1] * self.max_vertical_idx), tf.int32)
        neighbours = self.zone_mapping_table[self.get_keys(neighbours_horizontal, neighbours_vertical)]
        #check if neighbour exists and get distribution
        empty_neighbours = tf.cast(neighbours == -1, tf.bool)
        zone_distribution = tf.expand_dims(zone_distribution, axis=1)
        neighbours_distribution = tf.gather(zone_distribution, tf.math.abs(neighbours), axis=3, batch_dims=1)
        neighbours_distribution = tf.squeeze(tf.squeeze(neighbours_distribution, axis=1), axis=1)
        neighbours_distribution = tf.where(empty_neighbours, tf.zeros_like(neighbours_distribution), neighbours_distribution)
        neighbours_distribution = tf.reduce_sum(neighbours_distribution, axis=3)
        neighbours_distribution = tf.cast(neighbours_distribution / self.n_veh, tf.float32)
        neighbours_distribution = tf.where(tf.squeeze(empty_requests,axis=1), tf.ones_like(neighbours_distribution) * -1, neighbours_distribution)
        neighbours_distribution = tf.expand_dims(neighbours_distribution, axis=3)

        requests_state = tf.concat([requests_state, destination_distribution, neighbours_distribution], axis=3)
        return requests_state
    
    @tf.function
    def extend_requests_tensor_by_zone_distribution(self, requests_state, zone_distribution):
        empty_requests = tf.expand_dims(tf.reduce_sum(requests_state, axis=1) < 0, axis=1)
        destination_horizontal = tf.cast(tf.math.round(requests_state[:,2] * self.max_horizontal_idx), tf.int32)
        destination_vertical = tf.cast(tf.math.round(requests_state[:,3] * self.max_vertical_idx), tf.int32)
        destination = self.zone_mapping_table[self.get_keys(destination_horizontal, destination_vertical)]
        destination_distribution = tf.gather(zone_distribution, tf.math.abs(destination))
        destination_distribution = tf.expand_dims(tf.cast(destination_distribution / self.n_veh, tf.float32), axis=1)
        destination_distribution = tf.where(empty_requests, tf.ones_like(destination_distribution) * -1, destination_distribution)
        
        #create a graph of neighbours for each vehicle
        graph = tf.repeat(self.neighbour_graph, destination.shape[0] , axis=0)
        destination = tf.where(destination == -1, 0, destination)  # fix for MacOS as destination_neighbours cannot work with empty requests (gather cannot work with -1 as index)
        destination_neighbours = tf.gather(graph, destination, axis=1, batch_dims=1)
        neighbours_horizontal = tf.cast(tf.math.round(destination_neighbours[:,:,0] * self.max_horizontal_idx), tf.int32)
        neighbours_vertical = tf.cast(tf.math.round(destination_neighbours[:,:,1] * self.max_vertical_idx), tf.int32)
        neighbours = self.zone_mapping_table[self.get_keys(neighbours_horizontal, neighbours_vertical)]
        #check if neighbour exists and get distribution
        empty_neighbours = tf.cast(neighbours == -1, tf.bool)
        zone_distribution = tf.expand_dims(zone_distribution, axis=0)
        neighbours_distribution = tf.gather(zone_distribution, tf.math.abs(neighbours), axis=1)
        neighbours_distribution = tf.where(empty_neighbours, tf.zeros_like(neighbours_distribution), neighbours_distribution)
        neighbours_distribution = tf.reduce_sum(neighbours_distribution, axis=2)
        neighbours_distribution = tf.cast(neighbours_distribution / self.n_veh, tf.float32)
        neighbours_distribution = tf.expand_dims(tf.squeeze(neighbours_distribution, axis=0), axis=1)
        neighbours_distribution = tf.where(empty_requests, tf.ones_like(neighbours_distribution) * -1, neighbours_distribution)

        requests_state = tf.concat([requests_state, destination_distribution, neighbours_distribution], axis=1)
        return requests_state


