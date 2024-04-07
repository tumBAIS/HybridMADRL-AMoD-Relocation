import tensorflow as tf
from environment import Environment


class ReplayBuffer(object):
    def __init__(self, size, normalized_rews, env:Environment):
        self.normalized_rews = normalized_rews
        self.n_veh = env.n_veh
        self.n_req_total = env.n_req_total
        self.req_dim = env.req_dim
        self.hvs_dim = env.hvs_dim
        self.veh_dim = env.veh_dim
        self.mis_dim = env.mis_dim
        self.rebalancing_bool = env.rebalancing_bool
        self.rebalancing_neighbours_bool = env.rebalancing_neighbours_bool
        self.rebalancing_request_generation = env.rebalancing_request_generation
        self.discount = env.discount
        self.n_req_max = env.n_req_max
        self.n_req_reb = env.n_req_rebalancing
        self.rebalancing_mode = env.rebalancing_mode
        self.env = env
        self.max_horizontal_idx = env.max_horizontal_idx
        self.max_vertical_idx = env.max_vertical_idx
        self.nodes_count = env.nodes_count
        
        # initialize all variables as size x + no. + shape
        self.requests_states = tf.Variable(tf.zeros((size,) + (self.n_req_total,) + (self.req_dim,), dtype=tf.float32))
        self.vehicles_states = tf.Variable(tf.zeros((size,) + (self.n_veh,) + (self.veh_dim,), dtype=tf.float32))
        self.misc_states = tf.Variable(tf.zeros((size,) + (self.mis_dim,), dtype=tf.float32))
        self.hvses = tf.Variable(tf.zeros((size,) + (self.n_veh,) + (self.hvs_dim,), dtype=tf.int32))
        self.rejects = tf.Variable(tf.zeros((size,) + (self.n_veh,), dtype=tf.int32))
        self.rews = tf.Variable(tf.zeros((size,) + (self.n_veh,), dtype=tf.float32))
        self.next_requests_states = tf.Variable(tf.zeros((size,) + (self.n_req_total,) + (self.req_dim,), dtype=tf.float32))
        self.next_vehicles_states = tf.Variable(tf.zeros((size,) + (self.n_veh,) + (self.veh_dim,), dtype=tf.float32))
        self.next_misc_states = tf.Variable(tf.zeros((size,) + (self.mis_dim,), dtype=tf.float32))
        self.next_hvses = tf.Variable(tf.zeros((size,) + (self.n_veh,) + (self.hvs_dim,), dtype=tf.int32))
        if self.rebalancing_neighbours_bool:
            self.acts = tf.Variable(tf.zeros((size,) + ((self.n_req_max + self.env.nodes_count),), dtype=tf.int32))
        else:
            self.acts = tf.Variable(tf.zeros((size,) + (self.n_req_total,), dtype=tf.int32))

        # if normalized_rews is true, then initialize masks for the rewards
        if self.normalized_rews:
            self.masks = tf.Variable(tf.zeros((size,) + (self.n_veh,), dtype=tf.float32))
        
        self.maxsize = size
        self.size = tf.Variable(0, dtype=tf.int32)
        self.next_idx = tf.Variable(0, dtype=tf.int32)

    @tf.function
    def add(self, obs, hvs, act, rejects, rew, next_obs, next_hvs, mask=None):
        """add a new experience to the replay buffer"""
        # at the next index, update all the variables with the new experience
        self.requests_states.scatter_nd_update([[self.next_idx]], [obs["requests_state"]])
        self.vehicles_states.scatter_nd_update([[self.next_idx]], [obs["vehicles_state"]])
        self.misc_states.scatter_nd_update([[self.next_idx]], [obs["misc_state"]])
        self.hvses.scatter_nd_update([[self.next_idx]], [tf.cast(hvs, tf.int32)])
        self.acts.scatter_nd_update([[self.next_idx]], [act])
        self.rejects.scatter_nd_update([[self.next_idx]], [rejects])
        self.rews.scatter_nd_update([[self.next_idx]], [rew])
        self.next_requests_states.scatter_nd_update([[self.next_idx]], [next_obs["requests_state"]])
        self.next_vehicles_states.scatter_nd_update([[self.next_idx]], [next_obs["vehicles_state"]])
        self.next_misc_states.scatter_nd_update([[self.next_idx]], [next_obs["misc_state"]])
        self.next_hvses.scatter_nd_update([[self.next_idx]], [tf.cast(next_hvs, tf.int32)])
        if self.normalized_rews:
            self.masks.scatter_nd_update([[self.next_idx]], [mask])
        
        # update the size and next index
        self.size.assign(tf.math.minimum(self.size + 1, self.maxsize))
        self.next_idx.assign((self.next_idx + 1) % self.maxsize)

    @tf.function
    def sample(self, batch_size):
        """sample a batch of experiences from the replay buffer"""
        idxes = tf.random.uniform((batch_size,), maxval=self.size, dtype=tf.int32)

        # gather the experiences at the sampled indices
        obses_req = tf.gather(self.requests_states, idxes)
        obses_veh = tf.gather(self.vehicles_states, idxes)
        obses_misc = tf.gather(self.misc_states, idxes)
        hvses = tf.gather(self.hvses, idxes)
        acts = tf.gather(self.acts, idxes)
        rejects = tf.gather(self.rejects, idxes)
        rews = tf.gather(self.rews, idxes)
        next_obses_req = tf.gather(self.next_requests_states, idxes)
        next_obses_veh = tf.gather(self.next_vehicles_states, idxes)
        next_obses_misc = tf.gather(self.next_misc_states, idxes)
        next_hvses = tf.gather(self.next_hvses, idxes)
  
        # if normalized_rews is true, then normalize the rewards
        if self.normalized_rews:
            flat_rews = tf.reshape(self.rews, [-1])
            flat_masks = tf.reshape(tf.cast(self.masks, dtype=tf.bool), [-1])  # flatten the masks, turn number into boolean --> 1 is True
            gather_indices = tf.where(flat_masks)  # get indices of non-zero masks
            check = (tf.rank(gather_indices) == 2)  # dont know what a rank in tensor is
            tf.debugging.Assert(check, [tf.shape(gather_indices)])
            masked_rews = tf.gather(flat_rews, tf.squeeze(gather_indices, [1]))  # get the rewards at the indices of non-zero masks
            std = tf.math.reduce_std(masked_rews)
            rews /= std
        
        # return the experiences
        obses = {"requests_state": obses_req,
                 "vehicles_state": obses_veh,
                 "misc_state": obses_misc}
        next_obses = {"requests_state": next_obses_req,
                      "vehicles_state": next_obses_veh,
                      "misc_state": next_obses_misc}

        return obses, hvses, acts, rejects, rews, next_obses, next_hvses

