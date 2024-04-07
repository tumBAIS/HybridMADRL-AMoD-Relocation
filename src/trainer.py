"""Training loop incl. validation and testing"""

import os
import copy
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
from replay_buffer import ReplayBuffer
from environment import Environment
from sac_discrete import SACDiscrete
from pathlib import Path
from benchmarks import Benchmark

class Trainer:
    def __init__(self, policy: SACDiscrete, env: Environment, args):
        self.policy = policy
        self.env = env

        # initialize training parameters
        self.episode_max_steps = int(args["episode_length"] / args["time_step_size"])  # no. of steps per episode
        self.n_veh = args["veh_count"]
        self.n_req_max = args["max_req_count"]
        self.n_req_total = self.env.n_req_total
        self.n_req_total_const = self.env.n_req_total_const
        self.n_req_dim = self.env.req_dim
        self.benchmark_bool = args["benchmark"]
        self.max_steps = args["max_steps"]
        self.min_steps = args["min_steps"]
        self.random_steps = args["random_steps"]
        self.update_interval = args["update_interval"]
        self.validation_interval = args["validation_interval"]
        self.tracking_interval = args["tracking_interval"]
        self.rb_size = args["rb_size"]
        self.batch_size = args["batch_size"]
        self.normalized_rews = args["normalized_rews"]
        self.rebalancing_bool = tf.constant(args["rebalancing_bool"], dtype=tf.bool)
        self.rebalancing_request_generation = args["rebalancing_request_generation"]
        if args["rebalancing_request_generation"] == "origin_destination_neighbours" and self.rebalancing_bool:
            self.rebalancing_neighbours_bool = tf.constant(True, dtype=tf.bool)
        else:
            self.rebalancing_neighbours_bool = tf.constant(False, dtype=tf.bool)
        self.record_bool = args["record_bool"]

        self.data_dir = args["data_dir"]
        self.results_dir = args["results_dir"]
        self.model_dir = args["model_dir"]

        # initialize model saving and potentially restore saved model in checkpoints
        self.set_check_point(self.model_dir)

        self.log_dir = str(Path(self.results_dir).parent) + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + str(args["random_seed"])
        
        if self.record_bool:
            # prepare TensorBoard output
            self.writer = tf.summary.create_file_writer(self.log_dir, name=np.random.randint(0, 1000, 1))
            self.writer.set_as_default()

            # save arguments and environment variables of current run
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            with open(self.log_dir + '/args.txt', 'w') as f:
                f.write(str(args))
            with open(self.log_dir + '/environ.txt', 'w') as f:
                f.write(str(dict(os.environ)))
            tf.summary.text("hyperparams", self.args2tensorboardMD(args, key='Parameter', val='Value'), step=0)
        
        if self.benchmark_bool:
            self.benchmark = Benchmark(args, self.env)

    @staticmethod
    def args2tensorboardMD(d, key='Name', val='Value'):
        '''convert args to markdown to save nicely in tensorboard'''
        rows = [f'| {key} | {val} |']
        rows += ['|------|--------|']
        rows += [f'| {k} | {v} |' for k, v in d.items()]
        return "  \n".join(rows)

    def set_check_point(self, model_dir):
        """initialize model saving and potentially restore saved model"""
        self.checkpoint = tf.train.Checkpoint(policy=self.policy)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.results_dir,
                                                             max_to_keep=500)
        if model_dir is not None:
            assert os.path.isdir(model_dir)
            latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
            self.checkpoint.restore(latest_path_ckpt)

    def __call__(self):
        """initialize training loop"""
        total_steps = 0
        episode_steps = 0
        episode_reward = 0.
        self.validation_rewards = []

        self.replay_buffer = ReplayBuffer(self.rb_size, self.normalized_rews, self.env)

        state, hvs = self.env.reset()

        # training loop
        while total_steps < self.max_steps and not self.benchmark_bool:
            if (total_steps + 1) % 100 == 0:
                tf.print("Started step {} / {}".format(total_steps + 1, self.max_steps))

            with tf.profiler.experimental.Trace('train', step_num=total_steps, _r=1):
                # get rebalancing requests or placeholder if not neighbour
                if self.rebalancing_neighbours_bool:
                    rebalancing_requests = self.env.get_rebalancing_requests(tf.expand_dims(state["vehicles_state"], axis=0))
                elif self.rebalancing_request_generation == "origin_destination_all" and self.rebalancing_bool:
                    rebalancing_requests = tf.zeros([1,self.n_veh, self.n_req_total - self.n_req_max, 8])
                else:
                    rebalancing_requests = tf.zeros([1,self.n_veh, self.n_req_total - self.n_req_max, 7])
                
                # set action to -1 for all possible actions if no requests are present
                if tf.reduce_all(state["requests_state"] == tf.zeros([self.n_req_total, self.n_req_dim])) and not self.rebalancing_bool:
                    action = -tf.ones(self.n_req_total, tf.int32) 
                elif total_steps < self.random_steps:  
                    # choose random actions if in random interval
                    request_masks = self.policy.get_masks(tf.expand_dims(state["requests_state"], axis=0), rebalancing_requests)
                    action, rejects = self.policy.actor.get_random_action(state, hvs, request_masks, rebalancing_requests)
                else:  
                    # choose action according to policy
                    action, rejected_perc, rejects = self.policy.get_action(state, hvs, rebalancing_requests)
                    # track the rejected requests
                    if self.record_bool: tf.summary.scalar(name="Rejected Percentage", data=rejected_perc, step=total_steps)

                # get next state, reward and new hidden vehicle states
                next_state, reward, reward_training, next_hvs = self.env.step(action, state)
                
                # if all requests are empty - do not add transition to ReplayBuffer (no info for training)
                if ~tf.reduce_all(state["requests_state"] == tf.zeros([self.n_req_total, self.n_req_dim])):  # condition does not apply as there are always rebalancing actions available
                    # add normalized transition to replay buffer
                    if self.normalized_rews:
                        mask = tf.one_hot(action, depth=self.n_veh, dtype=tf.int32) # indices that can be used to get Q(s,a) for correct a from Q(s), which is a vector with Q(s,a) for all possible a
                        mask = tf.stop_gradient(tf.transpose(mask, perm=[1, 0]))
                        rew_mask = tf.reduce_sum(mask, axis=1)
                        self.replay_buffer.add(obs=state, hvs=hvs, act=action, rejects=rejects, rew=reward_training, next_obs=next_state,
                                          next_hvs=next_hvs, mask=rew_mask)
                    # add unnormalized transition to replay buffer
                    else:
                        self.replay_buffer.add(obs=state, hvs=hvs, act=action, rejects=rejects, rew=reward_training, next_obs=next_state,
                                          next_hvs=next_hvs)
                        
                # update state and hidden vehicle states
                state = next_state
                hvs = next_hvs
                total_steps += 1
                episode_steps += 1
                episode_reward += tf.reduce_sum(reward).numpy()

                if self.record_bool:
                    tf.summary.experimental.set_step(total_steps)

                # update policy only when min_steps have run and update_interval fits
                if total_steps >= self.min_steps and total_steps % self.update_interval == 0:
                    states, hvses, acts, rejs, rews, next_states, next_hvses = self.replay_buffer.sample(self.batch_size)
                    if self.record_bool:
                        with tf.summary.record_if(total_steps % self.tracking_interval == 0):
                            self.policy.train(states, hvses, acts, rejs, rews, next_states, next_hvses)
                    else:
                        self.policy.train(states, hvses, acts, rejs, rews, next_states, next_hvses)


            # validate policy
            if total_steps % self.validation_interval == 0:
                avg_validation_reward = self.validate_policy()
                if self.record_bool: tf.summary.scalar(name="avg_reward_per_validation_episode", data=avg_validation_reward)
                self.validation_rewards.append(avg_validation_reward)

                self.checkpoint_manager.save()

            # reset environment if episode is finished
            if episode_steps == self.episode_max_steps:
                if self.record_bool: tf.summary.scalar(name="training_return", data=episode_reward)
                episode_steps = 0
                episode_reward = 0.
                state, hvs= self.env.reset()

        # save summary
        if self.record_bool: tf.summary.flush()

        self.test_policy()
        tf.print("Finished")


    def validate_policy(self):
        """compute average reward per validation episode achieved by current policy"""
        validation_reward = 0.
        validation_episodes = pd.read_csv(self.data_dir + '/validation_dates.csv').validation_dates.tolist()
        
        for i in range(len(validation_episodes)):
            state, hvs = self.env.reset(validation=True)

            for j in range(self.episode_max_steps):
                # get rebalancing requests or placeholder if not neighbour generation
                if self.rebalancing_neighbours_bool:
                    rebalancing_requests = self.env.get_rebalancing_requests(tf.expand_dims(state["vehicles_state"], axis=0))
                elif self.rebalancing_request_generation == "origin_destination_all" and self.rebalancing_bool:
                    rebalancing_requests = tf.zeros([1,self.n_veh, self.n_req_total - self.n_req_max, 8])
                else:
                    rebalancing_requests = tf.zeros([1,self.n_veh, self.n_req_total - self.n_req_max, 7])
                # set action to -1 for all possible actions if no requests are present, reject all requests
                if tf.reduce_all(state["requests_state"] == tf.zeros([self.n_req_total, self.n_req_dim])):
                    action = -tf.ones(self.n_req_total, tf.int32)
                else:  # choose action according to policy
                    action, _, _ = self.policy.get_action(state, hvs, rebalancing_requests, test=tf.constant(True))

                next_state, reward, _, hvs = self.env.step(action, state)

                validation_reward += tf.reduce_sum(reward).numpy()

                state = next_state

        avg_validation_reward = validation_reward / len(validation_episodes)

        self.env.remaining_validation_dates = copy.deepcopy(
            self.env.validation_dates)  # reset list of remaining validation dates

        return avg_validation_reward

    def test_policy(self):
        """compute rewards per test episode with best policy"""
        # load best policy
        if not self.benchmark_bool:
            ckpt_id = np.argmax(self.validation_rewards) + 1
            self.checkpoint.restore(self.results_dir + f"/ckpt-{ckpt_id}")
            self.checkpoint = tf.train.Checkpoint(policy=self.policy)
            # save best policy in results directory
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.log_dir, max_to_keep=500)
            self.checkpoint_manager.save()

        test_dates = pd.read_csv(self.data_dir + '/test_dates.csv').test_dates.tolist()

        test_rewards = []
        test_accepted_requests = []
        test_accepted_rebalancing_requests = []
        test_rejected_requests = []

        # test policy for each test date
        for i in range(len(test_dates)):
            test_reward = 0
            accepted_requests = 0
            accepted_rebalancing_requests = 0
            rejected_requests = 0

            state, hvs = self.env.reset(testing=True)

            for j in range(self.episode_max_steps):
                if self.rebalancing_neighbours_bool:
                    rebalancing_requests = self.env.get_rebalancing_requests(tf.expand_dims(state["vehicles_state"], axis=0))
                elif self.rebalancing_request_generation == "origin_destination_all" and self.rebalancing_bool:
                    rebalancing_requests = tf.zeros([1,self.n_veh, self.n_req_total - self.n_req_max, 8])
                else:
                    rebalancing_requests = tf.zeros([1,self.n_veh, self.n_req_total - self.n_req_max, 7])

                if tf.reduce_all(state["requests_state"] == tf.zeros([self.n_req_total, self.n_req_dim])):
                    action = -tf.ones(self.n_req_total, tf.int32)
                else:
                    if not self.benchmark_bool:
                        action, _, _ = self.policy.get_action(state, hvs, rebalancing_requests, test=tf.constant(True))
                    else:
                        action = self.benchmark.get_action(state, hvs, rebalancing_requests, test=tf.constant(True))
                        
                #track the accepted and rejected requests
                act_req, act_reb = tf.split(tf.expand_dims(action, axis=0), [self.n_req_max, action.shape[0]-self.n_req_max], axis=1)
                accepted_requests += tf.reduce_sum(tf.cast(tf.not_equal(act_req, -1), tf.int32)).numpy()
                accepted_rebalancing_requests += tf.reduce_sum(tf.cast(tf.not_equal(act_reb, -1), tf.int32)).numpy()
                try: 
                    if len(self.env.requests.index) - accepted_requests > 0:
                        rejected_requests += len(self.env.requests.index) - accepted_requests
                except:
                    pass

                next_state, reward, _,  hvs = self.env.step(action, state)
                test_reward += tf.reduce_sum(reward).numpy()

                state = next_state

            test_rewards.append(test_reward)
            test_accepted_requests.append(accepted_requests)
            test_accepted_rebalancing_requests.append(accepted_rebalancing_requests)
            test_rejected_requests.append(rejected_requests)

        if self.record_bool: 
            tf.summary.scalar(name="avg_test_reward", data=np.mean(test_rewards), step=1)
            for i in range(len(test_rewards)):
                tf.summary.scalar(name="test_reward", data=test_rewards[i], step=i+1)


        with open(self.log_dir + "/avg_test_reward.txt", 'w') as f:
            f.write(str(np.mean(test_rewards)))
        pd.DataFrame({"test_rewards_RL": test_rewards, "no_requests_acc": test_accepted_requests, "no_reb_requests_acc": test_accepted_rebalancing_requests, "no_requests_rejected": test_rejected_requests}, index=test_dates).to_csv(self.log_dir + "/test_rewards.csv")
