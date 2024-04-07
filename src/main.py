"""Parse parameters and run algorithm"""

# parse parameters
import argparse

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

parser.add_argument('--data_dir', type=str) # relative path to directory where data is stored
parser.add_argument('--benchmark', type=str, choices=['True', 'False'], default='False') # greedy or heuristic comparison to RL approach
parser.add_argument('--model', type=str, choices=['dispatching', 'dispatching_rebalancing'])  # differentiates settings (dispatching for ablation)
parser.add_argument('--random_seed', type=int) # random seed
parser.add_argument('--episode_length', type=int) # episode length in seconds
parser.add_argument('--time_step_size', type=int) # time step size in seconds
parser.add_argument('--veh_count', type=int) # no. of vehicles
parser.add_argument('--max_req_count', type=int) # max. no. of requests per time step
parser.add_argument('--max_waiting_time', type=int) # max. waiting time in seconds
parser.add_argument('--cost_parameter', type=float) # mileage-dependent cost for maintenance etc. in USD per meter
parser.add_argument('--max_steps', type=int) # no. of steps to interact with environment
parser.add_argument('--min_steps', type=int) # no. of steps before neural net weight updates begin
parser.add_argument('--random_steps', type=int) # no. of steps with random policy at the beginning
parser.add_argument('--noise_steps', type=int) # no. of steps noise added to policy after random steps
parser.add_argument('--update_interval', type=int) # no. of steps between neural net weight updates
parser.add_argument('--validation_interval', type=int) # no. of steps between validation runs (must be multiple of no. of time steps per episode)
parser.add_argument('--tracking_interval', type=int) # interval at which training data is saved 
parser.add_argument('--rebalancing_bool', type=str, choices=("True", "False")) # whether rebalancing requests are generated
parser.add_argument('--rebalancing_mode', type=str, choices=("costs","reward_shaping")) # how rebalancing requests are rewarded
parser.add_argument('--rebalancing_request_generation', type=str, choices=("origin_destination_all", "origin_destination_neighbours")) # how rebalancing requests are generated
parser.add_argument('--attention', type=str) # whether attention layer is used
parser.add_argument('--req_embedding_dim', type=int) # units of request embedding layer
parser.add_argument('--veh_embedding_dim', type=int) # units of vehicle embedding layer
parser.add_argument('--req_context_dim', type=int) # first dim of W in request context layer
parser.add_argument('--veh_context_dim', type=int) # first dim of W in vehicle context layer
parser.add_argument('--inner_units', type=str) # units of inner network (sequence of feedforward layers)
parser.add_argument('--outer_units', type=str) # units of outer network (sequence of feedforward layers) after reshaping
parser.add_argument('--regularization_coefficient', type=float) # coefficient for L2 regularization of networks (0 if no regularization)
parser.add_argument('--rb_size', type=int) # replay buffer size
parser.add_argument('--batch_size', type=int) # (mini-)batch size
parser.add_argument('--log_alpha', type=float) # log(alpha)
parser.add_argument('--tau', type=float) # smoothing factor for exponential moving average to update target critic parameters
parser.add_argument('--huber_delta', type=float) # delta value at which Huber loss becomes linear
parser.add_argument('--gradient_clipping', type=str) # whether gradient clipping is applied
parser.add_argument('--clip_norm', type=float) # global norm used for gradient clipping
parser.add_argument('--lr', type=float) # learning rate (if scheduled, this is the start value)
parser.add_argument('--discount', type=float) # discount factor (if scheduled, this is the start value)
parser.add_argument('--normalized_rews', type=str) # whether rewards are normalized when sampled from replay buffer (if so, they are divided by the standard deviation of rewards currently stored in the replay buffer)
parser.add_argument('--results_dir', type=str) # relative path to directory where results shall be saved
parser.add_argument('--model_dir', type=str, default=None) # relative path to directory with saved model that shall be restored in the beginning (overwriting default initialization of network weights)
parser.add_argument('--record_bool', type=str, choices=("True", "False"), default="True") # turn off recording in order to do create less data when debugging or testing
parser.add_argument('--adjusted_loss_bool', type=str, choices=("True", "False"), default="True") # turn off specialized loss function that is more stable

args = vars(parser.parse_args())

args["inner_units"] = [int(i) for i in args["inner_units"].split(':')]
args["outer_units"] = [int(i) for i in args["outer_units"].split(':')]

# raise errors if arguments are not valid
if args["benchmark"] == "False":
    args["benchmark"] = False
elif args["benchmark"] == "True":
    args["benchmark"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --benchmark.')


if args["attention"] == "False":
    args["attention"] = False
elif args["attention"] == "True":
    args["attention"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --attention.')

if args["gradient_clipping"] == "False":
    args["gradient_clipping"] = False
elif args["gradient_clipping"] == "True":
    args["gradient_clipping"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --gradient_clipping.')

if args["normalized_rews"] == "False":
    args["normalized_rews"] = False
elif args["normalized_rews"] == "True":
    args["normalized_rews"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --normalized_rews.')

if args["rebalancing_bool"] == "False":
    args["rebalancing_bool"] = False
elif args["rebalancing_bool"] == "True":
    args["rebalancing_bool"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --rebalancing_bool.')

if args["record_bool"] == "False":
    args["record_bool"] = False
elif args["record_bool"] == "True":
    args["record_bool"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --record_bool.')

if args["adjusted_loss_bool"] == "False":
    args["adjusted_loss_bool"] = False
elif args["adjusted_loss_bool"] == "True":
    args["adjusted_loss_bool"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --adjusted_loss_bool.')



import random
import os
import sys
import numpy as np
import tensorflow as tf

# cross-platform compability 
# environment setting for TensorFlow - more information on https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#xla-best-practices
if sys.platform == 'darwin': # for MacOS with M1 chip
    os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=0"  # enable XLA (accelerated linear algebra) for CPU and GPU
    tf.config.run_functions_eagerly(True)
    np.set_printoptions(threshold=np.inf)
    # alternative flag --tf_xla_cpu_global_jit but not required on MacOs
elif sys.platform == 'win32': # for Windows
    os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2"
    #os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=1 --tf_xla_cpu_global_jit"  # enable XLA (accelerated linear algebra) for CPU and GPU
else: # for linux cluster
    os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2"  # #original prop for clusterin LRZ

# set seed and further global settings
seed = args["random_seed"]
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# test with solution from https://discuss.tensorflow.org/t/attributeerror-module-keras-api-v2-keras-mixed-precision-has-no-attribute-experimental/11225/3
tf.keras.mixed_precision.set_global_policy('mixed_float16')  # optimized for GPU # enable mixed precision computations (Mixed precision computations aim to strike a balance between computational efficiency and numerical stability by performing some operations in float16 precision and others in float32 precision.)
# tf.keras.mixed_precision.set_global_policy('float32')  # optimized for CPU

# initialize Environment, Soft Actor-Critic and Trainer
from environment import Environment
from sac_discrete import SACDiscrete
from trainer import Trainer

env = Environment(args)
policy = SACDiscrete(args, env)
trainer = Trainer(policy, env, args)

trainer() 
