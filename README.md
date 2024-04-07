# Hybrid Multi-agent Deep Reinforcement Learning for Autonomous Mobility on Demand Systems

This software uses a combination of multi-agent Soft Actor-Critic, weighted bipartite matching and global critic loss to train and test a policy, represented by a neural network, that dispatches vehicles to requests and rebalances these vehicles in an Autonomous Mobility-on-demand system.

This method is proposed in:

> Zeno Woywood, Jasper I. Wiltfang, Julius Luy, Tobias Enders, Maximilian Schiffer (2024). Multi-Agent Soft Actor-Critic with Global Loss for Autonomous Mobility-on-Demand Fleet Control. arXiv preprint at arXiv: [Add link after upload].

All components (code, data, etc.) required to run the instances of the method considered in the paper and benchmarks are provided here. 

## Overview
The directory `/src` contains:
- Implementation of the control environment in `environment.py`
- Implementation of benchmark algorithms in `benchmark.py` (Benchmark can be executed by setting ).
- Implementation of multi-agent Soft Actor-Critic with global critic loss is split across the remaining files. 

The directory `/args` contains:
- Arguments for the experiments considered in the paper (see comments in `main.py` for explanations of these arguments). Benchmarks `Greedy` and (Rebalancing) `Heuristic` is selected based on the arg `rebalancing_bool = (True / False)` and can be executed by setting the arg `benchmark = (True / False)` with the respective setting arguments.

The directory `/data` contains:
- Pre-processed data for the two problem instances considered in the paper in multiple directories (e.g. `/nyc_data_11_small_zones`).

The code can be executed with `main.py` and the respective arguments to define the right data set, the used method, etc. Large parts of the code are based on code from this [GitHub repository](https://github.com/tumBAIS/HybridMADRL-AMoD), `trainer.py` and `sac_discrete.py` are partly based on code from this [GitHub repository](https://github.com/keiohta/tf2rl)

## Installation Instructions
Executing the code requires Python and the Python packages in `requirements.txt`, which can be installed with `pip install -r requirements.txt`. 
These packages include TensorFlow. In case of problems when trying to install TensorFlow, please refer to this [help page](https://www.tensorflow.org/install/errors).

## Code Execution
To run the code for an instance with arguments `[...].txt`, execute `python src/main.py @../args/[...].txt`. The argument are saved in directory `/args` where one also defines the correct data set. Replace `[...].txt` with the right set of arguments.

For typical instance and neural network sizes, a GPU should be used. 

Run the following code for a better overview of metrics:
```bash
tensorboard --logdir ./results
```
