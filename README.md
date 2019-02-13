
# Project 2: Continuous Control

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Instructions

Follow the instructions in `Continuous_Control.ipynb` to get started with training your own agent! ..Once training is done, then see how the trained agent performs

### supporting script
In model.py, the actual policy network and network for calculating value for a given state is implemented.

In agent.py, the function for the clipped surrogate objective function for PPO and to collect the trajectories by running old policy is implemented.
