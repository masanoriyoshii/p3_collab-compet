# p3_collab-compet


## Project Details

### The Environment

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Unity ML-Agents Tennis Environment](https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png)


In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

The project environment is similar to, but not identical to the Tennis environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md).

## Getting Started

**This repository is only for Windows 10 (64-bit).**

### Step 1
Follow this [instructions](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environments.


### Step 2

For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from the link below.
- [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip).

Then, place the file in the `p3_collab-compet/` folder, and unzip (or decompress) the file.


### Step 3
After you have followed the instructions above, open `Tennis.ipynb` (located in the `p3_collab-compet/` folder in the DRLND GitHub repository) and follow the instructions to learn how to use the Python API to control the agent.

## Instructions

- `Tennis.ipynb` : use to train the agent
- `agent.py` : use to construct the agents
- `model.py` : use to construct the network
- `Test_Learned.ipynb` : use to load the saved model weights and check if it works well
