
# Report


## Learning Algorithm

### MADDPG

I tried multi-agent deep deterministic policy gradient (MADDPG), which is mentioned in this [paper](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf).

>we find the centralized critic with deterministic policies works very well in practice, and
refer to it as multi-agent deep deterministic policy gradient (MADDPG).

(reference : https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)

### Hyperparameters

```
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 1        # learning timestep
TIMES_LEARN = 5        # number of sampling and learning
NOISE_THETA = 0.15      # how fast the variable reverts towards to the mean
NOISE_SIGMA = 0.20      # degree of volatility
```

### Neural Networks

#### Actor

##### init
```
self.fc1 = nn.Linear(state_size * 2, fc1_units)
self.bn1 = nn.BatchNorm1d(fc1_units)
self.fc2 = nn.Linear(fc1_units, fc2_units)
self.fc3 = nn.Linear(fc2_units, action_size)
```

##### forward
```
x = F.relu(self.bn1(self.fc1(state)))
x = F.relu(self.fc2(x))
return F.tanh(self.fc3(x))
```

#### Critic

##### init
```
self.fcs1 = nn.Linear(state_size * 2, fcs1_units)
self.bn1 = nn.BatchNorm1d(fcs1_units)
self.fc2 = nn.Linear(fcs1_units+action_size * 2, fc2_units)
self.fc3 = nn.Linear(fc2_units, 1)
```

##### forward
```
xs = F.relu(self.bn1(self.fcs1(state)))
x = torch.cat((xs, action), dim=1)
x = F.relu(self.fc2(x))
return self.fc3(x)
```


## Plot of Rewards






```
Episode 50	Score: 0.00	Average score over 100 episodes: 0.00	Max score over 50 episodes: 0.00
Episode 100	Score: 0.00	Average score over 100 episodes: 0.00	Max score over 50 episodes: 0.10
Episode 150	Score: 0.00	Average score over 100 episodes: 0.01	Max score over 50 episodes: 0.10
Episode 200	Score: 0.00	Average score over 100 episodes: 0.02	Max score over 50 episodes: 0.20
Episode 250	Score: 0.00	Average score over 100 episodes: 0.04	Max score over 50 episodes: 0.10
Episode 300	Score: 0.00	Average score over 100 episodes: 0.06	Max score over 50 episodes: 0.20
Episode 350	Score: 0.00	Average score over 100 episodes: 0.08	Max score over 50 episodes: 0.20
Episode 400	Score: 0.09	Average score over 100 episodes: 0.08	Max score over 50 episodes: 0.20
Episode 450	Score: 0.10	Average score over 100 episodes: 0.09	Max score over 50 episodes: 0.20
Episode 500	Score: 0.20	Average score over 100 episodes: 0.10	Max score over 50 episodes: 0.40
Episode 550	Score: 0.10	Average score over 100 episodes: 0.13	Max score over 50 episodes: 0.49
Episode 600	Score: 0.00	Average score over 100 episodes: 0.14	Max score over 50 episodes: 0.49
Episode 650	Score: 0.10	Average score over 100 episodes: 0.11	Max score over 50 episodes: 0.39
Episode 700	Score: 0.40	Average score over 100 episodes: 0.12	Max score over 50 episodes: 0.80
Episode 750	Score: 0.49	Average score over 100 episodes: 0.15	Max score over 50 episodes: 0.70
Episode 800	Score: 0.30	Average score over 100 episodes: 0.21	Max score over 50 episodes: 0.90
Episode 850	Score: 0.10	Average score over 100 episodes: 0.22	Max score over 50 episodes: 1.10
Episode 900	Score: 0.10	Average score over 100 episodes: 0.21	Max score over 50 episodes: 0.60
Episode 950	Score: 0.60	Average score over 100 episodes: 0.21	Max score over 50 episodes: 0.60
Episode 1000	Score: 0.10	Average score over 100 episodes: 0.21	Max score over 50 episodes: 1.20
Episode 1050	Score: 0.40	Average score over 100 episodes: 0.23	Max score over 50 episodes: 1.10
Episode 1100	Score: 0.50	Average score over 100 episodes: 0.40	Max score over 50 episodes: 2.40
Episode 1150	Score: 0.10	Average score over 100 episodes: 0.48	Max score over 50 episodes: 2.00
Episode 1200	Score: 1.40	Average score over 100 episodes: 0.43	Max score over 50 episodes: 1.80
Episode 1250	Score: 2.50	Average score over 100 episodes: 0.44	Max score over 50 episodes: 2.50
Episode 1300	Score: 0.70	Average score over 100 episodes: 0.45	Max score over 50 episodes: 1.80
Episode 1350	Score: 0.00	Average score over 100 episodes: 0.41	Max score over 50 episodes: 1.30
Episode 1400	Score: 1.50	Average score over 100 episodes: 0.49	Max score over 50 episodes: 2.60
The Environment Was Solved at Episode 1328	Score: 0.60	Average score over 100 episodes: 0.50	Max score over 50 episodes: 2.60
```

![p3](https://user-images.githubusercontent.com/4464676/81146870-1a848880-8fb4-11ea-9375-0036ec9d229a.png)

![](https://i.gyazo.com/8c4e0a5bbd43a01126e25097e87ede1a.gif)



## Ideas for Future Work

- tweak hyperparameters to improve the performance
- try other unity-ml projects
- try other methods which is likely to be suitable for a multi agent environment.
- try other techniques to improve performance, such as prioritized experience replay
