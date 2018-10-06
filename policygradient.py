import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli
import gym
import numpy as np
from itertools import count
from utils import plotProgress

env = gym.make('CartPole-v0')

#Hyper-parameters
lr = 1e-2
GAMMA = 0.99
BATCH_SIZE = 5

#Used to reduce the learning rate as we progress through epochs
RUNNING_GAMMA = 1

#Policy
class PG(nn.Module):
    def __init__(self):
        super(PG, self).__init__()
        self.l1 = nn.Linear(4, 24)
        self.l2 = nn.Linear(24, 36)
        self.l3 = nn.Linear(36, 1)

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = self.l3(out)
        return F.sigmoid(out)

reward_progress = []

#Model instance
policy = PG()

#RMS prop optimizer
optimizer = optim.RMSprop(policy.parameters(), lr=lr)

#We'll be collecting our experiences using these 3 arrays
state_pool = []
action_pool = []
reward_pool = []

for e in count():
    state = env.reset()

    for i in count(1):
        #Calculate action from policy
        state = torch.from_numpy(state).float()
        probs = policy(state)
        m = Bernoulli(probs)
        action = m.sample()
        action = action.data.numpy().astype(int)[0]

        #Feed our action to the environment
        next_state, reward, done, _ = env.step(action)

        #If done, its probably because we failed. In that case, nullify our reward
        if done:
            reward = 0

        #Collect experiences
        state_pool.append(state)
        action_pool.append(float(action))
        reward_pool.append(reward)

        state = next_state

        #Add to reward_pool and plot our progress
        if done:
            print("Reward: ", i)
            reward_progress.append(i)
            plotProgress(reward_progress)
            break

    #We'll be stepping every BATCH_SIZE epochs
    if e > 0 and e % BATCH_SIZE == 0:
        running_add = 0

        for i in reversed(range(len(state_pool))):
            if(reward_pool[i] == 0):
                running_add = 0
            else :
                running_add = running_add*GAMMA + reward_pool[i]
                reward_pool[i] = running_add

        reward_pool  = np.array(reward_pool)
        reward_pool = (reward_pool - reward_pool.mean())/reward_pool.std()

        optimizer.zero_grad()
        for j in range(len(state_pool)):
            state = state_pool[j]
            action = torch.tensor(action_pool[j]).float()
            reward = np.int(reward_pool[j])

            probs = policy(state)
            m = Bernoulli(probs)
            loss = -reward*m.log_prob(action)*RUNNING_GAMMA

            loss.backward()
        optimizer.step()
        RUNNING_GAMMA *= GAMMA

        state_pool = []
        action_pool = []
        reward_pool = []
