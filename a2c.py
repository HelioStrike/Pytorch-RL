import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gym
import numpy as np
from itertools import count
from utils import plotProgress

env = gym.make('CartPole-v0')

#Hyper-parameters
lr = 1e-2
GAMMA = 0.99
BATCH_SIZE = 5
OBSERVATIONS_DIM = 4
ACTIONS_DIM = 2

#Used to reduce the learning rate as we progress through epochs
RUNNING_GAMMA = 1

#Policy
class A3CNet(nn.Module):
    def __init__(self):
        super(A3CNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(OBSERVATIONS_DIM, 32),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, ACTIONS_DIM),
        )

        self.value = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out = self.model(x)
        advantage = self.advantage(out)
        value = self.value(out)

        return F.softmax(advantage), F.sigmoid(value)


reward_progress = []

#Model instance
policy = A3CNet()

#RMS prop optimizer
optimizer = optim.RMSprop(policy.parameters(), lr=lr)

#We'll be collecting our experiences for the pcoh using these 3 arrays
state_pool = []
action_pool = []
reward_pool = []

for e in count():
    state = env.reset()

    for i in count(1):
        #Calculate action from policy
        state = torch.from_numpy(state).float()
        logits, value = policy(state)
        m = Categorical(logits)
        action = m.sample().numpy()

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
        loss = 0
        for j in reversed(range(len(state_pool))):
            state = state_pool[j]
            action = torch.tensor(action_pool[j]).float()
            reward = np.int(reward_pool[j])

            logits, value = policy(state)
            logits = logits
            m = Categorical(logits)

            inter = reward - value
            value_loss = 0.5*inter.pow(2)
            policy_loss = -inter.detach()*m.log_prob(action)*RUNNING_GAMMA

            total_loss = value_loss + policy_loss
            loss += total_loss

        loss.backward()
        optimizer.step()
        RUNNING_GAMMA *= GAMMA

        state_pool = []
        action_pool = []
        reward_pool = []
