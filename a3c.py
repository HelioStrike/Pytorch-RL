import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gym
import numpy as np
from itertools import count
from utils import plotProgress

#Defining the environment
ENV_NAME = "CartPole-v0"

#Hyper-parameters
lr = 1e-2
GAMMA = 0.99
BATCH_SIZE = 5
OBSERVATIONS_DIM = 4
ACTIONS_DIM = 2
NUM_WORKERS = 4
#Used to reduce the learning rate as we progress through epochs
RUNNING_GAMMA = 1

#A3C network
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

#Model instance
policy = A3CNet()

#RMS prop optimizer
optimizer = optim.RMSprop(policy.parameters(), lr=lr)

#Worker class
class Worker:
    def __init__(self, env_name):
        self.env = gym.make(env_name)

        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []

    def collect_experiences(self):
        state = self.env.reset()

        total_reward = 0
        for i in count(1):
            #Calculate action from policy
            state = torch.from_numpy(state).float()
            logits, value = policy(state)
            m = Categorical(logits)
            action = m.sample().numpy()

            #Feed our action to the environment
            next_state, reward, done, _ = self.env.step(action)

            total_reward += reward

            #If done, its probably because we failed. In that case, nullify our reward
            if done:
                reward = 0

            #Collect experiences
            self.state_pool.append(state)
            self.action_pool.append(float(action))
            self.reward_pool.append(reward)

            state = next_state

            #Add to reward_pool and plot our progress
            if done:
                print("Reward: ", i)
                break

        return total_reward

    def make_step(self):
        running_add = 0

        #Normalizing rewards
        for i in reversed(range(len(self.state_pool))):
            if(self.reward_pool[i] == 0):
                running_add = 0
            else:
                running_add = running_add*GAMMA + self.reward_pool[i]
                self.reward_pool[i] = running_add

        self.reward_pool  = np.array(self.reward_pool)
        self.reward_pool = (self.reward_pool - self.reward_pool.mean())/self.reward_pool.std()

        loss = 0
        for j in reversed(range(len(self.state_pool))):
            state = self.state_pool[j]
            action = torch.tensor(self.action_pool[j]).float()
            reward = np.int(self.reward_pool[j])

            logits, value = policy(state)
            logits = logits
            m = Categorical(logits)

            inter = reward - value
            value_loss = 0.5*inter.pow(2)
            policy_loss = -inter.detach()*m.log_prob(action)*RUNNING_GAMMA

            total_loss = value_loss + policy_loss
            loss += total_loss

        loss.backward()

        #Emptying buffers
        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []

#Workers
workers = [Worker(ENV_NAME) for _ in range(NUM_WORKERS)]

#We'll use this array to plot the progress of our model
reward_progress = []

#We'll use this array to calculate the average reward acquired by each worker
rewards = []

#For each epoch
for e in count():
    #collect experiences and append to the rewards array
    for worker in workers:
        rewards.append(worker.collect_experiences())

    #append to reward_progress and plot it
    reward_progress.append(np.array(rewards).mean())
    rewards = []
    plotProgress(reward_progress)

    #train every BATCH_SIZE batches
    if e > 0 and e%BATCH_SIZE == 0:
        optimizer.zero_grad()
        for worker in workers:
            worker.make_step()
        optimizer.step()
        RUNNING_GAMMA *= GAMMA
