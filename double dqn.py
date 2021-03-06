import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *

#Hyperparameters
ACTIONS_DIM = 2
OBSERVATIONS_DIM = 4
MAX_ITERATIONS = 10**6
LEARNING_RATE = 1e-2

NUM_EPOCHS = 50

GAMMA = 0.99
REPLAY_MEMORY_SIZE = 1000
NUM_EPISODES = 10000
TARGET_UPDATE_FREQ = 100
MINIBATCH_SIZE = 100

RANDOM_ACTION_DECAY = 0.99
INITIAL_RANDOM_ACTION = 1

#We'll be using this array for plotting our rewards
reward_progress = []

#calculates model(observation)
def get_out_tensor(model, observation):
    np_obs = np.reshape(observation, [-1, OBSERVATIONS_DIM])
    return model(torch.from_numpy(np_obs).float())

#trains model
def train(model, observations, targets, criterion, optimizer):
    optimizer.zero_grad()

    out_tensor = get_out_tensor(model, observations)
    loss = criterion(out_tensor, torch.tensor(targets))
    loss.backward()

    optimizer.step()

#network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(OBSERVATIONS_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, ACTIONS_DIM)
        )

    def forward(self, x):
        return self.model(x)

#trains model by taking sample inputs
def update_action(action_model, target_model, sample_transitions, criterion, optimizer):
    random.shuffle(sample_transitions)
    batch_observations = []
    batch_targets = []

    for sample_transition in sample_transitions:
        old_observation, action, reward, observation = sample_transition

        targets = np.reshape(get_out_tensor(action_model, old_observation).detach().numpy(), ACTIONS_DIM)
        targets[action] = reward
        if observation is not None:
            predictions = get_out_tensor(target_model, observation).detach().numpy()
            new_action = np.argmax(predictions)
            targets[action] += GAMMA * predictions[0, new_action]

        batch_observations.append(old_observation)
        batch_targets.append(targets)

        train(action_model, batch_observations, batch_targets, criterion, optimizer)

def main():
    random_action_probability = INITIAL_RANDOM_ACTION

    #replay mempry
    replay = ReplayBuffer(REPLAY_MEMORY_SIZE)

    #model, target_model, loss function, optimizer
    model = Net()
    target_model = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #creating environment
    env = gym.make('CartPole-v0')

    #collects reward to plot it later
    total_reward = 0
    for episode in range(NUM_EPISODES):
        #reset env
        observation = env.reset()

        #collect experiences
        for iteration in range(MAX_ITERATIONS):
            random_action_probability *= RANDOM_ACTION_DECAY
            random_action_probability = max(random_action_probability, 0.1)
            old_observation = observation

            if np.random.random() < random_action_probability:
                action = np.random.choice(ACTIONS_DIM)
            else:
                action = get_out_tensor(model, observation).detach().numpy()
                action = np.argmax(action)

            observation, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                print("Episode", episode, "Score:", total_reward)
                reward_progress.append(total_reward)
                plotProgress(reward_progress)
                reward = -200
                total_reward = 0
                replay.add(old_observation, action, reward, None)
                break

            replay.add(old_observation, action, reward, observation)

        #if we have enough experiences, train the model
        if replay.size() >= MINIBATCH_SIZE:
            sample_transitions = replay.sample(MINIBATCH_SIZE)
            target_model.load_state_dict(model.state_dict())
            update_action(model, target_model, sample_transitions, criterion, optimizer)

if __name__ == '__main__':
    main()
