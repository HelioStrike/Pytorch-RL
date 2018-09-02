import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *

ACTIONS_DIM = 2
OBSERVATIONS_DIM = 4
MAX_ITERATIONS = 10**6
LEARNING_RATE = 1e-2

NUM_EPOCHS = 50

GAMMA = 0.99
REPLAY_MEMORY_SIZE = 1000
NUM_EPISODES = 10000
TARGET_UPDATE_FREQ = 100
MINIBATCH_SIZE = 32

RANDOM_ACTION_DECAY = 0.99
INITIAL_RANDOM_ACTION = 1

reward_progress = []

def get_out_tensor(model, observation):
    np_obs = np.reshape(observation, [-1, OBSERVATIONS_DIM])
    return model(torch.from_numpy(np_obs).float())

def train(model, observations, targets, criterion, optimizer):
    optimizer.zero_grad()

    out_tensor = get_out_tensor(model, observations)
    loss = criterion(out_tensor, torch.tensor(targets))
    loss.backward()

    optimizer.step()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(OBSERVATIONS_DIM, 32),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, ACTIONS_DIM)
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

        return value + advantage - advantage.mean()

    def select_action(self, observation, epsilon):
        if random.random() > epsilon:
            observation = torch.Tensor(observation)
            q_value = self.forward(observation)
            action  = np.argmax(q_value.detach().numpy())
        else:
            action = random.randrange(2)
        return action

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
    steps_until_reset = TARGET_UPDATE_FREQ
    random_action_probability = INITIAL_RANDOM_ACTION

    replay = ReplayBuffer(REPLAY_MEMORY_SIZE)

    EPSILON = 1
    EPSILON_DECAY_RATE = 0.99

    model = Net()
    target_model = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    env = gym.make('CartPole-v0')
    total_reward = 0
    for episode in range(NUM_EPISODES):
        observation = env.reset()

        for iteration in range(MAX_ITERATIONS):
            random_action_probability *= RANDOM_ACTION_DECAY
            random_action_probability = max(random_action_probability, 0.1)
            old_observation = observation

            action = model.select_action(observation, EPSILON)
            EPSILON *= EPSILON_DECAY_RATE

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

        if replay.size() >= MINIBATCH_SIZE:
            sample_transitions = replay.sample(MINIBATCH_SIZE)
            target_model.load_state_dict(model.state_dict())
            update_action(model, target_model, sample_transitions, criterion, optimizer)
            steps_until_reset -= 1

if __name__ == '__main__':
    main()
