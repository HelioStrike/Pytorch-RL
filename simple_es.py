import numpy as np
from itertools import count
import gym

#making environment
env = gym.make('CartPole-v0')

#hyperparameters
OBSERVATION_DIM = 4
ACTION_DIM = 1
POPULATION_SIZE = 200
NUM_GENERATIONS = 100
SIGMA = 0.1        #noise standard deviation
ALPHA = 10         #learning rate
GAMMA = 0.99       #for learning rate decay

#initial weight
w = np.random.randn(ACTION_DIM, OBSERVATION_DIM)

#for each generation
for generation in range(NUM_GENERATIONS):
    observation = env.reset()
    total_reward = 0

    #this loop shows our agent in action
    for f in count():
        env.render()
        action = np.dot(w, observation)[0].clip(0, 1).astype(int)
        observation, reward, done, info  = env.step(action)
        total_reward += reward
        if done:
            print("Generation", generation, "Reward:", total_reward)
            break

    #noise
    pop_w = np.random.randn(POPULATION_SIZE, ACTION_DIM, OBSERVATION_DIM)
    #reward pool
    rewards = []

    #for each noise vector in the population
    for p in range(POPULATION_SIZE):
        total_reward = 0
        observation = env.reset()

        #add noise
        w_guess = w + SIGMA*pop_w[p]

        #the weights sometimes are exploding to nan. This is to prevent it
        if float('nan') in w_guess:
            rewards.append(0)
            continue

        #calculate the reward to see how the new weight is doing
        for f in count():
            action = np.dot(w_guess, observation)[0].clip(0, 1).astype(int)
            observation, reward, done, info  = env.step(action)
            total_reward += reward
            if done:
                rewards.append(total_reward)
                break

    #normalize rewards
    rewards = np.array(rewards)
    rewards = (rewards - np.mean(rewards) + np.std(rewards))/np.std(rewards)

    #update the weight
    w = w + ALPHA/(POPULATION_SIZE*SIGMA) * \
        np.dot(rewards.reshape(1, POPULATION_SIZE), pop_w.reshape(POPULATION_SIZE, ACTION_DIM*OBSERVATION_DIM)).reshape(ACTION_DIM, OBSERVATION_DIM)
    ALPHA *= GAMMA
