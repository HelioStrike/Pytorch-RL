import random
from collections import deque
import matplotlib.pyplot as plt

class ReplayBuffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.transitions = deque()

    def add(self, observation, action, reward, observation2):
        if len(self.transitions) > self.max_size:
            self.transitions.popleft()
        self.transitions.append((observation, action, reward, observation2))

    def sample(self, count):
        return random.sample(self.transitions, count)

    def size(self):
        return len(self.transitions)

def plotProgress(arr):
    plt.figure(1)
    plt.plot(arr)
    plt.pause(0.001)
