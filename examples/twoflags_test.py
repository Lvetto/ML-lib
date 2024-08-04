import numpy as np
from random import random
from matplotlib import pyplot as plt

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import *

probabilites = [0.45, 0.55]

def reward(state, choice):
    return 1 if random() < probabilites[choice] else 0

def transition(state, q):
    return state

initial_state = np.array((1, 1))
agent = reinforcement_learning((2, 2, 2), [tanh_activation, tanh_activation], mse_loss)

rewards = []
for n in range(1000):
    agent.train(initial_state, transition, reward, 0.001, 0, 0.8, 10)

    t=0
    for i in range(100):
        t += reward(initial_state, np.argmax(agent.compute(initial_state)))

    rewards.append(t / 100)

plt.plot(rewards)
plt.show()
