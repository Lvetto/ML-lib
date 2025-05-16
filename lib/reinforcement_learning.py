import numpy as np
from lib.supervised_learning import supervised_learning
from random import random

class reinforcement_learning(supervised_learning):
    def __init__(self, layer_sizes, activations, loss):
        super().__init__(layer_sizes, activations, loss)

    # make the wrong choice on purpose during training with probability epsilon to promote exploration
    def choose_action(self, q, epsilon):
        if random() < epsilon:
            return np.random.choice(len(q))
        else:
            return np.argmax(q)
    
    def train(self, initial_state, transition_function, reward, learning_rate, discount, epsilon, epochs, decay_rate=0.9):
        # the agent can move between states by making choices (represented as a confidence level vector that should approach the expected value for each in a given state)
        # to train it, we give it an initial state and have it make a choice and simulate the outcome
        current_state = initial_state
        for epoch in range(epochs):
            # make a choice based on the current state and current weights
            q = self.compute(current_state)
            choice = self.choose_action(q, epsilon) # sometimes we ignore it

            # simulate the outcome and what the agent would do in the next state with the current weights
            next_state = transition_function(current_state, choice)
            q_next = self.compute(next_state)

            # approximate target output using the Belmann equation
            target = q.copy()
            target[choice] = reward(current_state, choice) + discount * np.max(q_next)  # we only consider the component corresponding to the choice made to increase stability

            # adjust the weights using backpropagation and advace to the next state
            self.backpropagate(current_state, target, learning_rate)
            current_state = next_state

            # decrease randomness over time. More randomness helps with training time in the beginning, but is not needed later
            epsilon *= decay_rate

