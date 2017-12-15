import numpy as np
from functools import reduce
from collections import namedtuple
import random
from torch import FloatTensor, LongTensor

# Transition = namedtuple("Transition", ["state", "action", "reward",
#                                        "next_state"])


class Transition(object):
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

    def extend(self, trans):
        """
        :type trans: Transition
        """
        self.state = np.concatenate([self.state, trans.state])
        self.action = np.concatenate([self.action, trans.action])
        self.reward = np.concatenate([self.reward, trans.reward])
        self.next_state = np.concatenate([self.next_state, trans.next_state])
        return self


class ReplayMemorySimple(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReplayMemory(object):
    def __init__(self, capacity, n_step=None, gamma=None):
        capacity = int(capacity)
        self.capacity = capacity
        self.states = np.array([None]*capacity)
        self.actions = np.array([None]*capacity)
        self.rewards = np.array([None]*capacity)
        self.position = 0
        self.size = 0
        self.n_step = n_step
        self.gamma = gamma
        if n_step is not None and gamma is None:
            raise Exception("")

    def push(self, state, action, reward):
        """
        :type state: FloatTensor
        :type action: LongTensor
        :type reward: FloatTensor
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.position = (self.position + 1) % self.capacity
        self.size += 1
        if self.size > self.capacity:
            self.size = self.capacity

    def sample(self, batch_size, n_step=None):
        if n_step is not None:
            self.n_step = n_step

        index = np.array([], dtype=int)
        while index.size < batch_size:
            index_ = np.random.choice(self.size, batch_size).astype(int)
            index_ = self._valid_index(index_)
            index = np.concatenate([index, index_])
        index = index[:batch_size]
        states_batch = self.states[index]
        actions_batch = self.actions[index]
        rewards_batch = self.rewards[index]
        next_states_batch = self.states[(index+1)%self.capacity]
        trans = Transition(states_batch, actions_batch, rewards_batch,
                   next_states_batch)
        if self.n_step is None:
            return trans
        else:
            n_returns = self.n_step_return(index)
            n_step_states = self.states[(index+self.n_step)%self.capacity]
            return trans, n_returns, n_step_states

    def _valid_index(self, index):
        if self.n_step is None:
            valid_index = [self.states[i] is not None for i in index]
            return index[valid_index]

        n = self.n_step

        def util(i):
            if i+n <= self.capacity:
                states = self.states[i:i+n]
            else:
                states = np.concatenate([self.states[i:],
                                         self.states[:i+n-self.capacity]])
            return np.all([state is not None for state in states])

        valid_index = [util(i) for i in index]
        return index[valid_index]

    def n_step_return(self, index):
        n = self.n_step

        def util(i):
            if i+n <= self.capacity:
                rewards = self.rewards[i:i+n]
            else:
                rewards = np.concatenate([self.rewards[i:],
                                          self.rewards[:i+n-self.capacity]])
            return reduce(lambda x,y: self.gamma*x+y, reversed(rewards))

        return [util(i) for i in index]

    def __len__(self):
        return self.size

    def reset(self):
        self.states = np.array([None] * self.capacity)
        self.actions = np.array([None] * self.capacity)
        self.rewards = np.array([None] * self.capacity)
        self.position = 0
        self.size = 0
