import random
from collections import namedtuple
from collections import deque
import numpy as np
from functools import reduce


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


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
        self.capacity = capacity
        self.states = np.array([None]*capacity)
        self.actions = np.array([None]*capacity)
        self.rewards = np.array([None]*capacity)
        self.dones = np.array([None]*capacity)
        self.position = 0
        self.size = 0
        self.n_step = n_step
        self.gamma = gamma
        if n_step is not None and gamma is None:
            raise Exception("")

    def push(self, state, action, reward, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.position = (self.position + 1) % self.capacity
        self.size += 1

    def sample(self, batch_size):
        index = np.array([])
        while index.size < batch_size:
            index_ = np.random.choice(self.size, batch_size)
            index_ = self._valid_index(index_)
            index = np.concatenate([index, index_])
        index = index[:batch_size]
        states_batch = self.states[index]
        actions_batch = self.actions[index]
        rewards_batch = self.actions[index]
        next_states_batch = self.states[(index+1)%self.capacity]
        done_batch = self.dones[(index+1)%self.capacity]
        trans = Transition(states_batch, actions_batch, rewards_batch,
                   next_states_batch, done_batch)
        if self.n_step is None:
            return trans
        else:
            n_returns = self.n_step_return(index)
            n_step_states = self.states[(index+self.n_step)%self.capacity]
            return trans, n_returns, n_step_states

    def _valid_index(self, index):
        if self.n_step is None:
            return index[self.dones[index] is False]

        n = self.n_step - 1

        def util(i):
            if i+n <= self.capacity:
                dones = self.dones[i:i+n]
            else:
                dones = np.concatenate([self.dones[i:],
                                        self.dones[:i+n-self.capacity]])
            return np.all(dones is False)

        valid_index = [util(i) for i in index]
        return index[valid_index]

    def n_step_return(self, index):
        n = self.n_step - 1

        def util(i):
            if i+n <= self.capacity:
                rewards = self.rewards[i+i+n]
            else:
                rewards = np.concatenate([self.rewards[i:],
                                          self.rewards[i+n-self.capacity]])
            return reduce(lambda x,y: self.gamma*x+y, reversed(rewards))

        return [util(i) for i in index]

    def __len__(self):
        return self.size


class NStepReturn(object):
    def __init__(self, n_step, gamma, model):
        self.n_step = n_step
        self.data = deque(maxlen=n_step)
        self.gamma = gamma
        self.gamma_n_1 = np.power(gamma, n_step-1)
        self.gamma_n = self.gamma_n_1 * gamma
        self.running_value = None
        self.model = model

    def push(self, state, reward):
        if len(self.data) == self.data.maxlen:
            if self.running_value is None:
                self.running_value = reduce(lambda x,y:self.gamma * x + y,
                                            reversed(self.data))
            else:
                self.running_value -= self.data.popleft()[1]
                self.running_value /= self.gamma
                self.running_value += self.gamma_n_1 * reward

        item = (state, reward)
        self.data.append(item)

    @property
    def return_value(self):
        q_value = self.model(self.data[-1][0]).max(1)[0]
        return self.running_value + self.gamma_n * q_value

    def reset(self):
        self.data.clear()
        self.running_value = None