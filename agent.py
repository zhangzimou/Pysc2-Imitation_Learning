import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from model import DQNModel
from memory import ReplayMemory
import copy
from utils import RewardStepPairs

LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor
ByteTensor = torch.ByteTensor
Tensor = FloatTensor


class Agent(object):
    def __init__(self, **kwargs):
        self.BATCH_SIZE = kwargs.pop("batch_size", 32)
        self.GAMMA = kwargs.pop("gamma", 0.99)
        self.EPS_START = kwargs.pop("eps_start", 0.95)
        self.EPS_END = kwargs.pop("eps_end", 0.01)
        self.EPS_DECAY = kwargs.pop("eps_decay", 2000)
        self.REPLAY_CAPACITY = kwargs.pop("replay_capacity", 50000)
        self.REPLAY_START = kwargs.pop("replay_start", 2000)
        self.DOUBLE_DQN = kwargs.pop("double_DQN", False)
        self.TARGET_UPDATE_FREQ = kwargs.pop("target_update_freq", 0.01)
        self.ACTION_REPEAT = kwargs.pop("action_repeat", 1)
        self.LR = kwargs.pop("learning_rate", 1e-3)
        self.GRADIENT_CLIPPING = kwargs.pop("gradient_clipping", 1)
        if self.TARGET_UPDATE_FREQ < 1:
            self.SOFT_UPDATE = True
            self.TARGET_UPDATE_FREQ = float(self.TARGET_UPDATE_FREQ)
        else:
            self.SOFT_UPDATE = False
            self.TARGET_UPDATE_FREQ = int(self.TARGET_UPDATE_FREQ)

        self.i_step = 0
        self.i_episode = 0
        self.is_render = False
        self.is_training = True
        self.reward_step_pairs = RewardStepPairs()

    def plot_reward(self):
        self.reward_step_pairs.plot()


class DQNAgent(Agent):
    def __init__(self, model, env, **kwargs):
        """
        :type model: DQNModel
        """
        super(DQNAgent, self).__init__(**kwargs)
        self.update_step = 0
        self.eps = self.EPS_START
        self.global_step = 0
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.in_dim = model.in_dim
        self.out_dim = model.out_dim
        self.memory = ReplayMemory(self.REPLAY_CAPACITY)
        # self.opt = optim.RMSprop(self.model.parameters())
        self.opt = optim.Adam(self.model.parameters(), lr=self.LR)
        self.env = env

    def select_action(self, state):
        self.global_step += 1
        if self.is_training:
            self.eps = self.EPS_START - \
                       (self.EPS_START-self.EPS_END)/self.EPS_DECAY * \
                       self.global_step
        else:
            self.eps = self.EPS_END
        if np.random.rand() < self.eps:
            return LongTensor([[np.random.randint(self.out_dim)]])
        else:
            var = Variable(state, volatile=True).type(FloatTensor)
            out = self.model(var)
            return out.max(1)[1].data.view(1, 1)

    def _calc_loss(self):
        batch = self.memory.sample(self.BATCH_SIZE)
        non_final_mask = ByteTensor(
            tuple([s is not None for s in batch.next_state]))
        non_final_next_states = Variable(
            torch.cat([s for s in batch.next_state if s is not None]))

        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        y_pred = self.model(state_batch).gather(1, action_batch).squeeze()
        q_next = Variable(
            torch.zeros(self.BATCH_SIZE).type(FloatTensor))
        target_q = self.target_model(non_final_next_states)
        if self.DOUBLE_DQN:
            max_act = self.model(non_final_next_states).max(1)[1]
            q_next[non_final_mask] = target_q.gather(1, max_act)
        else:
            q_next[non_final_mask] = target_q.max(1)[0]

        # next_state_values.volatile = False
        y = q_next * self.GAMMA + reward_batch
        loss = F.mse_loss(y_pred, y)
        return loss

    def update_policy(self):
        loss = self._calc_loss()
        self.opt.zero_grad()
        loss.backward()
        if self.GRADIENT_CLIPPING is not None:
            for param in self.model.parameters():
                param.grad.data.clamp_(-self.GRADIENT_CLIPPING,
                                       self.GRADIENT_CLIPPING)
        self.opt.step()

    def update_target_network(self):
        if not self.SOFT_UPDATE:
            self.update_step = (self.update_step + 1) % self.TARGET_UPDATE_FREQ
            if self.update_step == 0:
                state_dict = self.model.state_dict()
                self.target_model.load_state_dict(copy.deepcopy(state_dict))
        else:
            tw = self.target_model.state_dict().values()
            sw = self.model.state_dict().values()
            for t, s in zip(tw, sw):
                t.add_(self.TARGET_UPDATE_FREQ*(s-t))

    def _forward(self, obs, is_train, update_memory):
        state = torch.from_numpy(obs[None, :]).type(FloatTensor)
        action = self.select_action(state)
        act = action.numpy().squeeze()
        action_step = self.ACTION_REPEAT
        reward = 0
        done = False
        while action_step > 0:
            action_step -= 1
            next_obs, r, done, _ = self.env.step(act)
            reward += r
            if done:
                break

        self.reward_episode += reward
        if is_train or update_memory:
            reward = FloatTensor([reward])
            self.memory.push(state, action, reward)
            if done:
                self.memory.push(None, None, None)

        if len(self.memory) >= self.REPLAY_START and is_train:
            self.update_policy()
            self.update_target_network()

        if self.is_render:
            self.env.render()

        return next_obs, done

    def fit(self, is_train, update_memory=False, num_step=np.inf,
            num_episode=np.inf, max_episode_length=np.inf, is_render=False):
        if num_step == np.inf and num_episode == np.inf:
            raise Exception("")
        if num_step != np.inf and num_episode != np.inf:
            raise Exception("")

        self.is_render = is_render
        while self.i_episode < num_episode and self.i_step < num_step:
            self.i_episode += 1
            print("------------------------")
            print("episode: {}, step: {}".format(self.i_episode, self.i_step))
            obs = self.env.reset()
            self.reward_episode = 0
            episode_step = 0
            while episode_step < max_episode_length:
                episode_step += 1
                self.i_step += 1
                obs, done = self._forward(obs, is_train, update_memory)
                if done:
                    self.reward_step_pairs.push(self.reward_episode, self.i_step)
                    print("reward_episode {}".format(self.reward_episode))
                    break

    def train(self, **kwargs):
        if kwargs.pop("clear", True):
            self.i_episode = 0
            self.i_step = 0
            self.is_training = True
            self.reward_step_pairs.reset()
        print("Training starts...")
        self.fit(True, **kwargs)
        self.model.save()

    def run(self, **kwargs):
        if kwargs.pop("clear", True):
            self.i_episode = 0
            self.i_step = 0
            self.is_training = False
            self.reward_step_pairs.reset()
        print("Running starts...")
        self.fit(False, **kwargs)


class DQfDAgent(DQNAgent):
    def __init__(self, model, env, demo_memory, **kwargs):
        """
        :type model: DQNModel
        :type demo_memory: ReplayMemory
        """
        super(DQfDAgent, self).__init__(model, env, **kwargs)
        self.EXPERT_MARGIN = kwargs.pop("expert_margin", 0.8)
        self.DEMO_PER = kwargs.pop("demo_percent", 0.0)
        self.N_STEP = kwargs.pop("n_step", 1)
        self.LAMBDA_1 = kwargs.pop("lambda_1", 0)
        self.LAMBDA_2 = kwargs.pop("lambda_2", 0)
        self.LAMBDA_3 = kwargs.pop("lambda_3", 0)
        self.memory = ReplayMemory(self.REPLAY_CAPACITY, self.N_STEP, self.GAMMA)
        self.demo_memory = demo_memory
        self.demo_memory.n_step = self.N_STEP
        self.demo_memory.gamma = self.GAMMA
        self.is_pre_train = False
        # self.opt = optim.Adam(self.model.parameters(), lr=self.LR,
        #                       weight_decay=self.LAMBDA_3)

    def _calc_loss(self):
        non_demo_mask = ByteTensor([False] * self.BATCH_SIZE)
        if self.is_pre_train:
            batch, n_returns, n_step_states = self.demo_memory.sample(
                self.BATCH_SIZE)
        else:
            demo_num = int(self.BATCH_SIZE * self.DEMO_PER)
            replay_demo, n_returns_demo, n_step_states_demo = \
                self.demo_memory.sample(demo_num)
            replay_agent, n_returns_agent, n_step_states_agent = \
                self.memory.sample(self.BATCH_SIZE-demo_num)
            batch = replay_demo.extend(replay_agent)
            non_demo_mask[demo_num:] = 1
            n_returns = np.concatenate([n_returns_demo,
                                        n_returns_agent])
            n_step_states = np.concatenate([n_step_states_demo,
                                            n_step_states_agent])

        non_final_mask = ByteTensor(
            tuple([s is not None for s in batch.next_state]))
        non_final_next_states = Variable(
            torch.cat([s for s in batch.next_state if s is not None]))
        non_final_n_mask = ByteTensor(tuple([s is not None for s in
                                             n_step_states]))
        non_final_n_states = Variable(torch.cat([s for s in n_step_states if
                                                 s is not None]))

        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        n_returns_batch = Variable(torch.cat(n_returns))

        q_pred = self.model(state_batch)
        y_pred = q_pred.gather(1, action_batch).squeeze()

        q_next = Variable(torch.zeros(self.BATCH_SIZE).type(FloatTensor))
        q_n = Variable(torch.zeros(self.BATCH_SIZE).type(FloatTensor))
        target_q = self.target_model(non_final_next_states)
        target_q_n = self.target_model(non_final_n_states)
        if self.DOUBLE_DQN:
            max_act = self.model(non_final_next_states).max(1)[1]
            q_next[non_final_mask] = target_q.gather(1, max_act)
            max_act_n = self.model(non_final_n_states).max(1)[1]
            q_n[non_final_n_mask] = target_q_n.gather(1, max_act_n)
        else:
            q_next[non_final_mask] = target_q.max(1)[0]
            q_n[non_final_n_mask] = target_q_n.max(1)[0]

        # next_state_values.volatile = False
        y_1_step = q_next * self.GAMMA + reward_batch
        y_n_step = q_n * np.power(self.GAMMA, self.N_STEP) + n_returns_batch
        j_dq = F.mse_loss(y_pred, y_1_step)
        j_n = F.mse_loss(y_pred, y_n_step)

        # l(a_e, a)
        expert_margin = torch.zeros(self.BATCH_SIZE, self.out_dim)
        expert_margin[:, action_batch.data] = self.EXPERT_MARGIN
        q_l = q_pred + Variable(expert_margin)
        # max_a[Q(s,a)+l(a_e,a)] - Q(s,a_e)
        j_e = q_l.max(1)[0] - y_pred
        j_e[non_demo_mask] = 0
        j_e = j_e.sum()

        loss = j_dq + self.LAMBDA_1 * j_n + self.LAMBDA_2 * j_e
        return loss

    def pre_train(self, steps):
        self.i_episode = 0
        self.i_step = 0
        self.is_pre_train = True
        print("Pre training...")
        for i in range(steps):
            if i % 1000 == 0:
                print("Pre train steps: {}".format(i))
            self.update_policy()
            self.update_target_network()

        print("Pre train done")
        self.is_pre_train = False
