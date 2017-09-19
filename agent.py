import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from model import DQNModel
from utils import ReplayMemory, Transition, NStepReturn
import copy

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
        if self.TARGET_UPDATE_FREQ < 1:
            self.SOFT_UPDATE = True
            self.TARGET_UPDATE_FREQ = float(self.TARGET_UPDATE_FREQ)
        else:
            self.SOFT_UPDATE = False
            self.TARGET_UPDATE_FREQ = int(self.TARGET_UPDATE_FREQ)

        self.i_step = 0
        self.i_episode = 0
        self.env = None
        self.is_render = False


class DQNAgent(Agent):
    def __init__(self, model, **kwargs):
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
        self.opt = optim.Adam(self.model.parameters(), 1e-3)

    def select_action(self, state):
        self.global_step += 1
        self.eps = self.EPS_START - \
                   (self.EPS_START-self.EPS_END)/self.EPS_DECAY * self.global_step
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

        y_pred = self.model(state_batch).gather(1, action_batch)
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
        if len(self.memory) < self.REPLAY_START:
            return
        loss = self._calc_loss()
        self.opt.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
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

    def _forward(self, obs):
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
        reward = FloatTensor([reward])
        self.memory.push(state, action, reward)
        if done:
            self.memory.push(None, None, None)

        if self.is_render:
            self.env.render()

        self.update_policy()
        self.update_target_network()
        return next_obs, done

    def fit(self, env, num_step=np.inf, num_episode=np.inf,
            max_episode_length=np.inf, is_render=False):
        if num_step == np.inf and num_episode == np.inf:
            raise Exception("")
        if num_step != np.inf and num_episode != np.inf:
            raise Exception("")

        self.env = env
        self.is_render = is_render
        while self.i_episode < num_episode and self.i_step < num_step:
            self.i_episode += 1
            print("------------------------")
            print("episode: {}, step: {}".format(self.i_episode, self.i_step))
            obs = env.reset()
            self.reward_episode = 0
            episode_step = 0
            while episode_step < max_episode_length:
                episode_step += 1
                self.i_step += 1
                obs, done = self._forward(obs)
                if done:
                    print("reward_episode {}".format(self.reward_episode))
                    break


class DQfDAgent(DQNAgent):
    def __init__(self, model, demo_memory, **kwargs):
        super(DQfDAgent, self).__init__(model, **kwargs)
        self.PRE_TRAIN_STEP = kwargs.pop("pre_train_step", 100)
        self.EXPERT_MARGIN = kwargs.pop("expert_margin", 0.8)
        self.DEMO_PER = kwargs.pop("demo_percent", 0.3)
        self.N_STEP = kwargs.pop("n_step", 10)
        if kwargs != {}:
            raise Exception("No such keywords {}".format(kwargs))
        self.demo_memory = demo_memory
        self.n_step_return = NStepReturn(self.N_STEP, self.GAMMA)
        self.is_pre_train = False

    def _calc_loss(self):
        non_demo_mask = ByteTensor([False] * self.BATCH_SIZE)
        if self.is_pre_train:
            replay = self.demo_memory(self.BATCH_SIZE)
        else:
            demo_num = self.BATCH_SIZE * self.DEMO_PER
            replay_demo = self.demo_memory.sample(demo_num)
            replay_agent = self.memory.sample(self.BATCH_SIZE-demo_num)
            replay = replay_demo.extend(replay_agent)
            non_demo_mask[demo_num:] = 1

        batch = Transition(*zip(*replay))
        non_final_mask = ByteTensor(
            tuple([s is not None for s in batch.next_state]))
        non_final_next_states = Variable(
            torch.cat([s for s in batch.next_state if s is not None]))

        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        q_pred = self.model(state_batch)
        y_pred = q_pred.gather(1, action_batch)
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
        j_dq = F.mse_loss(y_pred, y)

        # l(a_e, a)
        expert_margin = torch.zeros(self.BATCH_SIZE, self.out_dim)
        expert_margin[:, action_batch.data] = self.EXPERT_MARGIN
        q_l = q_pred + Variable(expert_margin)
        # max_a[Q(s,a)+l(a_e,a)] - Q(s,a_e)
        j_e = q_l.max(1)[0] - y_pred
        j_e[non_demo_mask] = 0
        j_e = j_e.sum()


        loss = j_dq + j_e
        return loss

    def pre_train(self):
        self.is_pre_train = True

