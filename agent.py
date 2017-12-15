import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from model import DQNModel, ModelWrapper
from memory import ReplayMemory
import copy
from utils import RewardStepPairs, Container, MoveToBeaconProcessor

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
        self.VERBOSE = kwargs.pop("verbose", False)
        self.GET_DEMO = kwargs.pop("get_demo", False)
        self.rule_processor = kwargs.pop("rule_processor", None)
        self.state_processor = kwargs.pop("state_processor", None)
        if self.TARGET_UPDATE_FREQ < 1:
            self.SOFT_UPDATE = True
            self.TARGET_UPDATE_FREQ = float(self.TARGET_UPDATE_FREQ)
        else:
            self.SOFT_UPDATE = False
            self.TARGET_UPDATE_FREQ = int(self.TARGET_UPDATE_FREQ)

        self.i_step = 0
        self.i_episode = 0
        self.record_i_step = 0
        self.record_i_episode = 0
        self.is_render = False
        self.is_training = True
        self.is_test = False
        self.reward_step_pairs = RewardStepPairs()


class DQNAgent(Agent):
    def __init__(self, model, env, **kwargs):
        """
        :type model: ModelWrapper
        """
        Agent.__init__(self, **kwargs)
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
        self.container = Container(self.model.SAVE_MODEL_NAME)


    def select_action(self, state):
        if self.is_training:
            self.global_step += 1
            self.eps = self.EPS_START - \
                       (self.EPS_START-self.EPS_END)/self.EPS_DECAY * \
                       self.global_step
            if self.eps < self.EPS_END:
                self.eps = self.EPS_END

        if self.is_training and np.random.rand() < self.eps:
            return LongTensor([[np.random.randint(self.out_dim)]])
        else:
            var = Variable(state, volatile=True).type(FloatTensor)
            out = self.model(var)
            return out.max(1)[1].data.view(1, 1)

    def _DQ_loss(self, y_pred, reward_batch,
                 non_final_mask, non_final_next_states):
        q_next = Variable(torch.zeros(self.BATCH_SIZE).type(FloatTensor))
        target_q = self.target_model(non_final_next_states)
        if self.DOUBLE_DQN:
            max_act = self.model(non_final_next_states).max(1)[1].view(-1,1)
            q_next[non_final_mask] = target_q.gather(1, max_act).data
        else:
            q_next[non_final_mask] = target_q.max(1)[0].data

        # next_state_values.volatile = False
        y = q_next * self.GAMMA + reward_batch
        loss = F.mse_loss(y_pred, y)
        return loss

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
        loss = self._DQ_loss(y_pred, reward_batch,
                             non_final_mask, non_final_next_states)
        self.container.add("y_pred", torch.mean(y_pred.data))
        self.container.add("loss", loss.data[0])
        return loss

    def update_policy(self):
        loss = self._calc_loss()
        self.opt.zero_grad()
        loss.backward()
        if self.GRADIENT_CLIPPING:
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
        if self.state_processor:
            state = self.state_processor(obs)
        else:
            temp = obs[None, :] if len(obs.shape)==1 else obs[None, None, :]
            state = torch.from_numpy(temp).type(FloatTensor)

        if self.GET_DEMO:
            action = self.rule_processor(obs)
        else:
            action = self.select_action(state)

        act = action.numpy().squeeze()
        if self.VERBOSE:
            print("action: {}".format(act))
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
        if update_memory:
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

    def fit(self, is_train, update_memory=True, num_step=np.inf,
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
                    if self.is_test:
                        self.container.add("reward", self.reward_episode, self.record_i_step)
                    self.debug_print(is_train)
                    break

    def train(self, **kwargs):
        self.is_training = True
        if kwargs.pop("clear", True):
            self.i_episode = 0
            self.i_step = 0
            self.reward_step_pairs.reset()
        print("Training starts...")
        self.fit(True, **kwargs)
        self.model.save()
        self.container.save()

    def run(self, **kwargs):
        self.is_training = False
        if kwargs.pop("clear", True):
            self.i_episode = 0
            self.i_step = 0
            self.reward_step_pairs.reset()
        print("Running starts...")
        self.fit(False, **kwargs)

    def _test(self, num_step):
        self.record_steps()
        self.is_test = True
        self.run(num_step=num_step)
        self.restore_steps()
        self.is_test = False

    def train_test(self, num_step, test_period=1000, test_step=100):
        self.i_episode = 0
        self.i_step = 0
        while self.i_step < num_step:
            self._test(test_step)
            self.train(num_step=self.record_i_step+test_period, clear=False)

        self._test(test_step)

    def debug_print(self, is_train):
        print("reward_episode {}".format(self.reward_episode))
        print("eps {}".format(self.eps))
        if is_train:
            print("loss_episode {}".format(self.container.get("loss")))
            print("y_pred_episode {}".format(self.container.get("y_pred")))

    def record_steps(self):
        self.record_i_episode = self.i_episode
        self.record_i_step = self.i_step

    def restore_steps(self):
        self.i_episode = self.record_i_episode
        self.i_step = self.record_i_step

    def plot(self, **kwargs):
        self.container.plot(**kwargs)


class DQfDAgent(DQNAgent):
    def __init__(self, model, env, demo_memory, **kwargs):
        """
        :type model: ModelWrapper
        :type demo_memory: ReplayMemory
        """
        DQNAgent.__init__(self, model, env, **kwargs)
        self.EXPERT_MARGIN = kwargs.pop("expert_margin", 0.8)
        self.DEMO_PER = kwargs.pop("demo_percent", 0.3)
        self.N_STEP = kwargs.pop("n_step", 5)
        self.LAMBDA_1 = kwargs.pop("lambda_1", 0.1)
        self.LAMBDA_2 = kwargs.pop("lambda_2", 0.5)
        self.LAMBDA_3 = kwargs.pop("lambda_3", 0)
        self.memory = ReplayMemory(self.REPLAY_CAPACITY, self.N_STEP, self.GAMMA)
        self.demo_memory = demo_memory
        self.demo_memory.n_step = self.N_STEP
        self.demo_memory.gamma = self.GAMMA
        self.is_pre_train = False
        # self.opt = optim.Adam(self.model.parameters(), lr=self.LR,
        #                       weight_decay=self.LAMBDA_3)


    def _n_step_loss(self, y_pred, n_returns_batch,
                     non_final_n_mask, non_final_n_states):

        q_n = Variable(torch.zeros(self.BATCH_SIZE).type(FloatTensor))
        target_q_n = self.target_model(non_final_n_states)
        if self.DOUBLE_DQN:
            max_act_n = self.model(non_final_n_states).max(1)[1].view(-1, 1)
            q_n[non_final_n_mask] = target_q_n.gather(1, max_act_n).data
        else:
            q_n[non_final_n_mask] = target_q_n.max(1)[0].data

        y_n_step = q_n * np.power(self.GAMMA, self.N_STEP) + n_returns_batch
        return F.mse_loss(y_pred, y_n_step)

    def _expert_loss(self, q_pred, action_batch, non_demo_mask):
        y_pred = q_pred.gather(1, action_batch).squeeze()
        expert_margin = torch.zeros(self.BATCH_SIZE, self.out_dim)
        expert_margin[:, action_batch.data] = self.EXPERT_MARGIN
        q_l = q_pred + Variable(expert_margin)
        j_e = q_l.max(1)[0] - y_pred
        j_e[non_demo_mask] = 0
        return j_e.sum()

    def _collect_batch(self):
        non_demo_mask = ByteTensor([False] * self.BATCH_SIZE)
        if self.is_pre_train:
            batch, n_returns, n_step_states = self.demo_memory.sample(
                self.BATCH_SIZE)
        else:
            demo_num = int(self.BATCH_SIZE * self.DEMO_PER)
            replay_demo, n_returns_demo, n_step_states_demo = \
                self.demo_memory.sample(demo_num)
            replay_agent, n_returns_agent, n_step_states_agent = \
                self.memory.sample(self.BATCH_SIZE - demo_num)
            batch = replay_demo.extend(replay_agent)
            if demo_num != self.BATCH_SIZE:
                non_demo_mask[demo_num:] = 1
            # n_returns = np.concatenate([n_returns_demo,
            #                             n_returns_agent])
            n_returns_demo.extend(n_returns_agent)
            n_returns = n_returns_demo
            n_step_states = np.concatenate([n_step_states_demo,
                                            n_step_states_agent])

        return batch, n_returns, n_step_states, non_demo_mask

    def _calc_loss(self):
        batch, n_returns, n_step_states, non_demo_mask = self._collect_batch()

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

        dq_loss = self._DQ_loss(y_pred, reward_batch,
                                non_final_mask, non_final_next_states)
        n_step_loss = self._n_step_loss(y_pred, n_returns_batch,
                                        non_final_n_mask, non_final_n_states)
        expert_loss = self._expert_loss(q_pred, action_batch, non_demo_mask)
        loss = dq_loss + self.LAMBDA_1 * n_step_loss + self.LAMBDA_2 * expert_loss
        self.container.add("dq_loss", torch.mean(dq_loss.data))
        self.container.add("expert_loss", torch.mean(expert_loss.data))
        self.container.add("y_pred", torch.mean(y_pred.data))
        self.container.add("loss", torch.mean(loss.data))
        return loss

    def pre_train(self, steps):
        self.i_episode = 0
        self.i_step = 0
        self.is_pre_train = True
        print("Pre training...")
        # import gc
        # import resource
        for i in range(steps):
            if i % 500 == 0:
                print("Pre train steps: {}".format(i))
            self.update_policy()
            self.update_target_network()
            # if i % 100 == 0:
            #     gc.collect()
            #     max_mem_used = resource.getrusage(
            #         resource.RUSAGE_SELF).ru_maxrss
            #     print("{:.2f} MB".format(max_mem_used / 1024))

        print("Pre train done")
        self.is_pre_train = False

    def debug_print(self, is_train):
        print("reward_episode {}".format(self.reward_episode))
        print("eps {}".format(self.eps))
        if is_train:
            print("loss_episode {}".format(self.container.get("loss")))
            print("dq_loss_episode {}".format(self.container.get("dq_loss")))
            print("expert_loss {}".format(self.container.get("expert_loss")))
            print("y_pred_episode {}".format(self.container.get("y_pred")))


class Pysc2DQNAgent(DQNAgent):

    def __init__(self, model, env, **kwargs):
        DQNAgent.__init__(self, model, env, **kwargs)

    def select_action(self, state):
        # if self.GET_DEMO:
        #     state = self.processor(state.numpy().squeeze())
        #     state = MoveToBeaconProcessor(state.numpy().squeeze())
        #     state = torch.from_numpy(state[None, :]).type(Tensor)
        #     a_x, a_y = self._extract_max_action(state)
        #     return (a_x*self.out_dim + a_y).view(1, 1)

        if self.is_training:
            self.global_step += 1
            self.eps = self.EPS_START - \
                       (self.EPS_START - self.EPS_END) / self.EPS_DECAY * \
                       self.global_step
            if self.eps < self.EPS_END:
                self.eps = self.EPS_END

        if self.is_training and np.random.rand() < self.eps:
            return LongTensor([[np.random.randint(self.out_dim * self.out_dim)]])
        else:
            var = Variable(state, volatile=True).type(FloatTensor)
            out = self.model(var)
            out_x = out[:, :self.out_dim]
            a_x = out_x.max(1)[1].data
            out_y = out[:, self.out_dim:]
            a_y = out_y.max(1)[1].data
            return (a_x*self.out_dim + a_y).view(1, 1)

    def _split_q(self, Q):
        return Q[:, :self.out_dim], Q[:, self.out_dim:]

    def _extract_max_action(self, Q):
        q_x, q_y = self._split_q(Q)
        return q_x.max(1)[1], q_y.max(1)[1]

    def _calc_loss(self):
        batch = self.memory.sample(self.BATCH_SIZE)
        non_final_mask = ByteTensor(
            tuple([s is not None for s in batch.next_state]))
        non_final_next_states = Variable(
            torch.cat([s for s in batch.next_state if s is not None]))

        state_batch = Variable(torch.cat(batch.state))
        reward_batch = Variable(torch.cat(batch.reward))
        action_x, action_y = batch.action / self.out_dim, batch.action % self.out_dim
        action_x_batch = Variable(torch.cat(action_x))
        action_y_batch = Variable(torch.cat(action_y))

        Q = self.model(state_batch)
        Q_x, Q_y = self._split_q(Q)
        y_predx = Q_x.gather(1, action_x_batch)
        y_predy = Q_y.gather(1, action_y_batch)

        dq_loss_x = self._DQ_loss(y_predx, reward_batch,
                                  non_final_mask, non_final_next_states)
        dq_loss_y = self._DQ_loss(y_predy, reward_batch,
                                  non_final_mask, non_final_next_states)

        return dq_loss_x + dq_loss_y


class Pysc2DQfDAgent(Pysc2DQNAgent, DQfDAgent):

    def __init__(self, model, env, demo_memory, **kwargs):
        Pysc2DQNAgent.__init__(self, model, env, **kwargs)
        DQfDAgent.__init__(self, model, env, demo_memory, **kwargs)

    def _calc_loss(self):
        batch, n_returns, n_step_states, non_demo_mask = self._collect_batch()

        non_final_mask = ByteTensor(
            tuple([s is not None for s in batch.next_state]))
        non_final_next_states = Variable(
            torch.cat([s for s in batch.next_state if s is not None]))
        non_final_n_mask = ByteTensor(tuple([s is not None for s in
                                             n_step_states]))
        non_final_n_states = Variable(torch.cat([s for s in n_step_states if
                                                 s is not None]))

        state_batch = Variable(torch.cat(batch.state))
        reward_batch = Variable(torch.cat(batch.reward))
        n_returns_batch = Variable(torch.cat(n_returns))
        action_x, action_y = batch.action / self.out_dim, batch.action % self.out_dim
        action_x_batch = Variable(torch.cat(action_x))
        action_y_batch = Variable(torch.cat(action_y))

        Q = self.model(state_batch)
        Q_x, Q_y = self._split_q(Q)
        y_predx = Q_x.gather(1, action_x_batch)
        y_predy = Q_y.gather(1, action_y_batch)

        dq_loss = self._DQ_loss(y_predx, reward_batch,
                                non_final_mask, non_final_next_states) + \
                  self._DQ_loss(y_predy, reward_batch,
                                non_final_mask, non_final_next_states)

        n_step_loss = self._n_step_loss(y_predx, n_returns_batch,
                                        non_final_n_mask, non_final_n_states) + \
                      self._n_step_loss(y_predy, n_returns_batch,
                                        non_final_n_mask, non_final_n_states)

        expert_loss = self._expert_loss(Q_x, action_x_batch, non_demo_mask) + \
                      self._expert_loss(Q_y, action_y_batch, non_demo_mask)

        loss = dq_loss + self.LAMBDA_1 * n_step_loss + self.LAMBDA_2 * expert_loss
        self.container.add("dq_loss", torch.mean(dq_loss.data))
        self.container.add("expert_loss", torch.mean(expert_loss.data))
        self.container.add("y_pred", torch.mean(y_predx.data+y_predy.data))
        self.container.add("loss", torch.mean(loss.data))
        return loss