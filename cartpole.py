import gym
from agent import DQNAgent, DQfDAgent
from model import DQNModel, ModelWrapper
from utils import get_demo, plotTwoGraph
from memory import ReplayMemory


def test_dqn():
    env = gym.make("CartPole-v0")
    obs = env.reset()
    in_dim = obs.shape[0]
    out_dim = env.action_space.n

    model_dqn = ModelWrapper(DQNModel(in_dim, out_dim), in_dim, out_dim,
                             save_model=False,
                             load_model_if_exist=False,
                             save_model_name="cartpole_dqn")
    agent_dqn = DQNAgent(model_dqn, env, replay_start=200, target_update_freq=0.005,
                         eps_start=0.9, eps_decay=1000, double_DQN=True)
    # agent_dqn.train(num_step=10e4, is_render=False)
    agent_dqn.train_test(num_step=6000, test_period=200, test_step=500)
    # agent_dqn.plot()


def test_demo():
    env = gym.make("CartPole-v0")
    obs = env.reset()
    in_dim = obs.shape[0]
    out_dim = env.action_space.n

    model_dqn = ModelWrapper(DQNModel(in_dim, out_dim), in_dim, out_dim,
                             save_model=True,
                             load_model_if_exist=True,
                             save_model_name="dqn_model")
    agent_dqn = DQNAgent(model_dqn, env, replay_start=50,
                         target_update_freq=0.005)
    # agent_dqn.train(num_step=6e3, is_render=False)

    demo_memory = get_demo(agent_dqn, 5e3)
    model_dqfd = ModelWrapper(DQNModel(in_dim, out_dim), in_dim, out_dim,
                              save_model=True,
                              load_model_if_exist=False,
                              save_model_name="cartpole_dqfd")
    agent_dqfd = DQfDAgent(model_dqfd, env, demo_memory, double_DQN=True,
                           replay_start=200, target_update_freq=0.005,
                           eps_start=0.9, eps_decay=1000, n_step=1,
                           demo_percent=0.3, lambda_1=0, lambda_2=0.05,
                           expert_margin=0.5)
    agent_dqfd.pre_train(2000)
    # agent_dqfd.train(clear=True, update_memory=True, num_step=8e4,
    #                  is_render=False)
    agent_dqfd.train_test(num_step=6000, test_period=200, test_step=500)
    # agent_dqfd.plot(save_name="cartpole_dqfd")
    # agent_dqfd.plot()


def plot():
    plotTwoGraph("cartpole_dqfd", "cartpole_dqn", "DQfD", "DQN")

if __name__ == "__main__":
    test_dqn()
    test_demo()
    plot()