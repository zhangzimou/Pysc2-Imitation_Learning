import gym
from agent import DQNAgent, DQfDAgent
from model import DQNModel
from utils import get_demo
from memory import ReplayMemory


def main():
    env = gym.make("CartPole-v0")
    obs = env.reset()
    in_dim = obs.shape[0]
    out_dim = env.action_space.n

    model_dqn = DQNModel(in_dim, out_dim, save_model=True,
                         load_model_if_exist=True, save_model_name="dqn_model")
    agent_dqn = DQNAgent(model_dqn, env, replay_start=50,
                         target_update_freq=0.005)
    agent_dqn.train(num_step=6e3, is_render=False)
    # agent_dqn.run(num_step=2e3, is_render=False)


    # demo_memory = get_demo(agent_dqn, 5e3)
    # demo_memory = ReplayMemory(10)
    # model_dqfd = DQNModel(in_dim, out_dim)
    # agent_dqfd = DQfDAgent(model_dqfd, env, demo_memory,
    #                        replay_start=50, target_update_freq=0.01,
    #                        eps_start=0.3, eps_decay=1000)
    # agent_dqfd.memory = agent_dqfd.demo_memory
    # agent_dqfd.pre_train(200)
    # agent_dqfd.train(clear=False, update_memory=True, num_step=4e3,
    #                  is_render=False)
    # agent_dqfd.run(num_step=2e3, is_render=False)
    # agent_dqfd.plot_reward()


if __name__ == "__main__":
    main()
