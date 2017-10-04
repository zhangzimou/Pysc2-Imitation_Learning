import gym
from agent import DQNAgent
from model import DQNModel


env = gym.make("CartPole-v0")
obs = env.reset()
in_dim = obs.shape[0]
out_dim = env.action_space.n
model = DQNModel(in_dim, out_dim)
agent = DQNAgent(model)
agent.fit(env, 600, is_render=False)
