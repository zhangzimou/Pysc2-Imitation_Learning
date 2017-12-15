from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from pysc2.env import sc2_env
import gflags as flags
from agent import Pysc2DQNAgent, Pysc2DQfDAgent
from model import ModelWrapper, SimpleNet
from utils import Pysc2Wrapper, plotTwoGraph
from processor import StateProcessor, MineralProcessor
from functools import partial

FLAGS = flags.FLAGS
# MAP = "MoveToBeacon"
MAP = "CollectMineralShards"
DIM = 32
STEP_MUL = 32

# rp = BeaconProcessor
rp = MineralProcessor
sp = partial(StateProcessor, processor=rp)


def test_dqn():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(map_name=MAP,
                        screen_size_px=(DIM, DIM),
                        step_mul=STEP_MUL, visualize=False) as pysc2env:

        model = ModelWrapper(SimpleNet(DIM), DIM*DIM, DIM,
                             save_model=False,
                             load_model_if_exist=False,
                             save_model_name="sc_dqn")
        env = Pysc2Wrapper(pysc2env, DIM)
        agent = Pysc2DQNAgent(model, env, double_DQN=True,
                              state_processor=sp,
                              replay_capacity=4e3,
                              replay_start=100,
                              target_update_freq=0.005,
                              eps_decay=2e4,
                              verbose=False)
        # agent.train(num_step=6e4)
        agent.train_test(num_step=3e4, test_period=1000, test_step=300)
        # agent.run(num_step=1e5)


def test_demo():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(map_name=MAP,
                        screen_size_px=(DIM, DIM),
                        step_mul=STEP_MUL, visualize=False) as pysc2env:
        model = ModelWrapper(SimpleNet(DIM), DIM * DIM, DIM,
                             save_model=True,
                             load_model_if_exist=False,
                             save_model_name="sc_dqfd")
        env = Pysc2Wrapper(pysc2env, DIM)
        agent_demo = Pysc2DQNAgent(model, env, state_processor=sp,
                                   rule_processor=rp, get_demo=True)
        agent_demo.run(num_step=300)
        demo_memory = agent_demo.memory

        agent = Pysc2DQfDAgent(model, env, demo_memory, double_DQN=True,
                               state_processor=sp,
                               replay_start=100,
                               replay_capacity=4e3,
                               batch_size=32,
                               target_update_freq=0.005,
                               eps_decay=2e4,
                               n_step=1,
                               demo_percent=0.2, lambda_1=0, lambda_2=0.1,
                               expert_margin=1)
        agent.pre_train(1000)
        agent.train_test(num_step=3e4, test_period=1000, test_step=300)
        # agent.plot()


def plot():
    plotTwoGraph("sc_dqfd", "sc_dqn", "DQfD", "DQN")

if __name__ == "__main__":
    # test_dqn()
    test_demo()
    # plot()