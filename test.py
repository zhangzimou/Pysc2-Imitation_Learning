from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import app
import gflags as flags
from agent import Pysc2DQNAgent
from model import ModelWrapper, AtariNet, MoveToBeaconTest
from utils import Pysc2Wrapper, MoveToBeaconProcessor


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
FLAGS = flags.FLAGS
MAP1 = "MoveToBeacon"
MAP2 = "CollectMineralShards"
DIM = 32


def main():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(map_name=MAP1,
                        screen_size_px=(DIM, DIM),
                        step_mul=80, visualize=False) as pysc2env:

        model = ModelWrapper(MoveToBeaconTest(DIM), DIM*DIM, DIM)
        env = Pysc2Wrapper(pysc2env, DIM, MoveToBeaconProcessor)
        agent = Pysc2DQNAgent(model, env,
                              replay_capacity=5e4,
                              replay_start=2000,
                              target_update_freq=0.005,
                              eps_decay=2e4,
                              verbose=False)
        agent.train(num_step=6e4)


if __name__ == "__main__":
    main()