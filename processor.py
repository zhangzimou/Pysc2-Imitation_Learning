from __future__ import division
import numpy as np
from pysc2.lib import actions
from pysc2.lib import features
import torch


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]


def GetPlayerPosition(obs):
    player_y, player_x = (obs == _PLAYER_FRIENDLY).nonzero()
    result = np.zeros([2])
    if not player_y.any():
        return result
    result[0] = int(player_x.mean())
    result[1] = int(player_y.mean())
    return result


def StateProcessor(obs, processor):
    dim = obs.shape[0]
    action = processor(obs)
    x = action.numpy() // dim
    y = action.numpy() % dim
    result = np.zeros([2*dim])
    result[x] = 1
    result[dim+y] = 1
    return torch.from_numpy(result[None, :]).type(torch.FloatTensor)


def StateProcessor_(obs):
    dim = obs.shape[0]
    # result = np.zeros([2*dim+2])
    # result[-2:] = GetPlayerPosition(obs)
    result = np.zeros([2*dim])
    player_y, player_x = (obs == _PLAYER_FRIENDLY).nonzero()
    obs[player_y, player_x] = 0
    result[:dim] = np.sum(obs, axis=0)
    result[dim:2*dim] = np.sum(obs, axis=1)
    return torch.from_numpy(result[None, :]).type(torch.FloatTensor)


def StateProcessor2(obs):
    neutral_y, neutral_x = (obs == _PLAYER_NEUTRAL).nonzero()
    dim = obs.shape[0]
    result = np.zeros([2 * dim])
    if not neutral_y.any():
        return result
    x = int(neutral_x.mean())
    y = int(neutral_y.mean())
    result[x] = 1
    result[dim + y] = 1
    return torch.from_numpy(result[None, :]).type(torch.FloatTensor)


def BeaconProcessor(obs):
    neutral_y, neutral_x = (obs == _PLAYER_NEUTRAL).nonzero()
    dim = obs.shape[0]
    if not neutral_x.any():
        x = 0
        y = 0
    else:
        x = int(neutral_x.mean())
        y = int(neutral_y.mean())
    result = torch.LongTensor([1])
    result[0] = x*dim + y
    return result.view(1,1)


def MineralProcessor(obs):
    neutral_y, neutral_x = (obs == _PLAYER_NEUTRAL).nonzero()
    player_y, player_x = (obs == _PLAYER_FRIENDLY).nonzero()
    if not neutral_y.any() or not player_y.any():
        return torch.LongTensor([[0]])
    player = [int(player_x.mean()), int(player_y.mean())]
    closest, min_dist = None, None
    for p in zip(neutral_x, neutral_y):
        dist = np.linalg.norm(np.array(player) - np.array(p))
        if not min_dist or dist < min_dist:
            closest, min_dist = p, dist
    dim = obs.shape[0]
    result = torch.LongTensor([1])
    if closest is None:
        return result.view(1,1)
    result[0] = int(closest[0]*dim+closest[1])
    return result.view(1,1)


def BeaconProcessor_edge(obs):
    neutral_y, neutral_x = (obs == _PLAYER_NEUTRAL).nonzero()
    dim = obs.shape[0]
    if not neutral_x.any():
        nx = 0
        ny = 0
    else:
        nx = int(neutral_x.mean())
        ny = int(neutral_y.mean())
    px, py = GetPlayerPosition(obs)

    def isValid(xx):
        return 0 <= int(xx) < dim

    a, b = nx-px, ny-py
    x, y = None, None
    if a==0 and b==0:
        return torch.LongTensor([[0]])
    else:
        # x = px
        if a==0:
            # y=0
            if -py/b>=0:
                x, y = px, 0
            # y=dim-1
            if (dim-1-py)/b>=0:
                x, y = px, dim-1
        # y = py
        elif b==0:
            # x=0
            if -px/a>=0:
                x, y = 0, py
            # x=dim-1
            if (dim-1-px)/a>=0:
                x, y = dim-1, py
        else:
            # x=0
            t = -px/a
            if t>=0 and isValid(py+b*t):
                x, y = 0, py+b*t
            # x=dim-1
            t = (dim-1-px)/a
            if t>=0 and isValid(py+b*t):
                x, y = dim-1, py+b*t
            # y=0
            t = -py/b
            if t>=0 and isValid(px+a*t):
                x, y = px+a*t, 0
            # y=dim-1
            t = (dim-1-py)/b
            if t>=0 and isValid(px+a*t):
                x, y = px+a*t, dim-1

    if x is None:
        return torch.LongTensor([[0]])
    else:
        return torch.LongTensor([[int(x)*dim+int(y)]])


