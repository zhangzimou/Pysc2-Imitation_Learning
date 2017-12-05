import random
from collections import namedtuple
from collections import deque
import numpy as np

from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features
import gflags as flags

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
FLAGS = flags.FLAGS


class Pysc2Wrapper(object):

    def __init__(self, env, dim, processor=None):
        self.env = env
        self.dim = dim
        self.processor = processor

    def step(self, action):
        x = action // self.dim
        y = action % self.dim
        obs = self.env.step(actions=[actions.FunctionCall(_MOVE_SCREEN,
                                                    [_NOT_QUEUED, [x, y]])])
        # select army
        while _MOVE_SCREEN not in obs[0].observation["available_actions"]:
            obs = self.env.step(actions=[actions.FunctionCall(_SELECT_ARMY,
                                                         [_SELECT_ALL])])
        return self._process_obs(obs)

    def reset(self):
        obs = self.env.reset()
        # select army
        while _MOVE_SCREEN not in obs[0].observation["available_actions"]:
            obs = self.env.step(actions=[actions.FunctionCall(_SELECT_ARMY,
                                                         [_SELECT_ALL])])
        if self.processor:
            return self.processor(obs[0].observation["screen"][_PLAYER_RELATIVE])
        return obs[0].observation["screen"][_PLAYER_RELATIVE]

    def _process_obs(self, obs):
        reward = obs[0].reward.astype(np.float64)
        done = obs[0].last()
        observation = obs[0].observation["screen"][_PLAYER_RELATIVE]
        if self.processor:
            observation = self.processor(observation)
        return observation, reward, done, None

    def render(self):
        pass


def MoveToBeaconProcessor(state):
    neutral_y, neutral_x = (state == _PLAYER_NEUTRAL).nonzero()
    if not neutral_y.any():
        return np.array([0, 0])
    x = int(neutral_x.mean())
    y = int(neutral_y.mean())
    dim = state.shape[0]
    result = np.zeros([2*dim])
    result[x] = 1
    result[dim + y] = 1
    return result


class RewardStepPairs(object):
    def __init__(self, rewards=None, steps=None):
        if rewards is None and steps is None:
            self.rewards = []
            self.steps = []
        elif rewards is not None and steps is not None:
            self.rewards = rewards
            self.steps = steps
        else:
            raise Exception("")

    def push(self, reward, step):
        self.rewards.append(reward)
        self.steps.append(step)

    def plot(self, gamma=0.7):
        import matplotlib.pyplot as plt
        if gamma is None:
            plt.plot(self.steps, self.rewards)
        else:
            reward_smooth = self._smooth(gamma)
            plt.plot(self.steps, self.rewards, self.steps, reward_smooth)
        plt.show()

    def reset(self):
        self.rewards = []
        self.steps = []

    def _smooth(self, gamma):
        reward_smooth = [None] * len(self.rewards)
        reward_smooth[0] = self.rewards[0]
        for i in range(1, len(self.rewards)):
            reward_smooth[i] = gamma * reward_smooth[i-1] + \
                               (1-gamma) * self.rewards[i]
        return reward_smooth

    @staticmethod
    def combine(pairs):
        rewards = []
        steps = []
        for pair in pairs:
            rewards.append(pair.rewards)
            steps.append(pair.steps)
        rewards = np.array(rewards)
        steps = np.array(steps)
        index = np.argsort(steps)
        steps = steps[index].tolist()
        rewards = rewards[index].tolist()
        return RewardStepPairs(rewards, steps)


def get_demo(agent, size):
    agent.memory.reset()
    agent.run(update_memory=True, num_step=size)
    return agent.memory


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in
                range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')