from math import pi

import gym
import numpy as np
from math import pi, floor
from cmath import rect, phase

from gym import spaces


def wrap_to_pi(a):
    """
    Wrap angle to [-pi  pi]

    :param a: angle to convert
    :return: new angle in the [-pi  pi]
    """
    return a - (2 * pi * floor((a + pi) / (2 * pi)))


def get_polar(x0, y0, xt, yt):
    """
    Compute the polar angle and the range between [x0 y0] and [xt yt]

    :param x0:
    :param y0:
    :param xt:
    :param yt:
    :return: Angle to target in radian in [-Pi  Pi] and norm
    """
    c = complex(yt - y0, xt - x0)
    return -phase(c), abs(c)


def preprocess_state(state):

    # obs = np.zeros(8)
    # # theta_att_wz, ro_att_wz   =  get_polar(state[0],state[1], state[9],state[10])
    # # theta_targ_wz, ro_targ_wz =  get_polar(state[4],state[5], state[9],state[10])

    # obs[0] = state [3] / pi         # Attacker heading
    # obs[1] = state [2] / 1200       # Attacker speed
    # #obs[2] = ro_att_wz / 100000     # Distance Attacker vs win zone
    # #obs[3] = theta_att_wz / pi      # Heading Attacker vs win zone

    # obs[4] = state [7] / pi         # Target heading
    # obs[5] = state [6] / 1200       # Target speed
    # obs[6] = ro_targ_wz / 100000    # Distance Target vs win zone
    # obs[7] = theta_targ_wz / pi     # Heading Target vs win zone

    #######################################
    # V2
    # obs = np.zeros(8)
    # theta_att_wz, ro_att_wz   =  get_polar(state[0],state[1], state[9],state[10])
    # theta_targ_wz, ro_targ_wz =  get_polar(state[4],state[5], state[9],state[10])

    # obs[0] = state [3] / pi         # Attacker heading
    # obs[1] = state [2] / 1200       # Attacker speed
    # obs[2] = ro_att_wz / 100000     # Distance Attacker vs win zone
    # obs[3] = theta_att_wz / pi      # Heading Attacker vs win zone

    # obs[4] = state [7] / pi         # Target heading
    # obs[5] = state [6] / 1200       # Target speed
    # obs[6] = ro_targ_wz / 100000    # Distance Target vs win zone
    # obs[7] = theta_targ_wz / pi     # Heading Target vs win zone
    #######################################

    # ? Need to add heading and ro vs loose zone?
    obs = np.zeros(11)

    theta_a_wz, ro_a_wz = get_polar(state[0], state[1], state[9], state[10])
    theta_t_wz, ro_t_wz = get_polar(state[4], state[5], state[9], state[10])
    theta_t_lz, ro_t_lz = get_polar(state[4], state[5], state[11], state[12])
    theta_a_t, ro_a_t = get_polar(state[0], state[1], state[4], state[5])

    obs[0] = (theta_a_wz) / pi
    obs[1] = (ro_a_wz - 50000) / 50000

    obs[2] = (theta_t_wz) / pi
    obs[3] = (ro_t_wz - 50000) / 50000

    obs[4] = (theta_a_t) / pi
    obs[5] = (ro_a_t - 50000) / 50000

    obs[6] = (state[2] - 600) / 600
    obs[7] = state[3] / pi

    # obs[8] = (state[6] -600) / 600
    obs[8] = state[7] / pi

    obs[9] = (ro_t_lz - 50000) / 50000
    obs[10] = theta_t_lz / pi

    # obs[0] = (state[0] - 50000) / 50000
    # obs[1] = (state[1] - 50000) / 50000
    # obs[2] = (state[2] -600) / 600
    # obs[3] = state[3] / pi
    # obs[4] = (state[4] - 50000) / 50000
    # obs[5] = (state[5] - 50000) / 50000
    # obs[6] = (state[6] -600) / 600
    # obs[7] = state[7] / pi
    # obs[8] = (state[8] - 50000) / 50000
    # obs[9] = (state[9] - 50000) / 50000
    # obs[10] = (state[10] - 50000) / 50000
    # obs[11] = (state[11] - 50000) / 50000
    # # obs[12] = (ro - 50000) / 100000
    # obs[13] = theta / pi
    return obs


class AcasWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(11,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.nb_step_in_rand = 0

    def reset(self):
        state = self.env.reset()
        return preprocess_state(state)

    def step(self, action):
        state, reward, done, info = self.env.step([action[0] * 200, action[1] * pi / 2])
        self.nb_step_in_rand += 1
        return preprocess_state(state), reward, done, info

    def increase_random(self):
        if self.nb_step_in_rand > 50000:
            self.env.increase_random()
            self.nb_step_in_rand = 0

    def render(self, mode):
        return self.env.render(mode)
