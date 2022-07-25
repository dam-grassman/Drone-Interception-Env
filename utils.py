from numpy.random import rand
from math import pi, floor
from cmath import rect, phase
import numpy as np

def get_random_pos(c, r):
    """
    Returns a random position in the square of center c and
    length side r.

    :param a: Center of the square
    :param a: length of the square side
    :return: position x, y
    """
    x = (c[0] - r / 2) + rand() * r
    y = (c[1] - r / 2) + rand() * r
    return x, y


def wrap_to_pi(a):
    """
    Wrap angle to [-pi  pi]

    :param a: angle to convert
    :return: new angle in the [-pi  pi]
    """
    return a - (2 * pi * floor((a + pi) / (2 * pi)))


def get_cart(ro, a):
    """
    Computation of cartesian coordinate from polar coordinate

    :param ro: norm of the vector
    :param a: angle of the vector (0 is the NORTH)
    :return: x0,y0 of the vector
    """
    r = rect(ro, a)
    return -r.imag, r.real


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

def state_dict(state):
    return {
        "attacker_x": state[0],
        "attacker_y": state[1],
        "attacker_speed": float(state[2]),
        "attacker_heading": state[3],
        "target_x": state[4],
        "target_y": state[5],
        "target_speed": float(state[6]),
        "target_heading": state[7],
        "acas_state": int(state[8]),
        "win_zone_x": state[9],
        "win_zone_y": state[10],
        "loose_zone_x": state[11],
        "loose_zone_y": state[12],
    }

def render_from_state(state):
    img = get_np_images(
        np.array([state["win_zone_x"], state["win_zone_y"]]),
        np.array([state["loose_zone_x"], state["loose_zone_y"]]),
        np.array([state["attacker_x"], state["attacker_y"]]),
        np.array([state["target_x"], state["target_y"]]),
    )
    return img

def get_corners(center, size=10):
    corners = np.array(
        [
            round(center[1] / 100) - size,
            round(center[1] / 100) + size,
            round(center[0] / 100) - size,
            round(center[0] / 100) + size,
        ],
        dtype=int,
    )
    return np.clip(corners, 0, 1000)

def get_np_images(p1, p2, p3, p4):
    img = np.zeros((1000, 1000, 3), dtype="uint8")
    corners = get_corners(p1, 20)
    img[corners[0] : corners[1], corners[2] : corners[3], 0] = 255
    img[corners[0] : corners[1], corners[2] : corners[3], 2] = 255
    corners = get_corners(p2, 20)
    img[corners[0] : corners[1], corners[2] : corners[3], 1] = 255
    img[corners[0] : corners[1], corners[2] : corners[3], 2] = 255
    corners = get_corners(p3)
    img[corners[0] : corners[1], corners[2] : corners[3], 0] = 100
    img[corners[0] : corners[1], corners[2] : corners[3], 1] = 250
    corners = get_corners(p4)
    img[corners[0] : corners[1], corners[2] : corners[3], 0] = 255
    return img[::-1]
