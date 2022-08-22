import math
import numpy as np


def relative_distance(rx, rz, px, pz):
    r_d = math.sqrt((rz - pz)**2 + (pz - px)**2)

    return r_d


def relative_angle(rx, rz, px, pz):
    r_t = math.atan2((px - rx), (pz - rz))

    return r_t


def relative_velocity(rvx, pvx):
    r_v = pvx - rvx

    return r_v
