import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import math
import csv as csv
import pandas as pd
import sys

# np.clip anthing less than 0 will becomes 0 and anything > n_quad - 1 will be n_quad - 1
# if statement in quantization is upgrading to multi-dimensional numpy array


def quantization(x: Union[float, np.ndarray], x_min: float, x_max: float, n_quad: int) -> Union[int, np.ndarray]:
    index = ((x - x_min) / (x_max - x_min) * n_quad)
    if isinstance(index, np.ndarray):
        retval = index.astype(int)
        return np.clip(retval, 0, n_quad - 1)
    else:
        retval = int(index)
        return max(0, min(n_quad - 1, retval))


def create_v_table(q_table1, q_table2, velocities, mode="max_v"):
    """will add the Q tables for each pedestrian then create a v table and find the optimal"""

    q_total = q_table1 + q_table2

    n_distance = q_total.shape[0]
    n_angle = q_total.shape[1]
    n_velocity = q_total.shape[2]

    v_table = np.zeros((n_distance, n_angle, n_velocity))
    for i_distance in range(n_distance):
        for i_angle in range(n_angle):
            for i_velocity in range(n_velocity):
                total = np.sum(q_total[i_distance, i_angle, i_velocity, :])
                if total == 0:
                    v_table[i_distance, i_angle, i_velocity] == -1
                elif mode == "expected_v":
                    probabilities = q_total[i_distance,
                                            i_angle, i_velocity, :] / total

                    # 期待速度を計算する
                    v_table[i_distance, i_angle, i_velocity] = np.sum(
                        probabilities * velocities)

                elif mode == "max_v":
                    i = np.argmax(q_total[i_distance, i_angle, i_velocity, :])
                    v_table[i_distance, i_angle, i_velocity] = velocities[i]
    print('v_table', v_table)
    v_t = np.save("v_table.npy", v_table)

    return v_table


def get_expected_v(x, y, velocity_index, v_table):
    """get expected velocity at a specific state/position 

    :param x: relative x position 
    :param y: relative y position
    :param velocity_index: velocity_index 
    :param v_table: v_table 
    :return: This function will return the expected velocity at position (x, y) and velocity_index 
    """

    distance = np.sqrt(x**2, y**2)
    distance_index = quantization(distance, 0.4, 40, v_table.shape[0])
    angle = np.arctan2(y, x)
    angle_index = quantization(angle, -math.pi, math.pi, v_table.shape[1])

    return v_table[distance_index, angle_index, velocity_index]


def plot_v_table(v_table, velocity_index, min_v, max_v, n_resolution=500):
    relative_xs = np.linspace(-10, 10, num=n_resolution)
    relative_ys = np.linspace(-10, 10, num=n_resolution)
    mesh_relative_xs, mesh_relative_ys = np.meshgrid(relative_xs, relative_ys)
    z_mesh = get_expected_v(
        mesh_relative_xs, mesh_relative_ys, velocity_index, v_table)

    plt.pcolormesh(relative_xs, relative_ys, z_mesh,
                   shading="auto", vmin=min_v, vmax=max_v)
    plt.xlabel("relative x position", fontsize=30)
    plt.ylabel("relative y position", fontsize=30)
    plt.tick_params(axis='x', labelsize=26)
    plt.tick_params(axis='y', labelsize=26)
    plt.rc('xtick', labelsize=26)
    plt.rc('ytick', labelsize=26)
    # plt.title(
    #     'A heat map showing the velocity at a given position, feelings considered', fontsize=30)
    plt.colorbar()
    plt.show()


def main():
    q_table1 = np.load(
        'New_Experiments/Expt23_100/Q_tables/100_model/q_table.npy')

    q_table2 = np.load(
        'New_Experiments/Expt24_100/Q_tables/100_model/q_table.npy')

    velocities = np.linspace(0, 1.6, num=17)
    mode = "max_v"

    v_table = create_v_table(q_table1, q_table2, velocities, mode=mode)

    # plot_v_table(v_table, velocity_index=1, min_v=-1, max_v=1.6)


if __name__ == "__main__":
    main()
