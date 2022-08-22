from ctypes import Union
from venv import create
import numpy as np
import math
from environment_combi import SimulatorEnvironment
import pandas as pd
from typing import Union
import csv as csv
from copy import deepcopy
from qtable_agent_peds import QtableAgent

"""
Union - 

isinstance - 

clip - Values outside the intervals are clipped to the interval edges. For example if the interval is [0, 1] and
the value is smaller than 0 it becomes 0 and if it is greater than 1 becomes 1. 
"""


def quantization(x: Union[float, np.ndarray], x_min: float, x_max: float, n_quad: int) -> Union[int, np.ndarray]:
    index = ((x - x_min) / (x_max - x_min) * n_quad)
    if isinstance(index, np.ndarray):
        retval = index.astype(int)
        return np.clip(retval, 0, n_quad-1)
    else:
        retval = int(index)
        return max(0, min(n_quad - 1, retval))


def create_environment(is_unity):
    dt = 0.01
    W1 = (10 * 1/3)
    W2 = (1/5 * 1/1.6)
    W3 = 5
    W4 = 5
    a = 0.2
    if is_unity:
        pass
    else:
        environment = SimulatorEnvironment(dt, W1, W2, W3, W4, a)
    return environment


def get_action(q_table1, q_table2, velocities):
    q_total = q_table1 + q_table2

    n_distance = q_total.shape[0]
    n_angle = q_total.shape[1]
    n_velocity = q_total.shape[2]

    best_actions = []

    v_table = np.zeros((n_distance, n_angle, n_velocity))
    velocities_table = []
    for i_distance in range(n_distance):
        for i_angle in range(n_angle):
            for i_velocity in range(n_velocity):
                i_action = np.argmax(
                    q_total[i_distance, i_angle, i_velocity, :])  # with this we get the action index
                best_actions.append(i_action)
                v_table[i_distance, i_angle, i_velocity] = velocities[i_action]
                velocities_table.append([i_distance, i_angle,
                                         i_velocity, velocities[i_action]])

    print('velocities_table', velocities_table)

    return velocities_table


def pass_velocity(environment, velocities_table, xmin, xmax, amin, amax, vmin, vmax, n_d, n_a, n_v):

    states = environment.reset()

    # discretize the initial states
    d1_discrete_initial = quantization(states[0], xmin, xmax, n_d)
    d2_discrete_initial = quantization(states[1], xmin, xmax, n_d)
    a1_discrete_initial = quantization(states[2], amin, amax, n_a)
    a2_discrete_initial = quantization(states[3], amin, amax, n_a)
    v1_discrete_initial = quantization(states[4], vmin, vmax, n_v)
    v2_discrete_initial = quantization(states[5], vmin, vmax, n_v)

    initial_discrete_states = [d1_discrete_initial, d2_discrete_initial,
                               a1_discrete_initial, a2_discrete_initial, v1_discrete_initial, v2_discrete_initial]

    velocity_at_states = []
    for i in range(len(velocities_table)):
        velocity_at_states.append(velocities_table[i][3])
    # print(velocity_at_states)

    for i_distance in range(len(velocities_table[0, :])):
        for i_angle in range(len(velocities_table[1, :])):
            for i_velocity in range(len(velocities_table[2, :])):
                for i in range(len(velocity_at_states)):
                    pass


def main():
    q_table1 = np.load(
        'New_Experiments/Expt23_100/Q_tables/100_model/q_table.npy')

    q_table2 = np.load(
        'New_Experiments/Expt24_100/Q_tables/100_model/q_table.npy')

    is_unity = input(
        "Use the unity environment to visualize? (y/n)").strip == "n"
    environment = create_environment(is_unity)

    agent = QtableAgent(
        action_candidates=np.linspace(0, 1.6, 17),
        quantization=[
            (0.4, 40.0, 10),
            (-math.pi, math.pi, 6),
            (-1.4, 1.4, 2)],
        epsilon=0.1,
        alpha=0.1,
        gamma=0.9
    )

    velocities = np.linspace(0, 1.6, num=17)

    velocities_table = get_action(q_table1, q_table2, velocities)

    pass_velocity(environment, velocities_table, 0.4, 40.0, -
                  math.pi, math.pi, -1.4, 1.4, 10, 6, 2)


if __name__ == "__main__":
    main()
