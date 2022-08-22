from msilib.schema import Directory
import numpy as np
from tensorforce.agents import Agent
from environment_two import SimulatorEnvironment
from unity_split import UnityEnvironment
from Qtable_agent import QtableAgent
from copy import deepcopy
from Qtable_agent import quantization
from Qtable_agent import QtableAgent
import matplotlib.pyplot as plt
import math
import os
import csv as csv
import pandas as pd


def create_environment(is_unity):
    dt = 0.01
    W1 = (10*1/3)
    W2 = (1/5*1/1.6)
    W3 = 5
    W4 = 5
    a = 0.2
    if is_unity:
        environment = UnityEnvironment(dt, W1, W2, W3, W4, a)
    else:
        environment = SimulatorEnvironment(dt, W1, W2, W3, W4, a)
    return environment


def visualize_simulation(environment,  q_table1, q_table2, velocities, dt, mode):
    if mode == "max":
        statesp1_list, statesp2_list, actions_list = send_velocity(
            environment, q_table1, q_table2, velocities)

    statesp1_list = np.array(statesp1_list)
    statesp2_list = np.array(statesp2_list)
    relative_distancep1 = statesp1_list[:, 0]
    relative_anglep1 = statesp1_list[:, 1]
    relative_distancep2 = statesp2_list[:, 0]
    relative_anglep2 = statesp2_list[:, 1]
    time_step = np.linspace(0, dt*len(relative_distancep1),
                            num=len(relative_distancep1))

    plt.subplot(3, 1, 1)
    plt.plot(time_step, relative_distancep1,
             color='black', linewidth=5, label="Ped A")
    plt.plot(time_step, relative_distancep2,
             color='red', linewidth=5, label="Ped B")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.subplot(3, 1, 2)
    plt.plot(time_step, relative_anglep1,
             color='black', linewidth=5, label="Ped A")
    plt.plot(time_step, relative_anglep2,
             color='red', linewidth=5, label="Ped B")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.subplot(3, 1, 3)
    plt.plot(time_step, actions_list, color='green', linewidth=5)
    plt.ylim(0, 2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.show()


def send_velocity(environment, q_table1, q_table2, velocities):
    statesp1_list = []
    statesp2_list = []
    actions_list = []

    n_distance = q_table1.shape[0]
    n_angle = q_table1.shape[1]
    n_velocity = q_table1.shape[2]

    v_table = np.zeros((n_distance, n_angle, n_velocity))

    statep1, statep2 = environment.reset()
    terminal = False

    while not terminal:

        distance_p1_real = statep1[0]
        angle_p1_real = statep1[1]
        velocityp1_real = statep1[2]

        distance_p2_real = statep2[0]
        angle_p2_real = statep2[1]
        velocityp2_real = statep2[2]

        distancep1 = quantization(distance_p1_real, 0.4, 40.0, 10)
        anglep1 = quantization(angle_p1_real, -math.pi, math.pi, 6)
        velocityp1 = quantization(velocityp1_real, -1.4, 1.4, 2)

        distancep2 = quantization(distance_p2_real, 0.4, 40.0, 10)
        anglep2 = quantization(angle_p2_real, -math.pi, math.pi, 6)
        velocityp2 = quantization(velocityp2_real, -1.4, 1.4, 2)

        q_total = (((distance_p2_real) / (distance_p1_real + distance_p2_real)) * q_table1) + \
            (((distance_p1_real) / (distance_p1_real + distance_p2_real)) * q_table2)

        # now calculate the expected velocity from this
        for i_distance in range(n_distance):
            for i_angle in range(n_angle):
                for i_velocity in range(n_velocity):
                    total = np.sum(q_total[i_distance, i_angle, i_velocity, :])
                    if total == 0:
                        v_table[i_distance, i_angle, i_velocity] == -1
                    else:
                        probabilities = q_total[i_distance,
                                                i_angle, i_velocity, :] / total
                        v_table[i_distance, i_angle, i_velocity] = np.sum(
                            probabilities * velocities)

        if q_total[distancep1, anglep1, velocityp1].any() >= q_total[distancep2, anglep2, velocityp2].any():
            velocity_action = v_table[distancep1, anglep1, velocityp1]
        else:
            velocity_action = v_table[distancep2, anglep2, velocityp2]

        print(velocity_action)

        statep1, statep2, terminal, reward, feeling, next_state = environment.execute(
            actions=velocity_action)

        statesp1_list.append(deepcopy(statep1))
        statesp2_list.append(deepcopy(statep2))
        actions_list.append(deepcopy(velocity_action))

    return statesp1_list, statesp2_list, actions_list


def main():

    is_unity = input(
        "Use unity environment to do experiment?(y/n)").strip() == "y"
    environment = create_environment(is_unity)

    q_table1 = np.load(
        'New_Experiments/Expt37_100/Q_tables/100_model/q_table.npy')

    q_table2 = np.load(
        'New_Experiments/Expt39_100/Q_tables/100_model/q_table.npy')

    velocities = np.linspace(0, 1.6, 17)
    n_actions = len(velocities)
    dt = 0.01

    send_velocity(environment, q_table1, q_table2, velocities)

    visualize_simulation(environment, q_table1, q_table2,
                         velocities, dt, mode="max")


if __name__ == '__main__':
    main()
