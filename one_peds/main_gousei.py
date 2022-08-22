import numpy as np
from tensorforce.agents import Agent
from environment import SimulatorEnvironment
from unity_environment import UnityEnvironment
from qtable_agent import QtableAgent
from copy import deepcopy
from qtable_agent import quantization
import matplotlib.pyplot as plt
import math
import os
import csv as csv
import pandas as pd


def create_environment(is_unity):
    dt = 0.01
    W1 = (10*1/3)
    W2 = (1/5 * 1/1.6)
    W3 = 5
    a = 0.2
    if is_unity:
        environment = UnityEnvironment(dt, W1, W2, W3, a)
    else:
        environment = SimulatorEnvironment(dt, W1, W2, W3, a)
    return environment


def visualize_q_table(environment, q_table, velocities, n_actions, dt):

    states_list, actions_list, terminal_list, reward_list, feeling_list, position_list = simulate_action(
        environment, q_table, velocities, n_actions)

    states_list = np.array(states_list)
    relative_distances = states_list[:, 0]
    relative_thetas = states_list[:, 1]
    relative_velocities = states_list[:, 2]
    time_step = np.linspace(0, dt*len(relative_distances),
                            num=len(relative_distances))

    position = np.array(position_list)

    rx = position[:, 0]
    rz = position[:, 2]
    px = position[:, 3]
    pz = position[:, 5]

    plt.subplot(3, 1, 1)
    plt.plot(time_step, relative_distances, color='black', linewidth=5)

    plt.subplot(3, 1, 2)
    plt.plot(time_step, relative_thetas, color='green', linewidth=5)

    plt.subplot(3, 1, 3)
    plt.plot(time_step, actions_list, color='red', linewidth=5)

    plt.show()


def get_velocity(q_table, velocities):
    n_distance = q_table.shape[0]
    n_angle = q_table.shape[1]
    n_velocity = q_table.shape[2]

    v_table = np.zeros((n_distance, n_angle, n_velocity))
    for i_distance in range(n_distance):
        for i_angle in range(n_angle):
            for i_velocity in range(n_velocity):
                action_index = np.argmax(
                    q_table[i_distance, i_angle, i_velocity, :])
                v_table[i_distance, i_angle,
                        i_velocity] = velocities[action_index]
    return v_table


def simulate_action(environment, q_table, velocities, n_actions):
    states_list = []
    actions_list = []
    terminal_list = []
    reward_list = []
    feeling_list = []
    position_data = []

    states = environment.reset()
    terminal = False

    while not terminal:

        distance_real = states[0]
        angle_real = states[1]
        velocity_real = states[2]

        distance = quantization(distance_real, 0.4, 40.0, 10)
        angle = quantization(angle_real, -math.pi, math.pi, 6)
        velocity = quantization(velocity_real, -1.4, 1.4, 2)

        v_table = get_velocity(q_table, velocities)

        velocity_action = v_table[distance, angle, velocity]

        states, terminal, reward, feeling, next_data = environment.execute(
            actions=velocity_action)

        states_list.append(deepcopy(states))
        actions_list.append(deepcopy(velocity_action))
        terminal_list.append(deepcopy(terminal))
        reward_list.append(deepcopy(reward))
        feeling_list.append(deepcopy(feeling))
        position_data.append(deepcopy(next_data))

    print("states_list", states_list)
    print("terminals", terminal_list)
    print("actions_list", actions_list)

    return states_list, actions_list, terminal_list, reward_list, feeling_list, position_data


def main():

    is_unity = input(
        "Use unity environment to do experiment?(y/n)").strip() == "y"
    environment = create_environment(is_unity)

    # q_table 'New_Experiments/Expt28_100/Q_tables/100_model/q_table.npy' is very good for an example.
    q_table = np.load(
        'New_Experiments/Expt20_100/Q_tables/100_model/q_table.npy')

    velocities = np.linspace(0, 1.6, 17)
    n_actions = len(velocities)
    dt = 0.01

    get_velocity(q_table, velocities)

    simulate_action(environment, q_table, velocities, n_actions)

    visualize_q_table(environment, q_table, velocities, n_actions, dt)


if __name__ == "__main__":
    main()
