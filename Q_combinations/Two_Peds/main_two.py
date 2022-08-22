from msilib.schema import Directory
from re import I
from turtle import distance
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
    W1 = (0.6*1/3)
    W2 = (0.1*1/1.6)
    W3 = 0.15
    W4 = 0.15
    a = 0.2
    if is_unity:
        environment = UnityEnvironment(dt, W1, W2, W3, W4, a)
    else:
        environment = SimulatorEnvironment(dt, W1, W2, W3, W4, a)
    return environment


def visualize_simulation(environment, q_table1, q_table2, velocities, dt, method):

    statesp1_list, statesp2_list, actions_list = send_velocity(
        environment, q_table1, q_table2, velocities, method)

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
             color='black', linewidth=5, label="Ped 1")
    plt.plot(time_step, relative_distancep2,
             color='red', linewidth=5, label="Ped 2")
    plt.legend(loc="upper right")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 10)

    plt.subplot(3, 1, 2)
    plt.plot(time_step, relative_anglep1,
             color='black', linewidth=5, label="Ped 1")
    plt.plot(time_step, relative_anglep2,
             color='red', linewidth=5, label="Ped 2")
    plt.legend(loc="upper right")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(-3, 3)

    plt.subplot(3, 1, 3)
    plt.plot(time_step, actions_list, color='green', linewidth=5)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 2)

    plt.legend()

    plt.show()


def visualize_all(data1, data2, data3):
    t1_q = data1['time_step'].values
    d1_q = data1['relative_distancep1'].values
    a1_q = data1['relative_anglep1'].values
    d2_q = data1['relative_distancep2'].values
    a2_q = data1['relative_anglep2'].values
    vr_q = data1['action'].values

    t1_ca = data2['time_step'].values
    d1_ca = data2['relative_distancep1'].values
    a1_ca = data2['relative_anglep1'].values
    d2_ca = data2['relative_distancep2'].values
    a2_ca = data2['relative_anglep2'].values
    vr_ca = data2['action'].values

    t1_sa = data3['time_step'].values
    d1_sa = data3['relative_distancep1'].values
    a1_sa = data3['relative_anglep1'].values
    d2_sa = data3['relative_distancep2'].values
    a2_sa = data3['relative_anglep2'].values
    vr_sa = data3['action'].values

    plt.subplot(3, 1, 1)
    plt.plot(t1_q, d1_q, label='rd_p1_Q', linewidth=3)
    plt.plot(t1_ca, d1_ca, label='rd_p1_ca', linewidth=3)
    plt.plot(t1_sa, d1_sa, label='rd_p1_sa', linewidth=3)
    plt.plot(t1_q, d2_q, label='rd_p2_Q', linewidth=3)
    plt.plot(t1_ca, d2_ca, label='rd_p2_ca', linewidth=3)
    plt.plot(t1_sa, d2_sa, label='rd_p2_sa', linewidth=3)
    plt.legend(loc='lower right')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 10)

    plt.subplot(3, 1, 2)
    plt.plot(t1_q, a1_q, label='ra_p1_Q', linewidth=3)
    plt.plot(t1_ca, a1_ca, label='ra_p1_ca', linewidth=3)
    plt.plot(t1_sa, a1_sa, label='ra_p1_sa', linewidth=3)
    plt.plot(t1_q, a2_q, label='ra_p2_Q', linewidth=3)
    plt.plot(t1_ca, a2_ca, label='ra_p2_ca', linewidth=3)
    plt.plot(t1_sa, a2_sa, label='ra_p2_sa', linewidth=3)
    plt.legend()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(-3, 3)

    plt.subplot(3, 1, 3)
    plt.plot(t1_q, vr_q, label='combined Q', linewidth=3)
    plt.plot(t1_ca, vr_ca, label='Combined Action', linewidth=3)
    plt.plot(t1_sa, vr_sa, label='Seperate Action', linewidth=3)
    plt.legend()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 2)

    plt.show()


def save_data(environment, q_table1, q_table2, velocities, dt, method="combined_q"):
    statesp1_list, statesp2_list, actions_list = send_velocity(
        environment, q_table1, q_table2, velocities, method)

    statesp1_list = np.array(statesp1_list)
    statesp2_list = np.array(statesp2_list)
    relative_distancep1 = statesp1_list[:, 0]
    relative_anglep1 = statesp1_list[:, 1]
    relative_distancep2 = statesp2_list[:, 0]
    relative_anglep2 = statesp2_list[:, 1]
    time_step = np.linspace(0, dt*len(relative_distancep1),
                            num=len(relative_distancep1))

    if input("Save data in a CSV file?(y/n)") == "y":
        if method == "combine_q":
            with open("combine_q.csv", 'w', newline='') as csvfile:
                fieldnames = ['time_step', 'relative_distancep1', 'relative_anglep1',
                              'relative_distancep2', 'relative_anglep2', 'action']

                thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
                thewriter.writeheader()

                for i in range(len(relative_distancep1)):
                    thewriter.writerow({'time_step': time_step[i], 'relative_distancep1': relative_distancep1[i], 'relative_anglep1': relative_anglep1[i],
                                        'relative_distancep2': relative_distancep2[i], 'relative_anglep2': relative_anglep2[i], 'action': actions_list[i]})

        if method == "combine_action":
            with open("combine_action.csv", 'w', newline='') as csvfile:
                fieldnames = ['time_step', 'relative_distancep1', 'relative_anglep1',
                              'relative_distancep2', 'relative_anglep2', 'action']

                thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
                thewriter.writeheader()

                for i in range(len(relative_distancep1)):
                    thewriter.writerow({'time_step': time_step[i], 'relative_distancep1': relative_distancep1[i], 'relative_anglep1': relative_anglep1[i],
                                        'relative_distancep2': relative_distancep2[i], 'relative_anglep2': relative_anglep2[i], 'action': actions_list[i]})

        if method == "seperate_action":
            with open("seperate_action.csv", 'w', newline='') as csvfile:
                fieldnames = ['time_step', 'relative_distancep1', 'relative_anglep1',
                              'relative_distancep2', 'relative_anglep2', 'action']

                thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
                thewriter.writeheader()

                for i in range(len(relative_distancep1)):
                    thewriter.writerow({'time_step': time_step[i], 'relative_distancep1': relative_distancep1[i], 'relative_anglep1': relative_anglep1[i],
                                        'relative_distancep2': relative_distancep2[i], 'relative_anglep2': relative_anglep2[i], 'action': actions_list[i]})


def combine_Q_test(q_table1, q_table2, distance_1, distance_2):
    n_distance = q_table1.shape[0]
    n_angle = q_table1.shape[1]
    n_velocity = q_table1.shape[2]
    n_action = q_table1.shape[3]

    q_total = np.zeros((n_distance, n_angle, n_velocity, n_action))

    for i_distance in range(n_distance):
        for i_angle in range(n_angle):
            for i_velocity in range(n_velocity):
                for i_action in range(n_action):
                    q1_sum = np.abs(np.sum(
                        q_table1[i_distance, i_angle, i_velocity, :]))
                    q2_sum = np.abs(np.sum(
                        q_table2[i_distance, i_angle, i_velocity, :]))
                    if q1_sum == 0:
                        q1_total = 1
                    else:
                        q1_total = q1_sum
                    if q2_sum == 0:
                        q2_total = 1
                    else:
                        q2_total = q2_sum
                    q_total[i_distance, i_angle, i_velocity, i_action] = ((1 / q1_total) * (distance_2/(distance_1 + distance_2))) * (
                        q_table1[i_distance, i_angle, i_velocity, i_action]) + ((1 / q2_total) * (distance_1/(distance_1 + distance_2))) * (q_table2[i_distance, i_angle, i_velocity, i_action])
    print(q_total)
    return q_total


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


def Q_totals(qtable1, qtable2):
    n_distance = qtable1.shape[0]
    n_angle = qtable1.shape[1]
    n_velocity = qtable1.shape[2]
    n_action = qtable1.shape[3]

    for i_distance in range(n_distance):
        for i_angle in range(n_angle):
            for i_velocity in range(n_velocity):
                for i_action in range(n_action):
                    q1_sum = np.abs(
                        np.sum(qtable1[i_distance, i_angle, i_velocity, :]))
                    q2_sum = np.abs(
                        np.sum(qtable2[i_distance, i_angle, i_velocity, :]))
                    if q1_sum == 0:
                        q1_total = 1
                    else:
                        q1_total = q1_sum
                    if q2_sum == 0:
                        q2_total = 1
                    else:
                        q2_total = q2_sum

    return q1_total, q2_total


def optimal_velocities(q_table1, q_table2, velocities):
    n_distance = q_table1.shape[0]
    n_angle = q_table1.shape[1]
    n_velocity = q_table1.shape[2]
    n_action = q_table1.shape[3]

    v_table_p1 = np.zeros((n_distance, n_angle, n_velocity))
    v_table_p2 = np.zeros((n_distance, n_angle, n_velocity))

    for i_distance in range(n_distance):
        for i_angle in range(n_angle):
            for i_velocity in range(n_velocity):
                p1_action_index = np.argmax(
                    q_table1[i_distance, i_angle, i_velocity, :])
                p2_action_index = np.argmax(
                    q_table2[i_distance, i_angle, i_velocity, :])
                v_table_p1[i_distance, i_angle,
                           i_velocity] = velocities[p1_action_index]
                v_table_p2[i_distance, i_angle,
                           i_velocity] = velocities[p2_action_index]

    return v_table_p1, v_table_p2


def send_velocity(environment, q_table1, q_table2, velocities, method):
    statesp1_list = []
    statesp2_list = []
    actions_list = []
    position_data = []

    statep1, statep2, next_state = environment.reset()
    terminal = False

    while not terminal:
        distance_p1_real = statep1[0]
        angle_p1_real = statep1[1]
        velocity_p1_real = statep1[2]

        distance_p2_real = statep2[0]
        angle_p2_real = statep2[1]
        velocity_p2_real = statep2[2]

        position_robot_z = next_state[2]
        position_robot_x = next_state[0]

        position_p1_z = next_state[5]
        position_p1_x = next_state[3]

        position_p2_z = next_state[8]
        position_p2_x = next_state[6]

        distancep1 = quantization(distance_p1_real, 0.4, 40, 10)
        anglep1 = quantization(angle_p1_real, -math.pi, math.pi, 6)
        velocityp1 = quantization(velocity_p1_real, -1.4, 1.4, 2)

        distancep2 = quantization(distance_p2_real, 0.4, 40.0, 10)
        anglep2 = quantization(angle_p2_real, -math.pi, math.pi, 6)
        velocityp2 = quantization(velocity_p2_real, -1.4, 1.4, 2)

        if method == "combine_q":

            if position_robot_z <= position_p1_z:
                q_tableCombined = combine_Q_test(
                    q_table1, q_table2, distance_p1_real, distance_p2_real)
                v_table = get_velocity(q_tableCombined, velocities)
                velocity_action = v_table[distancep1, anglep1, velocityp1]
            if position_robot_z > position_p1_z:
                v_table = get_velocity(q_table2, velocities)
                velocity_action = v_table[distancep2, anglep2, velocityp2]
            if position_robot_z > position_p2_z:
                velocity_action = np.amax(velocities)

        if method == "combine_action":
            v_table_p1, v_table_p2 = optimal_velocities(
                q_table1, q_table2, velocities)

            q1_total, q2_total = Q_totals(q_table1, q_table2)

            velocity_p1 = v_table_p1[distancep1, anglep1, velocityp1]
            velocity_p2 = v_table_p2[distancep2, anglep2, velocityp2]
            if position_robot_z > position_p2_z:
                velocity_action = np.amax(velocities)
            else:
                velocity_action = ((distance_p2_real) / ((distance_p1_real + distance_p2_real)) *
                                   velocity_p1) + ((distance_p1_real) / ((distance_p1_real + distance_p2_real)) *
                                                   velocity_p2)

        if method == "seperate_action":
            v_table_p1, v_table_p2 = optimal_velocities(
                q_table1, q_table2, velocities)

            q1_total, q2_total = Q_totals(q_table1, q_table2)

            velocity_p1 = v_table_p1[distancep1, anglep1, velocityp1]
            velocity_p2 = v_table_p2[distancep2, anglep2, velocityp2]
            if position_robot_z < position_p1_z:

                velocity_action = ((distance_p2_real) / ((distance_p1_real + distance_p2_real)) *
                                   velocity_p1) + ((distance_p1_real) / ((distance_p1_real + distance_p2_real)) *
                                                   velocity_p2)

            if position_robot_z > position_p1_z and position_robot_z <= position_p2_z:
                velocity_action = velocity_p2

            if position_robot_z > position_p2_z:
                velocity_action = np.amax(velocities)

        statep1, statep2, terminal, reward, feeling, next_state = environment.execute(
            actions=velocity_action)

        statesp1_list.append(deepcopy(statep1))
        statesp2_list.append(deepcopy(statep2))
        actions_list.append(deepcopy(velocity_action))

        print('velocity_action', velocity_action)

    return statesp1_list, statesp2_list, actions_list


def main():

    if input("Run the simulation?(y/n)").strip() == "y":
        is_unity = input(
            "Use unity environment to do experiment?(y/n)").strip() == "y"
        environment = create_environment(is_unity)

        q_table1 = np.load(
            'New_Experiments/Expt64_100/Q_tables/100_model/q_table.npy')

        q_table2 = np.load(
            'New_Experiments/Expt67_100/Q_tables/100_model/q_table.npy')

        velocities = np.linspace(0, 1.6, 17)
        n_actions = len(velocities)
        dt = 0.01
        method = "combine_q"

        send_velocity(environment, q_table1, q_table2,
                      velocities, method)

        visualize_simulation(environment, q_table1, q_table2,
                             velocities, dt, method)

        save_data(environment, q_table1, q_table2,
                  velocities, dt, method)

    else:
        input("plot all the data?(y/n)").strip == "y"
        data1 = pd.read_csv(
            'New_Experiments/Combined/new_qmethod/37-55/Data/combine_q.csv')
        data2 = pd.read_csv(
            'New_Experiments/Combined/new_qmethod/37-55/Data/combine_action.csv')
        data3 = pd.read_csv(
            'New_Experiments/Combined/new_qmethod/37-55/Data/seperate_action.csv')

        visualize_all(data1, data2, data3)


if __name__ == '__main__':
    main()
