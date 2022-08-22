from pickletools import read_uint2
from re import L
from turtle import color
from xmlrpc.server import resolve_dotted_attribute
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


def plot_q(time, r_d, r_t, r_v, v_robot, q, mode="scatter"):
    times = np.array(time)
    d = np.array(r_d)
    t = np.array(r_t)
    v = np.array(r_v)
    v_r = np.array(v_robot)
    Q = np.array(q)

    if mode == "surface":
        # Create 2D mesh grid
        X, Y = np.meshgrid(d, v_r)

        Z = np.tile(Q, (len(Q), 1))

        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')

        surf = ax.plot_surface(X, Y, Z, edgecolors='yellow')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.zaxis.set_tick_params(labelsize=12)

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot_wireframe(X, Y, Z)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.zaxis.set_tick_params(labelsize=12)
        ax.legend(loc="upper right")
        ax.set_xlim(0, 6)

        plt.show()

    elif mode == "scatter":
        fig = plt.figure(figsize=(12, 12))

        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.scatter(d, v_r, Q)
        ax.legend(loc="upper right")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.zaxis.set_tick_params(labelsize=16)
        ax.set_xlim(0, 6)

        ax = fig.add_subplot(1, 2, 2, projection="3d")
        ax.scatter(t, v_r, Q)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.zaxis.set_tick_params(labelsize=12)

        plt.show()

        # plt.scatter(times, Q)
        # plt.show()


def states_graph(time, r_d, r_t, r_v, v_robot, q, f):
    times = np.array(time)
    d = np.array(r_d)
    t = np.array(r_t)
    v = np.array(r_v)
    v_r = np.array(v_robot)
    Q = np.array(q)
    feeling = np.array(f)

    u_labels = np.unique(feeling)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(3, 1, 1)

    for i in u_labels:
        ax.scatter(v_r[feeling == i], d[feeling == i], label=i)
    # ax.set_xlabel('Wheelchair_Velocity (m/s)', fontsize=16)
    # ax.set_ylabel('Relative Distance (m)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_ylim(0, 10)

    ax = fig.add_subplot(3, 1, 2)
    for i in u_labels:
        ax.scatter(t[feeling == i], d[feeling == i], label=i)
    # ax.set_xlabel('Relative Angle (rad)', fontsize=16)
    # ax.set_ylabel('Relative Distance (m)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_ylim(0, 10)

    ax = fig.add_subplot(3, 1, 3)
    for i in u_labels:
        ax.scatter(v_r[feeling == i], t[feeling == i], label=i)
    # ax.set_xlabel('Wheelchair_Velocity (m/s)', fontsize=16)
    # ax.set_ylabel('Relative Angle (rad)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(loc="upper right")

    plt.show()


def feeling_graph(time, r_d, r_t, r_v, v_robot, q, f):
    times1 = np.array(time)
    times2 = np.array(time)
    times3 = np.array(time)
    d = np.array(r_d)
    t = np.array(r_t)
    v = np.array(r_v)
    v_r = np.array(v_robot)
    Q = np.array(q)
    feeling = np.array(f)

    u_labels = np.unique(feeling)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in u_labels:
        ax.scatter(d[feeling == i], v_r[feeling == i],
                   Q[feeling == i], label=i)

    plt.legend(loc="upper right")
    ax.set_xlim(0, 6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.zaxis.set_tick_params(labelsize=12)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in u_labels:
        ax.scatter(t[feeling == i], v_r[feeling == i],
                   Q[feeling == i], label=i)

    plt.legend(loc="upper right")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.zaxis.set_tick_params(labelsize=12)
    plt.show()


def distance_profile(data1, data2, data3, data4, data5, data6):

    t1 = data1['Time_Step'].values
    vr1 = data1['Wheelchair_z_velocity'].values
    rd1 = data1['relative_distance'].values
    rt1 = data1['relative_angle'].values
    rv1 = data1['relative_velocity'].values
    vr1L = data1['Wheelchair_z_velocity_learned'].values

    t2 = data2['Time_Step'].values
    vr2 = data2['Wheelchair_z_velocity'].values
    rd2 = data2['relative_distance'].values
    rt2 = data2['relative_angle'].values
    rv2 = data2['relative_velocity'].values
    vr2L = data2['Wheelchair_z_velocity_learned'].values

    t3 = data3['Time_Step'].values
    vr3 = data3['Wheelchair_z_velocity'].values
    rd3 = data3['relative_distance'].values
    rt3 = data3['relative_angle'].values
    rv3 = data3['relative_velocity'].values
    vr3L = data3['Wheelchair_z_velocity_learned'].values

    t4 = data4['Time_Step'].values
    vr4 = data4['Wheelchair_z_velocity'].values
    rd4 = data4['relative_distance'].values
    rt4 = data4['relative_angle'].values
    rv4 = data4['relative_velocity'].values
    vr4L = data4['Wheelchair_z_velocity_learned'].values

    t5 = data5['Time_Step'].values
    vr5 = data5['Wheelchair_z_velocity'].values
    rd5 = data5['relative_distance'].values
    rt5 = data5['relative_angle'].values
    rv5 = data5['relative_velocity'].values
    vr5L = data5['Wheelchair_z_velocity_learned'].values

    t6 = data6['Time_Step'].values
    vr6 = data6['Wheelchair_z_velocity'].values
    rd6 = data6['relative_distance'].values
    rt6 = data6['relative_angle'].values
    rv6 = data6['relative_velocity'].values
    vr6L = data6['Wheelchair_z_velocity_learned'].values

    plt.subplot(3, 1, 1)
    plt.plot(t1, rd1, linewidth=3)
    plt.plot(t2, rd2, linewidth=3)
    plt.plot(t3, rd3, linewidth=3)
    plt.plot(t4, rd4, linewidth=3)
    plt.plot(t5, rd5, linewidth=3)
    plt.plot(t6, rd6, linewidth=3)
    # plt.xlabel('Time Step', fontsize=16)
    # plt.ylabel('Relative Distance (m)', fontsize=16)
    plt.ylim(0, 6)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.subplot(3, 1, 2)
    plt.plot(t1, rt1, linewidth=3)
    plt.plot(t2, rt2, linewidth=3)
    plt.plot(t3, rt3, linewidth=3)
    plt.plot(t4, rt4, linewidth=3)
    plt.plot(t5, rt5, linewidth=3)
    plt.plot(t6, rt6, linewidth=3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.xlabel('Time Step', fontsize=16)
    # plt.ylabel('Relative Angle (rad)', fontsize=16)

    plt.subplot(3, 1, 3)
    plt.plot(t1, vr1L, label='epoch = 290', linewidth=3)
    plt.plot(t2, vr2L, label='epoch = 292', linewidth=3)
    plt.plot(t3, vr3L, label='epoch = 294', linewidth=3)
    plt.plot(t4, vr4L, label='epoch = 296', linewidth=3)
    plt.plot(t5, vr5L, label='epoch = 298', linewidth=3)
    plt.plot(t6, vr6L, label='epoch = 300', linewidth=3)
    plt.legend()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.xlabel('Time Step', fontsize=16)
    # plt.ylabel('Wheelchair Velocity (m/s)', fontsize=16)

    plt.show()

    plt.plot(t1, vr1, label='epoch = 290', linewidth=3)
    plt.plot(t2, vr2, label='epoch = 292', linewidth=3)
    plt.plot(t3, vr3, label='epoch = 294', linewidth=3)
    plt.plot(t4, vr4, label='epoch = 296', linewidth=3)
    plt.plot(t5, vr5, label='epoch = 298', linewidth=3)
    plt.plot(t6, vr6, label='epoch = 300', linewidth=3)
    plt.legend()
    # plt.xlabel('Time Step', fontsize=16)
    # plt.ylabel('Wheelchair Velocity (m/s)', fontsize=16)

    plt.show()


def main():

    Data = pd.read_csv(
        'New_Experiments/Expt29_300/Data/Expt29.csv')

    time = Data['Time_Step'].values
    wheelchair_velocity = Data['Wheelchair_z_velocity'].values
    relative_distance = Data['relative_distance'].values
    relative_angle = Data['relative_angle'].values
    relative_velocity = Data['relative_velocity'].values
    feelings = Data['feelings'].values
    rewards = Data['rewards'].values
    Q_values = Data['q_values'].values
    relative_distance_index = Data['rdi'].values
    relative_angle_index = Data['rti'].values

    mode = "scatter"
    # plot_q(time, relative_distance, relative_angle,
    #        relative_velocity, wheelchair_velocity, Q_values, mode)

    # # feeling_graph(time, relative_distance, relative_angle,
    # #               relative_velocity, wheelchair_velocity, Q_values, feelings)

    states_graph(time, relative_distance, relative_angle,
                 relative_velocity, wheelchair_velocity, Q_values, feelings)

    data1 = pd.read_csv(
        'New_Experiments/Expt29_300/csv_steps/Expt28_290.csv')
    data2 = pd.read_csv(
        'New_Experiments/Expt29_300/csv_steps/Expt28_292.csv')
    data3 = pd.read_csv(
        'New_Experiments/Expt29_300/csv_steps/Expt28_294.csv')
    data4 = pd.read_csv(
        'New_Experiments/Expt29_300/csv_steps/Expt28_296.csv')
    data5 = pd.read_csv(
        'New_Experiments/Expt29_300/csv_steps/Expt28_298.csv')
    data6 = pd.read_csv(
        'New_Experiments/Expt29_300/csv_steps/Expt28_300.csv')

    distance_profile(data1, data2, data3, data4, data5, data6)


if __name__ == "__main__":
    main()
