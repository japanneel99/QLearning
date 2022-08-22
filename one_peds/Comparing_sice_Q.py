import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob


def comparing_Q(DataF, DataNF, mode):
    timeF = DataF['Time_Step'].values
    vr_F = DataF['Wheelchair_z_velocity'].values
    d_F = DataF['relative_distance'].values
    t_F = DataF['relative_angle'].values
    v_F = DataF['relative_velocity'].values
    feelingsF = DataF['feelings'].values
    QF = DataF['q_values'].values

    timeNF = DataNF['Time_Step'].values
    vr_NF = DataNF['Wheelchair_z_velocity'].values
    d_NF = DataNF['relative_distance'].values
    t_NF = DataNF['relative_angle'].values
    v_NF = DataNF['relative_velocity'].values
    feelingsNF = DataNF['feelings'].values
    QNF = DataNF['q_values'].values

    if mode == 'surface':
        X, Y = np.meshgrid(d_F, vr_F)
        Z = np.tile(QF, (len(QF), 1))

        X1, Y1 = np.meshgrid(d_NF, vr_NF)
        Z1 = np.tile(QNF, (len(QNF), 1))

        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')

        surf = ax.plot_surface(X, Y, Z, cmap='plasma')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.zaxis.set_tick_params(labelsize=12)
        ax.set_xlim(0, 6)

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax.plot_surface(X1, Y1, Z1, cmap='plasma')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.zaxis.set_tick_params(labelsize=12)
        ax.legend(loc="upper right")
        ax.set_xlim(0, 6)

        plt.show()

    elif mode == "scatter":
        fig = plt.figure(figsize=(12, 12))

        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.scatter(d_F, vr_F, QF, c='green', label='Feeling')
        ax.scatter(d_NF, vr_NF, QNF, c='red', label='No feeling')
        ax.legend(loc="upper right")
        ax.set_zlim(-40, 80)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.zaxis.set_tick_params(labelsize=16)
        ax.set_xlim(0, 6)

        ax = fig.add_subplot(1, 2, 2, projection="3d")
        ax.scatter(t_F, vr_F, QF, c='green', label='Feeling')
        ax.scatter(t_NF, vr_NF, QNF, c='red', label='No feeling')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.zaxis.set_tick_params(labelsize=12)

        plt.show()


def velocity_comparison(DataF, DataNF):

    timeF = DataF['Time_Step'].values
    vr_F = DataF['Wheelchair_z_velocity_learned'].values
    d_F = DataF['relative_distance'].values
    t_F = DataF['relative_angle'].values
    v_F = DataF['relative_velocity'].values

    timeNF = DataNF['Time_Step'].values
    vr_NF = DataNF['Wheelchair_z_velocity_learned'].values
    d_NF = DataNF['relative_distance'].values
    t_NF = DataNF['relative_angle'].values
    v_NF = DataNF['relative_velocity'].values

    plt.subplot(3, 1, 1)
    plt.plot(timeF, d_F, linewidth=3, c='green')
    plt.plot(timeNF, d_NF, linewidth=3, c='red')
    plt.ylim(0, 6)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.subplot(3, 1, 2)
    plt.plot(timeF, t_F, linewidth=3, c='green')
    plt.plot(timeNF, t_NF, linewidth=3, c='red')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.subplot(3, 1, 3)
    plt.plot(timeF, vr_F, linewidth=3, label='With Feeling', c='green')
    plt.plot(timeNF, vr_NF, linewidth=3, label='No Feeling', c='red')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend()

    plt.show()


def main():

    DataF = pd.read_csv(
        'New_Experiments/Expt29_300/Data/Expt29_300L.csv')

    DataNF = pd.read_csv(
        'New_Experiments/Expt29_300NF/Data/Expt29_300NFL.csv')

    Data1 = pd.read_csv(
        'New_Experiments/Expt29_300/Data/Expt29.csv')

    Data2 = pd.read_csv(
        'New_Experiments/Expt29_300NF/Data/Expt29NF.csv')

    velocity_comparison(DataF, DataNF)

    comparing_Q(Data1, Data2, mode="surface")


if __name__ == "__main__":
    main()
