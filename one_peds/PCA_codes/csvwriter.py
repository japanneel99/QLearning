
from configparser import Interpolation
from re import I
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from pandas import DataFrame
from sklearn import datasets
from mpl_toolkits import mplot3d


def save_csv():
    # q_table = np.load(
    #     f"New_Experiments/Expt23_100/Q_tables/100_model/q_table.npy")

    q_table = np.load("0_model/q_table.npy")

    q_r = []
    r_d = []
    r_t = []
    r_v = []
    a = []

    for rd in range(10):
        for rt in range(6):
            for rv in range(2):
                for action in range(17):
                    q = q_table[rd, rt, rv, action]
                    q_r.append([rd, rt, rv, action, q])

    print(q_table)
    print(q_r)

    df = pd.DataFrame(q_r, columns=[
        "relative_distance_index", "relative_angle_index", "relative_velocity_index", "action_index", "q_value"])

    # df.to_csv(
    #     'New_Experiments/Expt23_100/Data/q_table.csv', index=False)

    df.to_csv('test.csv')


def plot_graph():
    # Data1D = pd.read_csv(
    #     'New_Experiments/Expt23_100/Data/q_table.csv')

    Data1D = pd.read_csv('test.csv')

    x = Data1D['relative_distance_index'].values
    y = Data1D['relative_angle_index'].values
    z = Data1D['q_value'].values
    a = Data1D['relative_velocity_index'].values
    acc = Data1D['action_index'].values

    x_1 = np.array(x)
    y_1 = np.array(y)
    #y_1 = np.array(a)
    z_1 = np.array(z)
    a_mesh = np.array(acc)

    # create a 2d x, y grid (both x and y will be 2D)
    X, Y = np.meshgrid(x_1, y_1)

    # repeat Z to make it a 2d grid
    Z = np.tile(z_1, (len(z_1), 1))

    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    surf = ax.plot_surface(X, Y, Z, edgecolors='yellow', vmin=-100, vmax=100)
    ax.set_zlim(-100, 80)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.zaxis.set_tick_params(labelsize=16)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_wireframe(X, Y, Z)
    ax.set_zlim(-100, 80)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.zaxis.set_tick_params(labelsize=16)

    plt.show()

    # 2d  plot of relative distance index vs q_table and relative angle vs q_table
    fig, axs = plt.subplots(2)
    axs[0].scatter(X, Z)
    axs[1].scatter(Y, Z, c='orangered')


def main():
    if input("Save values in a csv for 3D plotting and GMM? (y/n)").strip() == "y":
        save_csv()

    if input("Plot a surface plot to visualize the Q values, surface and mesh plot? (y/n)").strip() == "y":
        plot_graph()
        plt.show()


if __name__ == "__main__":
    main()
