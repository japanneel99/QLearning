import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Data = pd.read_csv('Expt1_position.csv')

rx = Data['robot_x'].values
rz = Data['robot_z'].values
p1x = Data['p1x'].values
p1z = Data['p1z'].values
p2x = Data['p2x'].values
p2z = Data['p2z'].values
p3x = Data['p3x'].values
p3z = Data['p3z'].values
p4x = Data['p4x'].values
p4z = Data['p4z'].values

rx_1 = rx[::10]
rz_1 = rz[::10]
p1x1 = p1x[::10]
p1z1 = p1z[::10]
p2x1 = p2x[::10]
p2z1 = p2z[::10]
p3x1 = p3x[::10]
p3z1 = p3z[::10]
p4x1 = p4x[::10]
p4z1 = p4z[::10]

fig = plt.figure()
plt.title(
    "Dynamic graph showing the position of the wheelchair and position epoch=50")
plt.xlabel("x_position")
plt.ylabel("z_position")

i = len(rx_1)


def animate(i):
    print("drawing (%d/%d)" % (i, len(rx_1)))
    if i != 0:
        plt.cla()
    plt.xlim(-5, 15)
    plt.ylim(0, 22)
    plt.scatter(rx_1[i], rz_1[i], label="robot")
    plt.scatter(p1x1[i], p1z1[i], label='ped1')
    plt.scatter(p2x1[i], p2z1[i], label='ped2')
    plt.scatter(p3x1[i], p3z1[i], label='ped3')
    plt.scatter(p4x1[i], p4z1[i], label='ped4')


animation = FuncAnimation(
    fig, animate, frames=len(rx), interval=1)

plt.legend()
plt.show()

plt.close()
