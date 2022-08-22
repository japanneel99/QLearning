import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Data = pd.read_csv('New_Experiments/Expt23_100/Data/Expt23_PosL.csv')
# Data = pd.read_csv('Expt7_Pos.csv')

rx = Data['robot_x'].values
rz = Data['robot_z'].values
px = Data['pedestrian_x'].values
pz = Data['pedestrian_z'].values


rx_1 = rx[::10]
rz_1 = rz[::10]
px_1 = px[::10]
pz_1 = pz[::10]

fig = plt.figure()
plt.title(
    "Dynamic graph showing the position of the wheelchair and position epoch=100")
plt.xlabel('x_position')
plt.ylabel('z_position')

i = len(rx_1)

print(len(rx_1))


def animate(i):
    print("drawing (%d/%d)" % (i, len(rx_1)))
    if i != 0:
        plt.cla()
        plt.xlim(3, 10)
        plt.ylim(-1, 10)
        plt.scatter(rx_1[i], rz_1[i], label='Robot')
        plt.scatter(px_1[i], pz_1[i], label="Pedestrian")


animation = FuncAnimation(fig, animate, frames=len(rx_1), interval=1)

plt.legend()
plt.show()

plt.savefig("dynamic_velocity.png")

plt.close()
