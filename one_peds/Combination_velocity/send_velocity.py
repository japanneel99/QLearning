from venv import create
import numpy as np
import math
from environment_combi import SimulatorEnvironment
import pandas as pd
import csv as csv
from copy import deepcopy
from qtable_agent_peds import QtableAgent


def create_environment(is_unity):
    dt = 0.01
    W1 = (10*1/3)
    W2 = (1/5 * 1/1.6)
    W3 = 5
    W4 = 5
    a = 0.2  # 行動ではなくガウス分布のa
    if is_unity:
        pass
    else:
        environment = SimulatorEnvironment(dt, W1, W2, W3, W4, a)
    return environment


def get_velocity(v_table):

    velocity_states = []

    for i_distance in range(10):
        for i_angle in range(6):
            for i_velocity in range(2):
                velocity_state = v_table[i_distance, i_angle, i_velocity]
                velocity_states.append(
                    [i_distance, i_angle, i_velocity, velocity_state])
    print('velocity states', velocity_states)

    return velocity_states


def send_velocity(environment, agent, velocity):

    states_distance = []
    states_angle = []
    states_velocity = []
    distance = []

    states_list = []

    states = environment.reset()
    terminal = False

    action_send = []

    for i in range(len(velocity)):
        actions = velocity[i][3]
        action_send.append(actions)

    print(action_send)


def main():

    v_table = np.load('v_table.npy')

    velocities = np.linspace(0, 1.6, num=17)

    is_unity = input(
        "Use_unity_environment to do experiment?(y/n)").strip() == "y"
    environment = create_environment(is_unity)

    agent = QtableAgent(action_candidates=np.linspace(0, 1.6, 8),
                        quantization=[(0.4, 16.0, 4),
                                      (0.4, 16, 4),
                                      (-math.pi, +math.pi, 3),
                                      (-math.pi, +math.pi, 3),
                                      (-1.4, 1.4, 2),
                                      (-1.4, 1.4, 2)
                                      ],
                        epsilon=0.1,
                        alpha=0.1,
                        gamma=0.9)

    velocity = get_velocity(v_table)

    send_velocity(environment, agent, velocity)


if __name__ == "__main__":
    main()
