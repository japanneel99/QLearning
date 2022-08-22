from pickle import FALSE
from turtle import pos, position
from matplotlib import animation
from tensorforce.agents import Agent
import numpy as np
from environment_four import SimulatorEnvironment
from unity_environment_four import UnityEnvironment
from qtable_agentFour import QtableAgent
from copy import deepcopy
import matplotlib.pyplot as plt
import math
import os
import csv as csv
import pandas as pd
from matplotlib.animation import FuncAnimation


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


def visualize_q_table(agent, environment):
    dt = 0.01

    states_list, actions_list, terminal_list, reward_list, feeling_list, position_data = simulate(
        environment, agent, False, custom_epsilon=0.0)

    states_list = np.array(states_list)
    relative_distance_p1 = states_list[:, 0]
    relative_distance_p2 = states_list[:, 1]
    relative_distance_p3 = states_list[:, 2]
    relative_distance_p4 = states_list[:, 3]
    relative_angle_p1 = states_list[:, 4]
    relative_angle_p2 = states_list[:, 5]
    relative_angle_p3 = states_list[:, 6]
    relative_angle_p4 = states_list[:, 7]
    relative_velocity_p1 = states_list[:, 8]
    relative_velocity_p2 = states_list[:, 9]
    relative_velocity_p3 = states_list[:, 10]
    relative_velocity_p4 = states_list[:, 11]
    time_step = np.linspace(
        0, dt*len(relative_distance_p1), num=len(relative_distance_p1))

    rewards_episode = []
    episode = []

    for i in range(len(reward_list)):
        step = reward_list[i]
        x = i
        rewards_episode.append(step)
        episode.append(x)

    for i in range(10):
        print((i+1)+10, "mean_episode_rewards",
              np.mean(rewards_episode[10*i: 10*(i+1)]))

    rewards_per_10 = []
    episodes_10 = []

    for i in range(10):
        x_1 = (i+1)*10
        y_1 = np.mean(rewards_episode[10*i: 10*(i+1)])
        episodes_10.append(x_1)
        rewards_per_10.append(y_1)

    position = np.array(position_data)

    rx = position[:, 0]
    rz = position[:, 2]
    p1x = position[:, 3]
    p1z = position[:, 5]
    p2x = position[:, 6]
    p2z = position[:, 8]
    p3x = position[:, 9]
    p3z = position[:, 11]
    p4x = position[:, 12]
    p4z = position[:, 14]

    plt.subplot(3, 1, 1)
    plt.title("epoch = 200 Human feeling")
    plt.plot(time_step, relative_distance_p1)
    plt.plot(time_step, relative_distance_p2)
    plt.plot(time_step, relative_distance_p3)
    plt.plot(time_step, relative_angle_p4)
    plt.ylabel("relative distance")
    plt.xlabel("Time step")
    plt.ylim(0, 14)
    plt.subplot(3, 1, 2)
    plt.plot(time_step, relative_angle_p1)
    plt.plot(time_step, relative_angle_p2)
    plt.plot(time_step, relative_angle_p3)
    plt.plot(time_step, relative_angle_p4)
    plt.ylim(-4, 4)
    plt.ylabel("relative angle")
    plt.xlabel("Time step")
    plt.subplot(3, 1, 3)
    plt.plot(time_step, actions_list)
    plt.ylim(0, 2)
    plt.ylabel("Wheelchair z velocity")
    plt.xlabel("Time step")
    plt.show()

    agent.save(directory="saved_variables")

    if input("Save states data in a CSV file?(y/n)") == "y":
        with open("Expt1_200L.csv", 'w', newline='') as csvfile:
            fieldnames = ['Time_Step', 'Wheelchair_z_velocity', 'relative_distance_p1', 'relative_angle_p1',
                          'relative_distance_p2', 'relative_angle_p2', 'relative_distance_p3', 'relative_angle_p3',
                          'relative_distance_p4', 'relative_angle_p4']

            thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
            thewriter.writeheader()

            for i in range(len(relative_distance_p1)):
                thewriter.writerow({'Time_Step': time_step[i], 'Wheelchair_z_velocity': actions_list[i], 'relative_distance_p1': relative_distance_p1[i], 'relative_angle_p1': relative_angle_p1[i],
                                    'relative_distance_p2': relative_distance_p2[i], 'relative_angle_p2': relative_angle_p2[i],
                                    'relative_distance_p3': relative_distance_p3[i], 'relative_angle_p3': relative_angle_p3[i],
                                    'relative_distance_p4': relative_distance_p4[i], 'relative_angle_p4': relative_angle_p4[i]})

    if input("Save position data of all agents?(y/n)") == "y":
        with open("Expt1_positionL.csv", 'w', newline='') as csvfile:
            fieldnames = ['robot_x', 'robot_z', 'p1x', 'p1z',
                          'p2x', 'p2z', 'p3x', 'p3z', 'p4x', 'p4z']

            thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
            thewriter.writeheader()

            for i in range(len(rx)):
                thewriter.writerow({'robot_x': rx[i], 'robot_z': rz[i], 'p1x': p1x[i], 'p1z': p1z[i],
                                   'p2x': p2x[i], 'p2z': p2z[i], 'p3x': p3x[i], 'p3z': p3z[i], 'p4x': p4x[i], 'p4z': p4z[i]})


def simulate(environment, agent, learn, custom_epsilon=None):
    states_list = []
    actions_list = []
    terminal_list = []
    reward_list = []
    feeling_list = []
    position_data = []

    states = environment.reset()
    terminal = False
    while not terminal:
        if isinstance(agent, QtableAgent):
            actions = agent.act(states=states, custom_epsilon=custom_epsilon)
        else:
            actions = agent.act(states=states)

        states, terminal, reward, feeling, next_data = environment.execute(
            actions=actions)

        if learn:
            agent.observe(terminal=terminal, reward=reward)

        states_list.append(deepcopy(states))
        actions_list.append(deepcopy(actions))
        terminal_list.append(deepcopy(terminal))
        reward_list.append(deepcopy(reward))
        feeling_list.append(deepcopy(feeling))
        position_data.append(deepcopy(next_data))

    return states_list, actions_list, terminal_list, reward_list, feeling_list, position_data


def learn(environment, agent, n_epoch, save_every_10, use_experience):
    states_learned = []
    actions_learned = []
    terminal_learned = []
    rewards_learned = []
    feeling_learned = []
    positions_learned = []

    for i in range(n_epoch):
        print("%d th learning......" % i)

        states_list, actions_list, terminal_list, reward_list, feeling_list, position_data = simulate(
            environment, agent, not use_experience, None)

        states_learned.append(deepcopy(states_list))
        actions_learned.append(deepcopy(actions_list))
        terminal_learned.append(deepcopy(terminal_list))
        rewards_learned.append(deepcopy(reward_list))
        feeling_learned.append(deepcopy(feeling_list))
        positions_learned.append(deepcopy(position_data))

        if i % 50 == 0 and save_every_10:
            directory = "%d_model" % i
            if not os.path.exists(directory):
                os.mkdir(directory)
                agent.save(directory)

        if use_experience:
            agent.experience(states=states_list, actions=actions_list,
                             terminal=terminal_list, reward=reward_list)
            agent.update()

    return states_learned, actions_learned, terminal_learned, rewards_learned, feeling_learned, positions_learned


def main():
    is_unity = input(
        "Use unity environment to do experiment?(y/n)").strip() == "y"
    environment = create_environment(is_unity)

    if input("Use q-table Q-Learning?(y/n)").strip() == "n":
        agent = Agent.create(agent='tensorfoce', environment=environment, update=64, optimizer=dict(
            optimizer='adam', learning_rate=1e-3), objective='policy_gradient', reward_estimation=dict(horizon=20))

    elif input("load saved q-table from simulator?(y/n)") == "n":
        agent = QtableAgent(action_candidates=np.linspace(0, 1.6, 8),
                            quantization=[(0.4, 40.0, 4),
                                          (0.4, 40.0, 4),
                                          (0.4, 40.0, 4),
                                          (0.4, 40.0, 4),
                                          (-math.pi, +math.pi, 3),
                                          (-math.pi, +math.pi, 3),
                                          (-math.pi, +math.pi, 3),
                                          (-math.pi, +math.pi, 3),
                                          (-1.4, 1.4, 2),
                                          (-1.4, 1.4, 2),
                                          (-1.4, 1.4, 2),
                                          (-1.4, 1.4, 2)
                                          ],
                            epsilon=0.1,
                            alpha=0.1,
                            gamma=0.9)
    else:
        n = int(input("Which generation do you want to load?").strip())
        directory = "%d_model" % n
        agent = QtableAgent.load(directory)

    if input("only visualization? - This will make the graph smooth.(y/n)").strip() == "y":
        visualize_q_table(agent, environment)
        return

    states_learned, actions_learned, terminal_learned, rewards_learned, feeling_learned, positions_learned = learn(
        environment, agent, n_epoch=201, save_every_10=True, use_experience=False)

    agent.save("saved_variables")

    visualize_q_table(agent, environment)


if __name__ == "__main__":
    main()
