import numpy as np
import random
import math
import matplotlib as plt
from tensorforce.environments import Environment
from math_functions import relative_distance
from math_functions import relative_angle
from math_functions import relative_velocity


class SimulatorEnvironment(Environment):
    def __init__(self, dt, W1, W2, W3, a):
        self.dt = dt
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.a = a

    def states(self):
        return dict(type='float', shape=(12,))

    def actions(self):
        return dict(type='float', min_value=0.0, max_value=1.6)

    def reset(self):
        self.pos_rx = 8.0
        self.pos_ry = 0.0
        self.pos_rz = 0.0
        self.vel_rx = 0.0
        self.vel_ry = 0.0
        self.pos_p1x = 5.0
        self.pos_p1y = 0.0
        self.pos_p1z = 4.0
        self.vel_p1x = 1.4
        self.vel_p1y = 0.0
        self.vel_p1z = 0.0
        self.pos_p2x = 12
        self.pos_p2y = 0.0
        self.pos_p2z = 6.0
        self.vel_p2x = -1.0
        self.vel_p2y = 0.0
        self.vel_p2z = 0.0
        self.pos_p3x = 2.0  # + random.randint(-2, 8)
        self.pos_p3y = 0.0
        self.pos_p3z = 8.0  # + random.randint(-4, 10)
        self.vel_p3x = 1.0  # + random.uniform(-0.2, 0.4)
        self.vel_p3y = 0.0
        self.vel_p3z = 0.0
        self.pos_p4x = 15.0  # + random.randint(-2, 8)
        self.pos_p4y = 0.0
        self.pos_p4z = 10.0  # + random.randint(-4, 10)
        self.vel_p4x = -1.0  # + random.uniform(-0.2, 0.4)
        self.vel_p4y = 0.0
        self.vel_p4z = 0.0

        self.data = np.array([self.pos_rx, self.pos_ry, self.pos_rz, self.pos_p1x, self.pos_p1y, self.pos_p1z, self.pos_p2x,
                             self.pos_p2y, self.pos_p2z, self.pos_p3x, self.pos_p3y, self.pos_p3z, self.pos_p4x, self.pos_p4y, self.pos_p4z])

        self.relative_distance_p1 = relative_distance(
            self.data[0], self.data[2], self.data[3], self.data[5])

        self.relative_distance_p2 = relative_distance(
            self.data[0], self.data[2], self.data[6], self.data[8])

        self.relative_distance_p3 = relative_distance(
            self.data[0], self.data[2], self.data[9], self.data[11])

        self.relative_distance_p4 = relative_distance(
            self.data[0], self.data[2], self.data[12], self.data[14])

        self.relative_angle_p1 = relative_angle(
            self.data[0], self.data[2], self.data[3], self.data[5])

        self.relative_angle_p2 = relative_angle(
            self.data[0], self.data[2], self.data[6], self.data[8])

        self.relative_angle_p3 = relative_angle(
            self.data[0], self.data[2], self.data[9], self.data[11])

        self.relative_angle_p4 = relative_angle(
            self.data[0], self.data[2], self.data[12], self.data[14])

        self.relative_velocity_p1 = relative_velocity(
            self.vel_p1x, self.vel_rx)
        self.relative_velocity_p2 = relative_velocity(
            self.vel_p2x, self.vel_rx)
        self.relative_velocity_p3 = relative_velocity(
            self.vel_p3x, self.vel_rx)
        self.relative_velocity_p4 = relative_velocity(
            self.vel_p4x, self.vel_rx)

        self.state = [self.relative_distance_p1, self.relative_distance_p2,
                      self.relative_distance_p3, self.relative_distance_p4,
                      self.relative_angle_p1, self.relative_angle_p2,
                      self.relative_angle_p3, self.relative_angle_p4,
                      self.relative_velocity_p1, self.relative_velocity_p2, self.relative_velocity_p3, self.relative_velocity_p4]

        return self.state

    def execute(self, actions):
        new_pos_rx = self.vel_rx * self.dt + self.pos_rx
        new_pos_ry = self.vel_ry * self.dt + self.pos_ry
        new_pos_rz = actions * self.dt + self.pos_rz
        new_pos_p1x = self.vel_p1x * self.dt + self.pos_p1x
        new_pos_p1y = self.vel_p1y * self.dt + self.pos_p1y
        new_pos_p1z = self.vel_p1z * self.dt + self.pos_p1z
        new_pos_p2x = self.vel_p2x * self.dt + self.pos_p2x
        new_pos_p2y = self.vel_p2y * self.dt + self.pos_p2y
        new_pos_p2z = self.vel_p2z * self.dt + self.pos_p2z
        new_pos_p3x = self.vel_p3x * self.dt + self.pos_p3x
        new_pos_p3y = self.vel_p3y * self.dt + self.pos_p3y
        new_pos_p3z = self.vel_p3z * self.dt + self.pos_p3z
        new_pos_p4x = self.vel_p4x * self.dt + self.pos_p4x
        new_pos_p4y = self.vel_p4y * self.dt + self.pos_p4y
        new_pos_p4z = self.vel_p4z * self.dt + self.pos_p4z

        feeling = 0

        r_p1 = relative_distance(
            new_pos_rx, new_pos_rz, new_pos_p1x, new_pos_p1z)
        r_p2 = relative_distance(
            new_pos_rx, new_pos_rz, new_pos_p2x, new_pos_p2z)
        r_p3 = relative_distance(
            new_pos_rx, new_pos_rz, new_pos_p3x, new_pos_p3z)
        r_p4 = relative_distance(
            new_pos_rx, new_pos_rz, new_pos_p4x, new_pos_p4z)

        terminal = new_pos_rz > 11

        reward = (self.W2*actions) - self.W3*(math.exp(-(self.a*(r_p1**2)))
                                              ) - self.W3*(math.exp(-(self.a*(r_p2**2)))) - self.W3*(math.exp(-(self.a*(r_p3**2)))) - self.W3*(math.exp(-(self.a*(r_p4**2))))

        # if new_pos_rz < new_pos_p1z:
        #     reward = (self.W2*actions) - self.W3*(math.exp(-(self.a*(r_p1**2)))
        #                                           ) - self.W3*(math.exp(-(self.a*(r_p2**2)))) - self.W3*(math.exp(-(self.a*(r_p3**2)))) - self.W3*(math.exp(-(self.a*(r_p4**2))))

        # elif (new_pos_p1z < new_pos_rz) & (new_pos_rz < new_pos_p2z):
        #     reward = (self.W2*actions) - self.W3*(math.exp(-(self.a*(r_p2**2)))) - self.W3 * \
        #         (math.exp(-(self.a*(r_p3**2)))) - \
        #         self.W3*(math.exp(-(self.a*(r_p4**2))))

        # elif (new_pos_p2z < new_pos_rz) & (new_pos_p3z > new_pos_rz):
        #     reward = (self.W2*actions) - self.W3*(math.exp(-(self.a *
        #                                                      (r_p3**2)))) - self.W3*(math.exp(-(self.a*(r_p4**2))))

        # elif (new_pos_p3z < new_pos_rz) & (new_pos_p4z > new_pos_rz):
        #     reward = (self.W2*actions) - self.W3 * \
        #         (math.exp(-(self.a*(r_p4**2))))
        # else:
        #     reward = (self.W2 * actions)

        self.pos_rx = new_pos_rx
        self.pos_ry = new_pos_ry
        self.pos_rz = new_pos_rz
        self.pos_p1x = new_pos_p1x
        self.pos_p1y = new_pos_p1y
        self.pos_p1z = new_pos_p1z
        self.pos_p2x = new_pos_p2x
        self.pos_p2y = new_pos_p2y
        self.pos_p2z = new_pos_p2z
        self.pos_p3x = new_pos_p3x
        self.pos_p3y = new_pos_p3y
        self.pos_p3z = new_pos_p3z
        self.pos_p4x = new_pos_p4x
        self.pos_p4y = new_pos_p4y
        self.pos_p4z = new_pos_p4z

        self.next_data = np.array([self.pos_rx, self.pos_ry, self.pos_rz, self.pos_p1x,
                                  self.pos_p1y, self.pos_p1z, self.pos_p2x, self.pos_p2y, self.pos_p2z, self.pos_p3x, self.pos_p3y, self.pos_p3z, self.pos_p4x, self.pos_p4y, self.pos_p4z])

        self.relative_distance_p1 = relative_distance(
            self.next_data[0], self.next_data[2], self.next_data[3], self.next_data[5])

        self.relative_distance_p2 = relative_distance(
            self.next_data[0], self.next_data[2], self.next_data[6], self.next_data[8])

        self.relative_distance_p3 = relative_distance(
            self.next_data[0], self.next_data[2], self.next_data[9], self.next_data[11])

        self.relative_distance_p4 = relative_distance(
            self.next_data[0], self.next_data[2], self.next_data[12], self.next_data[14])

        self.relative_angle_p1 = relative_angle(
            self.next_data[0], self.next_data[2], self.next_data[3], self.next_data[5])

        self.relative_angle_p2 = relative_angle(
            self.next_data[0], self.next_data[2], self.next_data[6], self.next_data[8])

        self.relative_angle_p3 = relative_angle(
            self.next_data[0], self.next_data[2], self.next_data[9], self.next_data[11])

        self.relative_angle_p4 = relative_angle(
            self.next_data[0], self.next_data[2], self.next_data[12], self.next_data[14])

        self.relative_velocity_p1 = relative_velocity(
            self.vel_p1x, self.vel_rx)
        self.relative_velocity_p2 = relative_velocity(
            self.vel_p2x, self.vel_rx)
        self.relative_velocity_p3 = relative_velocity(
            self.vel_p3x, self.vel_rx)
        self.relative_velocity_p4 = relative_velocity(
            self.vel_p4x, self.vel_rx)

        self.next_state = [self.relative_distance_p1, self.relative_distance_p2,
                           self.relative_distance_p3, self.relative_distance_p4,
                           self.relative_angle_p1, self.relative_angle_p2,
                           self.relative_angle_p3, self.relative_angle_p4,
                           self.relative_velocity_p1, self.relative_velocity_p2, self.relative_velocity_p3, self.relative_velocity_p4]

        return self.next_state, terminal, reward, feeling, self.next_data
