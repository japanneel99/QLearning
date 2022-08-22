from doctest import DocTestSuite
import io
import socket
import time
import json
import numpy as np
import math
import matplotlib as plt
import csv
import random
from math_functions import relative_distance
from math_functions import relative_angle
from math_functions import relative_velocity


class UnityEnvironment:
    def __init__(self, dt, W1, W2, W3, a):
        host = "127.0.0.1"
        port = 25001
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.dt = dt
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.a = a

    def states(self):
        return dict(type='float', shape=(12,))

    def actions(self):
        return dict(type='float', min_value=0.0, max_value=1.6)

    def get_state_from_unity(self):
        self.sock.sendall("GET_WHEELCHAIR_POSITION".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        wheelchair_position_dict = json.loads(receivedData)
        wheelchair_position = [wheelchair_position_dict['x'],
                               wheelchair_position_dict['y'], wheelchair_position_dict['z']]

        self.sock.sendall("GET_WHEELCHAIR_VELOCITY".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        wheelchair_vel_dict = json.loads(receivedData)
        wheelchair_velocity = [wheelchair_vel_dict['x'],
                               wheelchair_vel_dict['y'], wheelchair_vel_dict['z']]

        self.sock.sendall("GET_PEDESTRIAN1_POSITION".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        pedestrian1_pos_dict = json.loads(receivedData)
        pedestrian1_position = [pedestrian1_pos_dict['x'],
                                pedestrian1_pos_dict['y'], pedestrian1_pos_dict['z']]

        self.sock.sendall("GET_PEDESTRIAN1_VELOCITY".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        pedestrian1_vel_dict = json.loads(receivedData)
        pedestrian1_velocity = [pedestrian1_vel_dict['x'],
                                pedestrian1_vel_dict['y'], pedestrian1_vel_dict['z']]

        self.sock.sendall("GET_PEDESTRIAN2_POSITION".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        pedestrian2_pos_dict = json.loads(receivedData)
        pedestrian2_position = [pedestrian2_pos_dict['x'],
                                pedestrian2_pos_dict['y'], pedestrian2_pos_dict['z']]

        self.sock.sendall("GET_PEDESTRIAN2_VELOCITY".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        pedestrian2_vel_dict = json.loads(receivedData)
        pedestrian2_velocity = [pedestrian2_vel_dict['x'],
                                pedestrian2_vel_dict['y'], pedestrian2_vel_dict['z']]

        self.sock.sendall("GET_PEDESTRIAN3_POSITION".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        pedestrian3_pos_dict = json.loads(receivedData)
        pedestrian3_position = [pedestrian3_pos_dict['x'],
                                pedestrian3_pos_dict['y'], pedestrian3_pos_dict['z']]

        self.sock.sendall("GET_PEDESTRIAN3_VELOCITY".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        pedestrian3_vel_dict = json.loads(receivedData)
        pedestrian3_velocity = [pedestrian3_vel_dict['x'],
                                pedestrian3_vel_dict['y'], pedestrian3_vel_dict['z']]

        self.sock.sendall("GET_PEDESTRIAN4_POSITION".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        pedestrian4_pos_dict = json.loads(receivedData)
        pedestrian4_position = [pedestrian4_pos_dict['x'],
                                pedestrian4_pos_dict['y'], pedestrian4_pos_dict['z']]

        self.sock.sendall("GET_PEDESTRIAN4_VELOCITY".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        pedestrian4_vel_dict = json.loads(receivedData)
        pedestrian4_velocity = [pedestrian4_vel_dict['x'],
                                pedestrian4_vel_dict['y'], pedestrian4_vel_dict['z']]

        self.sock.sendall("GET_FEELING_REWARD".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")
        feeling = json.loads(receivedData)['info']
        print(receivedData)
        print("feeling received")

        return wheelchair_position, pedestrian1_position, pedestrian2_position, pedestrian3_position, pedestrian4_position, wheelchair_velocity, pedestrian1_velocity, pedestrian2_velocity, pedestrian3_velocity, pedestrian4_velocity, feeling

    def reset(self):
        print("resetting")
        self.sock.sendall("RESET".encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")

        wheelchair_position, pedestrian1_position, pedestrian2_position, pedestrian3_position, pedestrian4_position, wheelchair_velocity, pedestrian1_velocity, pedestrian2_velocity, pedestrian3_velocity, pedestrian4_velocity, feeling = self.get_state_from_unity()

        self.data = np.array([wheelchair_position[0], wheelchair_position[1], wheelchair_position[2],
                              wheelchair_velocity[0], wheelchair_velocity[1], wheelchair_velocity[2],
                              pedestrian1_position[0], pedestrian1_position[1], pedestrian1_position[2],
                              pedestrian1_velocity[0], pedestrian1_velocity[1], pedestrian1_velocity[2],
                              pedestrian2_position[0], pedestrian2_position[1], pedestrian1_position[2],
                              pedestrian2_velocity[0], pedestrian2_velocity[1], pedestrian2_velocity[2],
                              pedestrian3_position[0], pedestrian3_position[1], pedestrian3_position[2],
                              pedestrian3_velocity[0], pedestrian3_velocity[1], pedestrian3_velocity[2],
                              pedestrian4_position[0], pedestrian4_position[1], pedestrian4_position[2],
                              pedestrian4_velocity[0], pedestrian4_velocity[1], pedestrian4_velocity[2]])

        self.relative_distance_p1 = relative_distance(
            self.data[0], self.data[2], self.data[6], self.data[8])

        self.relative_distance_p2 = relative_distance(
            self.data[0], self.data[2], self.data[12], self.data[14])

        self.relative_distance_p3 = relative_distance(
            self.data[0], self.data[2], self.data[18], self.data[20])

        self.relative_distance_p4 = relative_distance(
            self.data[0], self.data[2], self.data[24], self.data[26])

        self.relative_angle_p1 = relative_angle(
            self.data[0], self.data[2], self.data[6], self.data[8])

        self.relative_angle_p2 = relative_angle(
            self.data[0], self.data[2], self.data[12], self.data[14])

        self.relative_angle_p3 = relative_angle(
            self.data[0], self.data[2], self.data[18], self.data[20])

        self.relative_angle_p4 = relative_angle(
            self.data[0], self.data[2], self.data[24], self.data[26])

        self.relative_velocity_p1 = relative_velocity(
            self.data[3], self.data[9])

        self.relative_velocity_p2 = relative_velocity(
            self.data[3], self.data[15])

        self.relative_velocity_p3 = relative_velocity(
            self.data[3], self.data[21])

        self.relative_velocity_p4 = relative_velocity(
            self.data[3], self.data[27])

        self.state = [self.relative_distance_p1, self.relative_distance_p2,
                      self.relative_distance_p3, self.relative_distance_p4,
                      self.relative_angle_p1, self.relative_angle_p2,
                      self.relative_angle_p3, self.relative_angle_p4,
                      self.relative_velocity_p1, self.relative_velocity_p2, self.relative_velocity_p3, self.relative_velocity_p4]

        print("Ended Reset")
        return self.state

    def execute(self, actions):
        print("Start Executing")

        wheelchair_position, pedestrian1_position, pedestrian2_position, pedestrian3_position, pedestrian4_position, wheelchair_velocity, pedestrian1_velocity, pedestrian2_velocity, pedestrian3_velocity, pedestrian4_velocity, feeling = self.get_state_from_unity()

        terminal = wheelchair_position[2] > 13

        rp1 = relative_distance(
            self.data[0], self.data[2], self.data[6], self.data[8])

        rp2 = relative_distance(
            self.data[0], self.data[2], self.data[12], self.data[14])

        rp3 = relative_distance(
            self.data[0], self.data[2], self.data[18], self.data[20])

        rp4 = relative_distance(
            self.data[0], self.data[2], self.data[24], self.data[26])

        reward = (self.W1 * feeling) + (self.W2*actions) - self.W3*(math.exp(-(self.a*(rp1**2)))
                                                                    ) - self.W3*(math.exp(-(self.a*(rp2**2)))) - self.W3*(math.exp(-(self.a*(rp3**2)))) - self.W3*(math.exp(-(self.a*(rp4**2))))

        msg = "SEND_ACTION,%f" % actions
        self.sock.sendall(msg.encode("UTF-8"))
        receivedData = self.sock.recv(1024).decode("UTF-8")

        self.data = np.array([wheelchair_position[0], wheelchair_position[1], wheelchair_position[2],
                              wheelchair_velocity[0], wheelchair_velocity[1], wheelchair_velocity[2],
                              pedestrian1_position[0], pedestrian1_position[1], pedestrian1_position[2],
                              pedestrian1_velocity[0], pedestrian1_velocity[1], pedestrian1_velocity[2],
                              pedestrian2_position[0], pedestrian2_position[1], pedestrian1_position[2],
                              pedestrian2_velocity[0], pedestrian2_velocity[1], pedestrian2_velocity[2],
                              pedestrian3_position[0], pedestrian3_position[1], pedestrian3_position[2],
                              pedestrian3_velocity[0], pedestrian3_velocity[1], pedestrian3_velocity[2],
                              pedestrian4_position[0], pedestrian4_position[1], pedestrian4_position[2],
                              pedestrian4_velocity[0], pedestrian4_velocity[1], pedestrian4_velocity[2]])

        self.relative_distance_p1 = relative_distance(
            self.data[0], self.data[2], self.data[6], self.data[8])

        self.relative_distance_p2 = relative_distance(
            self.data[0], self.data[2], self.data[12], self.data[14])

        self.relative_distance_p3 = relative_distance(
            self.data[0], self.data[2], self.data[18], self.data[20])

        self.relative_distance_p4 = relative_distance(
            self.data[0], self.data[2], self.data[24], self.data[26])

        self.relative_angle_p1 = relative_angle(
            self.data[0], self.data[2], self.data[6], self.data[8])

        self.relative_angle_p2 = relative_angle(
            self.data[0], self.data[2], self.data[12], self.data[14])

        self.relative_angle_p3 = relative_angle(
            self.data[0], self.data[2], self.data[18], self.data[20])

        self.relative_angle_p4 = relative_angle(
            self.data[0], self.data[2], self.data[24], self.data[26])

        self.relative_velocity_p1 = relative_velocity(
            self.data[3], self.data[9])

        self.relative_velocity_p2 = relative_velocity(
            self.data[3], self.data[15])

        self.relative_velocity_p3 = relative_velocity(
            self.data[3], self.data[21])

        self.relative_velocity_p4 = relative_velocity(
            self.data[3], self.data[27])

        self.next_data = np.array([self.data[0], self.data[1], self.data[2],
                                   self.data[6], self.data[7], self.data[8],
                                   self.data[12], self.data[13], self.data[14],
                                   self.data[18], self.data[19], self.data[20],
                                   self.data[24], self.data[25], self.data[26]])

        self.next_state = [self.relative_distance_p1, self.relative_distance_p2,
                           self.relative_distance_p3, self.relative_distance_p4,
                           self.relative_angle_p1, self.relative_angle_p2,
                           self.relative_angle_p3, self.relative_angle_p4,
                           self.relative_velocity_p1, self.relative_velocity_p2,
                           self.relative_velocity_p3, self.relative_velocity_p4]

        return self.next_state, terminal, reward, feeling, self.next_data
