from copy import deepcopy
import importlib
from typing import List, Tuple, Any, Union
import numpy as np
import warnings
import math
import pickle


def quantization(x: float, x_min: float, x_max: float, n_quad: int) -> int:
    """
    Quantization function
    """
    index = int((x - x_min) / (x_max - x_min) * n_quad)
    if index < 0:
        warnings.warn("index < 0")
        return 0
    elif index >= n_quad:
        warnings.warn("index >= n_quad")
        return n_quad - 1
    return index


class QtableAgent:
    def __init__(
        self,
        action_candidates: List[Any],
        quantization: List[Tuple[float, float, int]],
        epsilon: float,
        alpha: float,
        gamma: float
    ):

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.states = None
        self.action_candidates = np.array(action_candidates)
        self.n_action = len(action_candidates)
        self.quantization = quantization
        dimension = [quantization[i][2] for i in range(len(quantization))]
        self.q_table = np.zeros(dimension + [self.n_action])

    def act(self, states, custom_epsilon=None):
        self.states = states
        eps = self.epsilon if custom_epsilon is None else custom_epsilon

        # epsilon greedy algorithm
        if np.random.random_sample() < eps:
            self.last_action_index = np.random.randint(self.n_action)
        else:
            self.last_action_index = self.__select_best_action(states)
        return self.action_candidates[self.last_action_index]

    def observe(self, reward, terminal):
        best_action_index = self.__select_best_action(self.states)

        td_target = reward + self.gamma * \
            (self.__get_q_value(self.states, best_action_index))

        td_delta = td_target - \
            self.__get_q_value(self.states, self.last_action_index)

        q_value, discrete_states = self.__set_q_value(self.states, self.last_action_index, self.__get_q_value(
            self.states, self.last_action_index) + (self.alpha * td_delta))

        return q_value, discrete_states

    def __select_best_action(self, state):
        return np.argmax([self.__get_q_value(state, i) for i in range(self.n_action)])

    def __get_q_value(self, state, action_index):
        index = [quantization(state[i],
                              self.quantization[i][0],
                              self.quantization[i][1],
                              self.quantization[i][2])
                 for i in range(len(self.quantization))]
        self.q_table.__getitem__(tuple(index + [action_index]))

    def __set_q_value(self, state, action_index, val):
        index = [
            quantization(
                state[i],
                self.quantization[i][0],
                self.quantization[i][1],
                self.quantization[i][2])
            for i in range(len(self.quantization))]
        self.q_table.__setitem__(tuple(index + [action_index]), val)
        return self.q_table.__getitem__(tuple(index + [action_index])), index

    def save(self, directory='saved_variables'):
        np.save(directory + "/q_table.npy", self.q_table)
        args = (self.action_candidates, self.quantization,
                self.epsilon, self.alpha, self.gamma)
        with open(directory+"/args.pickles", "wb") as f:
            pickle.dump(args, f)

    @staticmethod
    def load(self, directory='saved_variables'):
        with open(directory+"/args.pickles", "rb") as f:
            args = pickle.load(f)
        action_candidates, quantization, epsilon, alpha, gamma = args
        agent = QtableAgent(action_candidates, quantization,
                            epsilon, alpha, gamma)
        agent.q_table = np.load(directory + "/q_table.npy")
        return agent
