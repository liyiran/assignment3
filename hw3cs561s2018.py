#!/usr/bin/env python
# coding=utf-8
import numpy as np
import sys
import time


class MDP:
    def policy_evaluation(self, action):
        up_down_walk = np.zeros((1, self.width))
        left_right_walk = np.zeros((self.length, 1))

        up_down_run = np.zeros((2, self.width))
        left_right_run = np.zeros((self.length, 2))
        new_map = None
        sum = None
        if action == self.walk_up:
            new_map = np.delete(np.concatenate((self.value[0:1, ...], self.value), axis=0), -1, axis=0)
            # self.value = np.multiply(self.p[:,:,self.up, action], new_map) + 
        elif action == self.walk_down:
            new_map = np.delete(np.concatenate((self.value, self.value[-1:, ...]), axis=0), 0, axis=0)

        elif action == self.walk_left:
            new_map = np.delete(np.concatenate((self.value[..., 0:1], self.value), axis=1), -1, axis=1)

        elif action == self.walk_right:
            new_map = np.delete(np.concatenate((self.value, self.value[..., -1:]), axis=1), 0, axis=1)

        elif action == self.run_up:
            new_map = np.delete(np.delete(np.concatenate((self.value[0:2:, ...], self.value), axis=0), -1, axis=0), -1, axis=0)
        elif action == self.run_down:
            new_map = np.delete(np.concatenate((self.value, self.value[-2:, ...]), axis=0), 1, axis=0)
            new_map = np.delete(new_map, 1, axis=0)
        elif action == self.run_left:
            new_map = np.delete(np.concatenate((self.value[..., 0:2], self.value), axis=1), -1, axis=1)
            new_map = np.delete(new_map, -1, axis=1)
        elif action == self.run_right:
            new_map = np.delete(np.concatenate((self.value, self.value[..., -2:]), axis=1), -1, axis=1)
            new_map = np.delete(new_map, -1, axis=1)
        return np.multiply(self.p[:, :, self.up, action].flatten().reshape(self.length, self.width), new_map) + \
               np.multiply(self.p[:, :, self.down, action].flatten().reshape(self.length, self.width), new_map) \
               + np.multiply(self.p[:, :, self.left, action].flatten().reshape(self.length, self.width), new_map) + \
               np.multiply(self.p[:, :, self.right, action].flatten().reshape(self.length, self.width), new_map)

    def value_iteration(self):
        u_p = 0
        delta = 0
        print(time.time())
        while delta < self.e * (1 - self.discount) / self.discount:
            print(time.time())
            u = u_p
            delta = 0
            max = np.full((self.length, self.width), -10000)
            for action in range(8):
                if action < 4:
                    u_p = np.maximum(max, self.policy_evaluation(action)) + self.reward_walk
                else:
                    u_p = np.maximum(max, self.policy_evaluation(action)) + self.reward_walk
                if np.min(np.abs(u - u_p)) > delta:
                    delta = np.min(np.abs(u - u_p))

    def __init__(self, length, width, p_walk, p_run, reward_run, reward_walk, discount, exit_list, e=10e-5):
        self.discount = discount
        self.reward_run = reward_run
        self.reward_walk = reward_walk
        self.length = length
        self.width = width
        self.e = e
        self.up = 0,
        self.left = 1,
        self.down = 2
        self.right = 3

        self.walk_up = 0
        self.walk_left = 1
        self.walk_down = 2
        self.walk_right = 3

        self.run_up = 4
        self.run_left = 5
        self.run_down = 6
        self.run_right = 7

        self.p = np.zeros((length, width, 4, 8))  # four directions and 8 actions

        self.p[:, :, self.up, self.walk_up] = p_walk
        self.p[:, :, self.left, self.walk_up] = .5 * (1 - p_walk)
        self.p[:, :, self.right, self.walk_up] = .5 * (1 - p_walk)

        self.p[:, :, self.right, self.walk_right] = p_walk
        self.p[:, :, self.up, self.walk_left] = .5 * (1 - p_walk)
        self.p[:, :, self.down, self.walk_left] = .5 * (1 - p_walk)

        self.p[:, :, self.down, self.walk_down] = p_walk
        self.p[:, :, self.left, self.walk_down] = .5 * (1 - p_walk)
        self.p[:, :, self.right, self.walk_down] = .5 * (1 - p_walk)

        self.p[:, :, self.left, self.walk_left] = p_walk
        self.p[:, :, self.up, self.walk_down] = .5 * (1 - p_walk)
        self.p[:, :, self.down, self.walk_down] = .5 * (1 - p_walk)

        self.p[:, :, self.up, self.run_up] = p_run
        self.p[:, :, self.left, self.run_up] = .5 * (1 - p_run)
        self.p[:, :, self.right, self.run_up] = .5 * (1 - p_run)

        self.p[:, :, self.right, self.run_right] = p_run
        self.p[:, :, self.up, self.run_right] = .5 * (1 - p_run)
        self.p[:, :, self.down, self.run_right] = .5 * (1 - p_run)

        self.p[:, :, self.down, self.run_down] = p_run
        self.p[:, :, self.left, self.run_down] = .5 * (1 - p_run)
        self.p[:, :, self.right, self.run_down] = .5 * (1 - p_run)

        self.p[:, :, self.left, self.run_left] = p_run
        self.p[:, :, self.up, self.run_left] = .5 * (1 - p_run)
        self.p[:, :, self.down, self.run_left] = .5 * (1 - p_run)

        self.policy = np.zeros((length, width))
        self.value = np.zeros((length, width))
        for exit in exit_list:
            location = exit[0]
            utility = exit[1]
            self.value[location[0], location[1]] = utility


def main():
    # def __init__(self, length, width, p_walk, p_run, reward_run, reward_walk, discount, exit_list, e=10e-5):
    mdp = MDP(length=500, width=600, p_walk=1, p_run=1, reward_run=0, reward_walk=0, discount=0.7, exit_list=[((100, 100), 10)])
    mdp.value_iteration()


if __name__ == "__main__":
    main()
