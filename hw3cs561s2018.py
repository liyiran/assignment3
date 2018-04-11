#!/usr/bin/env python
# coding=utf-8
import numpy as np
import sys
import time
import scipy.ndimage.interpolation as shift


class MDP:
    def policy_evaluation(self, action):
        # up_down_walk = np.zeros((1, self.width))
        # left_right_walk = np.zeros((self.length, 1))
        # 
        # up_down_run = np.zeros((2, self.width))
        # left_right_run = np.zeros((self.length, 2))
        
        # new_value_up = None
        # new_value_left = None
        # new_value_down = None
        # new_value_right = None
        if action < 4:  # walk
            new_value_up = shift.shift(self.value, (1, 0), mode="nearest")
            new_value_right = shift.shift(self.value, (0, 1), mode="nearest")
            new_value_down = shift.shift(self.value, (-1, 0), mode="nearest")
            new_value_left = shift.shift(self.value, (0, -1), mode="nearest")
        else:
            new_value_up = shift.shift(self.value, (2, 0), mode="nearest")
            new_value_right = shift.shift(self.value, (0, 2), mode="nearest")
            new_value_down = shift.shift(self.value, (-2, 0), mode="nearest")
            new_value_left = shift.shift(self.value, (0, -2), mode="nearest")
        return np.multiply(np.squeeze(self.p[:, :, self.up, action]), new_value_up) + \
               np.multiply(np.squeeze(self.p[:, :, self.down, action]), new_value_down) \
               + np.multiply(np.squeeze(self.p[:, :, self.left, action]), new_value_left) + \
               np.multiply(np.squeeze(self.p[:, :, self.right, action]), new_value_right)
        # if action == self.walk_up:
        #     new_map = shift.shift(self.value)
        #     # new_map = np.delete(np.concatenate((self.value[0:1, ...], self.value), axis=0), -1, axis=0)
        #     # self.value = np.multiply(self.p[:,:,self.up, action], new_map) + 
        # elif action == self.walk_down:
        #     # new_map = np.delete(np.concatenate((self.value, self.value[-1:, ...]), axis=0), 0, axis=0)
        #     pass
        # elif action == self.walk_left:
        #     pass
        # # new_map = np.delete(np.concatenate((self.value[..., 0:1], self.value), axis=1), -1, axis=1)
        # 
        # elif action == self.walk_right:
        #     pass
        # # new_map = np.delete(np.concatenate((self.value, self.value[..., -1:]), axis=1), 0, axis=1)
        # 
        # elif action == self.run_up:
        #     pass
        # # new_map = np.delete(np.delete(np.concatenate((self.value[0:2:, ...], self.value), axis=0), -1, axis=0), -1, axis=0)
        # elif action == self.run_down:
        #     pass
        # # new_map = np.delete(np.concatenate((self.value, self.value[-2:, ...]), axis=0), 1, axis=0)
        # # new_map = np.delete(new_map, 1, axis=0)
        # elif action == self.run_left:
        #     pass
        # # new_map = np.delete(np.concatenate((self.value[..., 0:2], self.value), axis=1), -1, axis=1)
        # # new_map = np.delete(new_map, -1, axis=1)
        # elif action == self.run_right:
        #     pass
        # new_map = np.delete(np.concatenate((self.value, self.value[..., -2:]), axis=1), -1, axis=1)
        # new_map = np.delete(new_map, -1, axis=1)

        # big_value = self.value_expend()
        # return np.einsum("jkxy,jkxy->jky", big_value, self.p)

    # def value_expend(self):
    #     big_value = self.value[:, :, np.newaxis, np.newaxis]
    #     big_value = np.repeat(big_value, 4, 2)
    #     big_value = np.repeat(big_value, 8, 3)
    #     # for direction in self.direction_enum:
    #     #     for action in self.action_enum:
    #     #         if action < 4:  # walk
    #     #             # swift one row/ column
    #     #             if direction == self.up:
    #     #                 pass
    #     #                 # big_value = np.delete(np.concatenate((big_value[0:1, :, direction, action], big_value), axis=0), -1, axis=0)
    #     #             elif direction == self.left:
    #     #                 pass
    #     #             elif direction == self.down:
    #     #                 pass
    #     #             elif direction == self.right:
    #     #                 pass
    #     #     else:  # run
    #     #         # switf two row/column
    #     #         if direction == self.up:
    #     #             pass
    #     #         elif direction == self.left:
    #     #             pass
    #     #         elif direction == self.down:
    #     #             pass
    #     #         elif direction == self.right:
    #     #             pass
    #     # if action == self.walk_up:
    #     #     big_value = np.delete(np.concatenate((big_value[0:1, ...], big_value), axis=0), -1, axis=0)
    #     # elif action == self.walk_down:
    #     #     big_value = np.delete(np.concatenate((big_value, big_value[-1:, ...]), axis=0), 0, axis=0)
    #     # elif action == self.walk_left:
    #     #     big_value = np.delete(np.concatenate((big_value[:, 0:1, :, :], big_value), axis=1), -1, axis=1)
    #     # elif action == self.walk_right:
    #     #     big_value = np.delete(np.concatenate((big_value, big_value[:, -1:, ...]), axis=1), 0, axis=1)
    #     # elif action == self.run_up:
    #     #     big_value = np.delete(np.delete(np.concatenate((big_value[0:2:, ...], big_value), axis=0), -1, axis=0), -1, axis=0)
    #     # elif action == self.run_down:
    #     #     big_value = np.delete(np.concatenate((big_value, big_value[-2:, ...]), axis=0), 1, axis=0)
    #     #     big_value = np.delete(big_value, 1, axis=0)
    #     # elif action == self.run_left:
    #     #     big_value = np.delete(np.concatenate((big_value[:, 0:2, ...], big_value), axis=1), -1, axis=1)
    #     #     big_value = np.delete(big_value, -1, axis=1)
    #     # elif action == self.run_right:
    #     #     big_value = np.delete(np.concatenate((big_value, big_value[:, -2:, ...]), axis=1), -1, axis=1)
    #     #     big_value = np.delete(big_value, -1, axis=1)
    # 
    #     return big_value

    def value_iteration(self):
        u_p = 0
        delta = 0
        print(time.time())
        while delta < self.e * (1 - self.discount) / self.discount:
            print(time.time())
            u = u_p
            delta = 0
            max = np.full((self.length, self.width), -10000)
            expect = self.policy_evaluation()
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
        self.up = 0
        self.left = 1
        self.down = 2
        self.right = 3
        self.direction_enum = (self.up, self.left, self.right, self.down)
        self.walk_up = 0
        self.walk_left = 1
        self.walk_down = 2
        self.walk_right = 3

        self.run_up = 4
        self.run_left = 5
        self.run_down = 6
        self.run_right = 7
        self.action_enum = (self.walk_up, self.walk_left, self.walk_right, self.walk_down, self.run_up, self.run_left, self.run_right, self.run_down)
        self.p = np.zeros((length, width, 4, 8))  # four directions and 8 actions

        self.p[:, :, self.up, self.walk_up] = p_walk
        self.p[:, :, self.left, self.walk_up] = .5 * (1 - p_walk)
        self.p[:, :, self.right, self.walk_up] = .5 * (1 - p_walk)

        self.p[:, :, self.right, self.walk_right] = p_walk
        self.p[:, :, self.up, self.walk_right] = .5 * (1 - p_walk)
        self.p[:, :, self.down, self.walk_right] = .5 * (1 - p_walk)

        self.p[:, :, self.down, self.walk_down] = p_walk
        self.p[:, :, self.left, self.walk_down] = .5 * (1 - p_walk)
        self.p[:, :, self.right, self.walk_down] = .5 * (1 - p_walk)

        self.p[:, :, self.left, self.walk_left] = p_walk
        self.p[:, :, self.up, self.walk_left] = .5 * (1 - p_walk)
        self.p[:, :, self.down, self.walk_left] = .5 * (1 - p_walk)

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
