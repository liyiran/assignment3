#!/usr/bin/env python
# coding=utf-8
import numpy as np
import sys
import time
import scipy.ndimage.interpolation as shift


class MDP:
    # def fix_wall_neighbor(self, new_value):
    #     wall_list = self.wall_list
    #     for wall in wall_list:
    #         new_value[wall[0], wall[1]] = 0  # the wall should have 0 value
    #         # fix 4 walk cells
    #         left = (wall[0], wall[1] - 1)
    #         right = (wall[0], wall[1] + 1)
    #         up = (wall[0] - 1, wall[1])
    #         down = (wall[0] + 1, wall[1])
    #         # fix 4 run cells
    #         left_left = (wall[0], wall[1] - 2)
    #         right_right = (wall[0], wall[1] + 2)
    #         up_up = (wall[0] - 2, wall[1])
    #         down_down = (wall[0] + 2, wall[1])

    def policy_evaluation(self, action):
        if action < 4:  # walk
            new_value_up = shift.shift(self.value, (1, 0), mode="nearest")
            wall_list = self.wall_dict[self.walk_up]
            new_value_up[wall_list[:, 0], wall_list[:, 1]] = self.value[wall_list[:, 0], wall_list[:, 1]]

            new_value_right = shift.shift(self.value, (0, -1), mode="nearest")
            wall_list = self.wall_dict[self.walk_right]
            new_value_right[wall_list[:, 0], wall_list[:, 1]] = self.value[wall_list[:, 0], wall_list[:, 1]]

            new_value_down = shift.shift(self.value, (-1, 0), mode="nearest")
            wall_list = self.wall_dict[self.walk_down]
            new_value_down[wall_list[:, 0], wall_list[:, 1]] = self.value[wall_list[:, 0], wall_list[:, 1]]

            new_value_left = shift.shift(self.value, (0, 1), mode="nearest")
            wall_list = self.wall_dict[self.walk_left]
            new_value_left[wall_list[:, 0], wall_list[:, 1]] = self.value[wall_list[:, 0], wall_list[:, 1]]
        else:
            new_value_up = shift.shift(self.value, (2, 0), mode="reflect")
            new_value_up[[0, 1]] = new_value_up[[1, 0]]  # swap the first two rows
            wall_list = self.wall_dict[self.run_up]
            new_value_up[wall_list[:, 0], wall_list[:, 1]] = self.value[wall_list[:, 0], wall_list[:, 1]]

            new_value_right = shift.shift(self.value, (0, -2), mode="reflect")
            new_value_right[:, [-1, -2]] = new_value_right[:, [-2, -1]]
            wall_list = self.wall_dict[self.run_right]
            new_value_right[wall_list[:, 0], wall_list[:, 1]] = self.value[wall_list[:, 0], wall_list[:, 1]]

            new_value_down = shift.shift(self.value, (-2, 0), mode="reflect")
            new_value_down[[-1, -2]] = new_value_down[[-2, -1]]
            wall_list = self.wall_dict[self.run_down]
            new_value_down[wall_list[:, 0], wall_list[:, 1]] = self.value[wall_list[:, 0], wall_list[:, 1]]

            new_value_left = shift.shift(self.value, (0, 2), mode="reflect")
            new_value_left[:, [0, 1]] = new_value_left[:, [1, 0]]
            wall_list = self.wall_dict[self.run_left]
            new_value_left[wall_list[:, 0], wall_list[:, 1]] = self.value[wall_list[:, 0], wall_list[:, 1]]

        new_value = np.multiply(np.squeeze(self.p[:, :, self.up, action]), new_value_up) + \
                    np.multiply(np.squeeze(self.p[:, :, self.down, action]), new_value_down) + \
                    np.multiply(np.squeeze(self.p[:, :, self.left, action]), new_value_left) + \
                    np.multiply(np.squeeze(self.p[:, :, self.right, action]), new_value_right)
        return new_value

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

    def build_wall(self):
        for wall in self.wall_list:
            # self.wall_walk_up.append((wall[0], wall[1]))
            if self.is_inside(wall[0] + 1, wall[1]):
                self.wall_up_1.append((wall[0] + 1, wall[1]))

            # self.wall_run_up.append((wall[0], wall[1]))
            if self.is_inside(wall[0] + 1, wall[1]):
                self.wall_up_2.append((wall[0] + 1, wall[1]))
            if self.is_inside(wall[0] + 2, wall[1]):
                self.wall_up_2.append((wall[0] + 2, wall[1]))

            # self.wall_walk_left.append((wall[0], wall[1]))
            if self.is_inside(wall[0], wall[1] + 1):
                self.wall_left_1.append((wall[0], wall[1] + 1))

            # self.wall_run_left.append((wall[0], wall[1]))
            if self.is_inside(wall[0], wall[1] + 1):
                self.wall_left_2.append((wall[0], wall[1] + 1))
            if self.is_inside(wall[0], wall[1] + 2):
                self.wall_left_2.append((wall[0], wall[1] + 2))

            # self.wall_walk_down.append((wall[0], wall[1]))
            if self.is_inside(wall[0] - 1, wall[1]):
                self.wall_down_1.append((wall[0] - 1, wall[1]))
            # self.wall_run_down.append((wall[0], wall[1]))

            if self.is_inside(wall[0] - 1, wall[1]):
                self.wall_down_2.append((wall[0] - 1, wall[1]))
            if self.is_inside(wall[0] - 2, wall[1]):
                self.wall_down_2.append((wall[0] - 2, wall[1]))

            # self.wall_walk_right.append((wall[0], wall[1]))
            if self.is_inside(wall[0], wall[1] - 1):
                self.wall_right_1.append((wall[0], wall[1] - 1))

            # self.wall_run_right.append((wall[0], wall[1]))
            if self.is_inside(wall[0], wall[1] - 1):
                self.wall_right_2.append((wall[0], wall[1] - 1))
            if self.is_inside(wall[0], wall[1] - 2):
                self.wall_right_2.append((wall[0], wall[1] - 2))

            self.wall_dict[self.walk_up] = np.array(self.wall_up_1)
            self.wall_dict[self.walk_left] = np.array(self.wall_left_1)
            self.wall_dict[self.walk_down] = np.array(self.wall_down_1)
            self.wall_dict[self.walk_right] = np.array(self.wall_right_1)
            self.wall_dict[self.run_up] = np.array(self.wall_up_2)
            self.wall_dict[self.run_left] = np.array(self.wall_left_2)
            self.wall_dict[self.run_down] = np.array(self.wall_down_2)
            self.wall_dict[self.run_right] = np.array(self.wall_right_2)

    def is_inside(self, x, y):
        return 0 <= x < self.length and 0 <= y < self.width

    def __init__(self, length, width, p_walk, p_run, reward_run, reward_walk, discount, exit_list, wall_list=None, e=10e-5):
        if wall_list is None:
            wall_list = []
        self.wall_list = wall_list
        self.wall_up_1 = []
        self.wall_left_1 = []
        self.wall_down_1 = []
        self.wall_right_1 = []

        self.wall_up_2 = []
        self.wall_left_2 = []
        self.wall_down_2 = []
        self.wall_right_2 = []
        self.wall_dict = {}
        self.discount = discount
        self.reward_run = reward_run
        self.reward_walk = reward_walk
        self.length = length
        self.width = width
        self.wall_list = wall_list
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
        self.build_wall()
        for exit_entry in exit_list:
            location = exit_entry[0]
            utility = exit_entry[1]
            self.value[location[0], location[1]] = utility


def main():
    # def __init__(self, length, width, p_walk, p_run, reward_run, reward_walk, discount, exit_list, e=10e-5):
    mdp = MDP(length=500, width=600, p_walk=1, p_run=1, reward_run=0, reward_walk=0, discount=0.7, exit_list=[((100, 100), 10)])
    mdp.value_iteration()


if __name__ == "__main__":
    main()
