#!/usr/bin/env python
# coding=utf-8
import numpy as np
import scipy.ndimage.interpolation as shift
import numpy.testing as test


class MDP:
    eps = np.finfo(np.float64).eps

    def policy_evaluation(self, action):
        has_wall = bool(self.wall_dict)
        if action < 4:  # walk
            # new_value_up = shift.shift(self.value, (1, 0), mode="nearest")
            new_value_up = np.delete(np.concatenate((self.value[0:1, ...], self.value), axis=0), -1, axis=0)
            if has_wall:
                wall_list = self.wall_dict[self.walk_up]
                new_value_up[wall_list[:, 0], wall_list[:, 1]] = self.value[wall_list[:, 0], wall_list[:, 1]]

            # new_value_right = shift.shift(self.value, (0, -1), mode="nearest")
            new_value_right = np.delete(np.concatenate((self.value, self.value[..., -1:]), axis=1), 0, axis=1)
            if has_wall:
                wall_list = self.wall_dict[self.walk_right]
                new_value_right[wall_list[:, 0], wall_list[:, 1]] = self.value[wall_list[:, 0], wall_list[:, 1]]

            # new_value_down = shift.shift(self.value, (-1, 0), mode="nearest")
            new_value_down = np.delete(np.concatenate((self.value, self.value[-1:, ...]), axis=0), 0, axis=0)
            if has_wall:
                wall_list = self.wall_dict[self.walk_down]
                new_value_down[wall_list[:, 0], wall_list[:, 1]] = self.value[wall_list[:, 0], wall_list[:, 1]]

            # new_value_left = shift.shift(self.value, (0, 1), mode="nearest")
            new_value_left = np.delete(np.concatenate((self.value[..., 0:1], self.value), axis=1), -1, axis=1)
            if has_wall:
                wall_list = self.wall_dict[self.walk_left]
                new_value_left[wall_list[:, 0], wall_list[:, 1]] = self.value[wall_list[:, 0], wall_list[:, 1]]
        else:
            one_step = np.delete(np.concatenate((self.value[0:2:, ...], self.value), axis=0), -1, axis=0)
            new_value_up = np.delete(one_step, -1, axis=0)
            if has_wall:
                wall_list = self.wall_dict[self.run_up]
                new_value_up[wall_list[:, 0], wall_list[:, 1]] = self.value[wall_list[:, 0], wall_list[:, 1]]

            one_step = np.delete(np.concatenate((self.value, self.value[..., -2:]), axis=1), 0, axis=1)
            new_value_right = np.delete(one_step, 0, axis=1)
            if has_wall:
                wall_list = self.wall_dict[self.run_right]
                new_value_right[wall_list[:, 0], wall_list[:, 1]] = self.value[wall_list[:, 0], wall_list[:, 1]]

            one_step = np.delete(np.concatenate((self.value, self.value[-2:, ...]), axis=0), 0, axis=0)
            new_value_down = np.delete(one_step, 0, axis=0)
            if has_wall:
                wall_list = self.wall_dict[self.run_down]
                new_value_down[wall_list[:, 0], wall_list[:, 1]] = self.value[wall_list[:, 0], wall_list[:, 1]]

            # old_value_left = shift.shift(self.value, (0, 2), mode="reflect")
            # old_value_left[:, [0, 1]] = old_value_left[:, [1, 0]]
            one_step = np.delete(np.concatenate((self.value[..., 0:2], self.value), axis=1), -1, axis=1)
            new_value_left = np.delete(one_step, -1, axis=1)
            # test.assert_array_almost_equal(old_value_left, new_value_left)
            if has_wall:
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
        # print(time.time())
        discount = self.e * (1 - self.discount) / self.discount
        # while delta >= discount:
        i = 0
        while True:
            # print(time.time())
            # self.value = u_p  # u should be self.value
            # delta = 0
            # max = np.full((self.length, self.width), -10000)
            # expect = self.policy_evaluation()
            max_value = None
            # print(self.value)
            # print("-----")
            # print(self.policy)
            # print("-----")
            u_p = self.value
            current_policy = np.zeros(self.policy.shape)
            for action in range(8):
                if action < 4:  # other actions
                    action_value = self.policy_evaluation(action) * self.discount + self.reward_walk
                    if max_value is None:
                        max_value = action_value
                        current_policy = np.zeros(self.policy.shape)
                    # current_policy[action_value - max_value > self.e] = action  # if the action has a better value, then the policy should be this action
                    current_policy[action_value > max_value] = action  # if the action has a better value, then the policy should be this action
                    max_value = np.maximum(max_value, action_value)  # walk
                    # if action == self.walk_up:
                    #     diff = action_value - max_value
                    #     print(action)
                    #     print(repr(diff[0, 2]))
                    #     print("---")
                    # u_p = max_value * self.discount + self.reward_walk
                    # max_value = np.maximum(action_value * self.discount + self.reward_walk, max_value)
                    # u_p = np.maximum(max_value, self.policy_evaluation(action)) + self.reward_walk
                else:
                    action_value = self.policy_evaluation(action) * self.discount + self.reward_run
                    # current_policy[action_value - max_value > self.e] = action
                    current_policy[action_value > max_value] = action
                    # self.policy[action_value > max_value] = action
                    max_value = np.maximum(max_value, action_value)  # run
                    # if action == self.run_up:
                    #     diff = action_value - max_value
                    #     print(action)
                    #     print(repr(diff[0, 2]))
                    #     print("---")
            self.fix_exit(max_value)
            min_value = np.amax(np.abs(self.value - max_value))
            # if min_value > delta:
            delta = min_value
            # print(delta)
            self.value = max_value
            # print(max_value[2:6, 53:57])
            self.policy = current_policy
            # print(self.policy)
            # print(max_value)
            # print(current_policy[0, -1])
            i += 1
            if delta < discount:
                print(i)
                return

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

    def __init__(self, length, width, p_walk, p_run, reward_run, reward_walk, discount, exit_list, wall_list=None, e=1e-8):
        if wall_list is None:
            wall_list = []
        self.wall_list = wall_list
        self.exit_list = exit_list
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
        self.direction_enum = (self.up, self.down, self.left, self.right)
        self.walk_up = 0
        self.walk_down = 1
        self.walk_left = 2
        self.walk_right = 3

        self.run_up = 4
        self.run_down = 5
        self.run_left = 6
        self.run_right = 7
        self.action_enum = (self.walk_up, self.walk_down, self.walk_left, self.walk_right, self.run_up, self.run_down, self.run_left, self.run_right)
        self.p = np.zeros((length, width, 4, 8), dtype="float64")  # four directions and 8 actions

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
        self.value = np.zeros((length, width), dtype="float64")
        self.build_wall()
        self.fix_exit(self.value)

    def fix_exit(self, value):
        for exit_entry in self.exit_list:
            location = exit_entry[0]
            utility = exit_entry[1]
            value[location[0], location[1]] = utility

    def out_put(self):
        for exit_entry in self.exit_list:
            location = exit_entry[0]
            self.policy[location[0], location[1]] = -1
        for wall in self.wall_list:
            self.policy[wall[0], wall[1]] = -2
        N, D = self.policy.shape
        str = ''
        for i in range(N):
            for j in range(D):
                if self.policy[i][j] == -1:
                    str += "Exit"
                elif self.policy[i][j] == -2:
                    str += "None"
                elif self.policy[i][j] == 0:
                    str += 'Walk Up'
                elif self.policy[i][j] == 1:
                    str += 'Walk Down'
                elif self.policy[i][j] == 2:
                    str += 'Walk Left'
                elif self.policy[i][j] == 3:
                    str += 'Walk Right'
                elif self.policy[i][j] == 4:
                    str += 'Run Up'
                elif self.policy[i][j] == 5:
                    str += 'Run Down'
                elif self.policy[i][j] == 6:
                    str += 'Run Left'
                elif self.policy[i][j] == 7:
                    str += 'Run Right'
                elif self.policy[i][j] == 8:
                    str += 'None'
                elif self.policy[i][j] == 9:
                    str += 'Exit'
                if j == D - 1:
                    str += "\n"
                    # if i != N_row - 1:
                    #     str += "\n"
                else:
                    str += ","
        return str


class Configuration:
    def read_file(self, path):
        lines = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip('\n')
                lines.append(line)

        grid_size = lines[0]
        wall_cells_num = int(lines[1])
        wall_cells_pos = lines[2:2 + wall_cells_num]
        term_states_num = int(lines[2 + wall_cells_num])
        term_states_pos = lines[3 + wall_cells_num:3 + wall_cells_num + term_states_num]
        tran_model = lines[3 + wall_cells_num + term_states_num]
        rewards = lines[4 + wall_cells_num + term_states_num]
        discount = np.float_(lines[5 + wall_cells_num + term_states_num])
        grid_sz = grid_size.split(",")
        length, width = int(grid_sz[0]), int(grid_sz[1])
        wall_list = []
        for ele in wall_cells_pos:
            ele = ele.split(" ")
            for ele2 in ele:
                ele2 = ele2.split(",")
                pnt = (length - int(ele2[0]), int(ele2[1]) - 1)
                wall_list.append(pnt)
        assert len(wall_list) == wall_cells_num
        term_list = []
        for ele in term_states_pos:
            ele = ele.split(" ")
            for ele2 in ele:
                ele2 = ele2.split(",")
                pnt = (length - int(ele2[0]), int(ele2[1]) - 1)
                utilize = np.float_(ele2[2])
                term_list.append((pnt, utilize))
        assert len(term_list) == term_states_num
        tran_m = tran_model.split(",")
        p_walk, p_run = np.float_(tran_m[0]), np.float_(tran_m[1])
        rw = rewards.split(",")
        r_walk, r_run = np.float_(rw[0]), np.float_(rw[1])
        return width, length, p_walk, p_run, r_walk, r_run, discount, wall_list, term_list


def main():
    # def __init__(self, length, width, p_walk, p_run, reward_run, reward_walk, discount, exit_list, e=10e-5):
    mdp = MDP(length=500, width=600, p_walk=1, p_run=1, reward_run=0, reward_walk=0, discount=0.7, exit_list=[((100, 100), 10)])
    mdp.value_iteration()


if __name__ == "__main__":
    main()
