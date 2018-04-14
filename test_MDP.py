from unittest import TestCase
from hw3cs561s2018 import MDP
import numpy as np
import numpy.testing as test
import scipy.ndimage.interpolation as shift
import unittest
from hw3cs561s2018 import Configuration


class TestMDP(TestCase):

    def test_eistain(self):
        m1 = np.array([[1, 2, 3]])
        m2 = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
        big = np.concatenate((m1, m2), axis=0)
        # print(big)

        print(np.delete(big, (-1, -2), axis=0))
        # first_two = m2[-1:,...]
        # print(first_two)
        # print(np.concatenate((first_two, m2), axis=0))

    def test_max(self):
        m1 = np.array([1.5, 2, 3])
        m2 = np.array([0, 2.5, 2.9])
        max = np.maximum(m1, m2)
        print(m1 == max)

    def test_expand_value(self):
        value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        value = value[:, :, np.newaxis, np.newaxis]
        value = np.repeat(value, 4, 2)
        value = np.repeat(value, 8, 3)
        print(value.shape)

        p = np.arange(3 * 3 * 4 * 8).reshape((3, 3, 4, 8))
        result = np.multiply(value, p)
        # print(p[:,:,0,0])
        print(result[:, :, 0, 0])

    @unittest.skip("direction is not working")
    def test_initial(self):
        mdp = MDP(width=3, length=3, p_walk=0.7, p_run=0.6, reward_walk=1, reward_run=1, discount=0, exit_list=[((0, 2), 1), ((1, 2), -1)])
        test.assert_array_equal(np.array([[0, 0, 1], [0, 0, -1], [0, 0, 0]]), mdp.value)

        p_up_walk_up = np.squeeze(mdp.p[:, :, mdp.up, mdp.walk_up])
        test.assert_array_equal(np.ones((3, 3)) * 0.7, p_up_walk_up)

        p_down_walk_up = np.squeeze(mdp.p[:, :, mdp.down, mdp.walk_up])
        test.assert_array_equal(np.zeros((3, 3)), p_down_walk_up)

        p_left_walk_up = np.squeeze(mdp.p[:, :, mdp.left, mdp.walk_up])
        test.assert_array_almost_equal(np.ones((3, 3)) * 0.15, p_left_walk_up)

        p_right_walk_up = np.squeeze(mdp.p[:, :, mdp.right, mdp.walk_up])
        test.assert_array_almost_equal(np.ones((3, 3)) * 0.15, p_right_walk_up)

        for direction in mdp.direction_enum:
            for action in mdp.action_enum:
                p = np.squeeze(mdp.p[:, :, direction, action])
                if direction == action:  # walk_direction = direction
                    test.assert_array_almost_equal(p, np.ones((3, 3)) * 0.7)
                elif direction == action - 4:  # run_direction = direction
                    test.assert_array_almost_equal(p, np.ones((3, 3)) * 0.6)
                # elif np.abs(direction - action) == 2 or np.abs(direction - action) == 6:  # opposite direction
                #     test.assert_array_almost_equal(p, np.zeros((3, 3)))
                # elif action < 4:  # walk
                #     test.assert_array_almost_equal(np.ones((3, 3)) * 0.15, p)
                # else:  # run
                #     test.assert_array_almost_equal(np.ones((3, 3)) * 0.2, p)

    # def test_shift_matrix(self):
    #     mdp = MDP(width=3, length=3, p_walk=0.7, p_run=0.6, reward_walk=1, reward_run=1, discount=0, exit_list=[((0, 2), 1), ((1, 2), -1)])
    #     big_value = mdp.value_expend()
    #     print(big_value[:, :, mdp.up, mdp.walk_up])
    #     big_value = np.delete(np.concatenate((big_value[0:1, ...], big_value), axis=0), -1, axis=0)
    #     print(big_value[:, :, mdp.up, mdp.walk_up])
    #     for direction in mdp.direction_enum:
    #         test.assert_array_almost_equal(np.array([[0, 0, 1], [0, 0, 1], [0, 0, -1]]), big_value[:, :, direction, mdp.walk_up])

    def test_index(self):
        xs = np.array([[5., 1., 2., 3., 4., 5., 6., 7., 8., 9.], [5., -2., 2., 3., 4., 5., 6., 7., 8., 9.]])
        idx = np.array([(0, 1), (0, 2)])
        xs[idx[:, 0], idx[:, 1]] = 100
        print(xs)
        # print(shift.shift(xs, (0, 1), mode="nearest"))
        # try:
        #     print(xs[100])
        # except:
        #     pass
        # self.assertTrue(True)

    def test_shift(self):
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        x = shift.shift(x, (2, 0), mode="reflect")
        print(x)
        x[:, [0, 1]] = x[:, [1, 0]]
        print(x)
        # print(np.roll(x, (1, 0)))

    def test_wall_builder(self):
        mdp = MDP(width=3, length=3, p_walk=0.7, p_run=0.6, reward_walk=1, reward_run=1, discount=0, wall_list=[(1, 1), (0, 1)], exit_list=[((0, 2), 1), ((1, 2), -1)])
        walk_up = mdp.wall_up_1
        self.assertListEqual([(2, 1), (1, 1)], walk_up)
        walk_left = mdp.wall_left_1
        self.assertListEqual([(2, 1), (1, 1)], walk_up)
        walk_down = mdp.wall_down_1
        self.assertListEqual([(0, 1)], walk_down)
        walk_right = mdp.wall_right_2
        self.assertListEqual([(1, 0), (0, 0)], walk_right)

        run_up = mdp.wall_up_2
        self.assertListEqual([(2, 1), (1, 1), (2, 1)], run_up)
        run_left = mdp.wall_left_2
        self.assertListEqual([(1, 2), (0, 2)], run_left)
        run_down = mdp.wall_down_2
        self.assertListEqual([(0, 1)], run_down)
        run_right = mdp.wall_right_2
        self.assertListEqual([(1, 0), (0, 0)], run_right)

    def test_policy_evaluation(self):
        mdp = MDP(width=3, length=3, p_walk=0.7, p_run=0.6, reward_walk=1, reward_run=1, discount=0, wall_list=[(1, 1)], exit_list=[((0, 2), 1), ((1, 2), -1)])
        mdp.value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        walk_up = mdp.policy_evaluation(mdp.walk_up)
        test.assert_array_almost_equal([[1.15, 2, 2.85], [1.9, 2.9, 3.9], [5.05, 8, 6.75]], walk_up)

        run_up = mdp.policy_evaluation(mdp.run_up)
        test.assert_array_almost_equal(np.array([[0.6 + 0.2 + 0.6, 1.2 + 0.4 + 0.4, 1.8 + 0.2 + 0.6],
                                                 [2.4 + 0.8 + 0.8, 5, 3.6 + 1.2 + 1.2],
                                                 [0.6 + 1.4 + 1.8, 4.8 + 1.6 + 1.6, 1.8 + 1.4 + 1.8]]), run_up)

    def test_small_value_iteration_no_wall(self):
        mdp = MDP(width=3, length=3, p_walk=0.7, p_run=0.7, reward_walk=0, reward_run=0, discount=0.1, e=1e-8, exit_list=[((0, 2), 1), ((1, 2), -1)])
        mdp.value_iteration()
        test.assert_array_almost_equal(np.array([[7, 3, 4], [0, 2, 0], [4, 4, 4]]), mdp.policy)

    def test_small_value_iteration_1_wall(self):
        mdp = MDP(width=3, length=3, p_walk=0.7, p_run=0.7, reward_walk=0, reward_run=0, wall_list=[(1, 1)], discount=0.1, exit_list=[((0, 2), 1), ((1, 2), -1)])
        mdp.value_iteration()
        test.assert_array_almost_equal(np.array([[7, 3, 0], [0, 2, 0], [4, 3, 4]]), mdp.policy)

    def test_case1(self):
        mdp = MDP(width=6, length=5, p_walk=0.8, p_run=0.6, reward_walk=-0.3, reward_run=-0.2, wall_list=[(1, 1), (4, 3)], discount=0.7, exit_list=[((0, 2), 10), ((2, 4), 5)], e=1e-8)
        mdp.value_iteration()
        test.assert_equal(np.array([[7., 3., 0., 2., 6., 6.],
                                    [0., 0., 0., 0., 0., 2.],
                                    [4., 3., 4., 4., 4., 2.],
                                    [4., 0., 4., 0., 0., 0.],
                                    [4., 4., 4., 4., 4., 2.]]), mdp.policy)

    def test_output(self):
        mdp = MDP(width=6, length=5, p_walk=0.8, p_run=0.6, reward_walk=-0.3, reward_run=-0.2, wall_list=[(1, 1), (4, 3)], discount=0.7, exit_list=[((0, 2), 10), ((2, 4), 5)], e=1e-8)
        mdp.policy = np.array([[7., 3., 0., 2., 6., 6.],
                               [0., 0., 0., 0., 0., 2.],
                               [4., 3., 4., 4., 4., 2.],
                               [4., 0., 4., 0., 0., 0.],
                               [4., 4., 4., 4., 4., 2.]])
        str = mdp.out_put()
        self.assertEqual(str,
                         """Run Right,Walk Right,Exit,Walk Left,Run Left,Run Left
Walk Up,None,Walk Up,Walk Up,Walk Up,Walk Left
Run Up,Walk Right,Run Up,Run Up,Exit,Walk Left
Run Up,Walk Up,Run Up,Walk Up,Walk Up,Walk Up
Run Up,Run Up,Run Up,None,Run Up,Walk Left\n"""
                         )

    def test_read_configuration(self):
        configuration = Configuration()
        # (width=6, length=5, p_walk=0.8, p_run=0.6, reward_walk=-0.3, reward_run=-0.2, wall_list=[(1, 1), (4, 3)], 
        # discount=0.7, exit_list=[((0, 2), 10), ((2, 4), 5)], e=1e-4)
        width, length, p_walk, p_run, r_walk, r_run, discount, wall_list, exit_list = configuration.read_file("input1.txt")
        self.assertEqual(width, 6)
        self.assertEqual(length, 5)
        self.assertEqual(0.8, p_walk)
        self.assertEqual(0.6, p_run)
        self.assertEqual(-0.3, r_walk)
        self.assertEqual(-0.2, r_run)
        self.assertListEqual([(1, 1), (4, 3)], wall_list)
        self.assertEqual(0.7, discount)
        self.assertListEqual([((0, 2), 10), ((2, 4), 5)], exit_list)

    def test_integration2(self):
        configuration = Configuration()
        width, length, p_walk, p_run, r_walk, r_run, discount, wall_list, exit_list = configuration.read_file("input2.txt")
        mdp = MDP(width=width, length=length, p_walk=p_walk, p_run=p_run, reward_walk=r_walk, reward_run=r_run, wall_list=wall_list, discount=discount, exit_list=exit_list, e=1e-8)
        mdp.value_iteration()
        str = mdp.out_put()
        with open("my_output2.txt", 'w') as f:
            f.write(str)
        with open("output2.txt", "r") as f:
            data = f.read()
            self.assertEqual(data, str)
