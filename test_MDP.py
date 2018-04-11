from unittest import TestCase
from hw3cs561s2018 import MDP
import numpy as np
import numpy.testing as test


class TestMDP(TestCase):
    def test_init(self):
        mpd = MDP(1000, 1000, 0.8, 0.7, 1, 9, 10)

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

        for direction in (mdp.up, mdp.left, mdp.right, mdp.down):
            for action in (mdp.walk_up, mdp.walk_left, mdp.walk_right, mdp.walk_down, mdp.run_up, mdp.run_left, mdp.run_right, mdp.run_down):
                p = np.squeeze(mdp.p[:, :, direction, action])
                if direction == action:  # walk_direction = direction
                    test.assert_array_almost_equal(p, np.ones((3, 3)) * 0.7)
                elif direction == action - 4:  # run_direction = direction
                    test.assert_array_almost_equal(p, np.ones((3, 3)) * 0.6)
                elif np.abs(direction - action) == 2 or np.abs(direction - action) == 6:  # opposite direction
                    test.assert_array_almost_equal(p, np.zeros((3, 3)))
                elif action < 4:  # walk
                    test.assert_array_almost_equal(np.ones((3, 3)) * 0.15, p)
                else: #run
                    test.assert_array_almost_equal(np.ones((3, 3)) * 0.2, p)
