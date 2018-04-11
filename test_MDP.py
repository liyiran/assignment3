from unittest import TestCase
from hw3cs561s2018 import MDP
import numpy as np


class TestMDP(TestCase):
    def test_init(self):
        mpd = MDP(1000, 1000, 0.8, 0.7, 1, 9, 10)

    def test_eistain(self):
        m1 = np.array([[1, 2, 3]])
        m2 = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
        big = np.concatenate((m1, m2), axis=0)
        # print(big)

        print(np.delete(big, (-1,-2), axis=0))
        # first_two = m2[-1:,...]
        # print(first_two)
        # print(np.concatenate((first_two, m2), axis=0))

    def test_max(self):
        m1 = np.array([1.5, 2, 3])
        m2 = np.array([0, 2.5, 2.9])
        max = np.maximum(m1, m2)
        print(m1 == max)
       
