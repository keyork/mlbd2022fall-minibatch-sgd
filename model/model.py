"""
@ File Name     :   model.py
@ Time          :   2022/10/04
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   None
@ Function List :   func1() -- func desc1
@ Class List    :   Class1 -- class1 desc1
@ Details       :   None
"""


import numpy as np


class LinearModel:
    def __init__(self):

        self.beta = np.array([0, 0, 0, 0])

    def init_model(self):

        self.beta = np.array([0, 0, 0, 0])

    def forward(self, x: np.array):

        y = x * self.beta
        return np.array([y.sum(axis=1)]).T
