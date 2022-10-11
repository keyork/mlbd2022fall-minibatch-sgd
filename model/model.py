"""
@ File Name     :   model.py
@ Time          :   2022/10/04
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   model, y=beta*x
@ Class List    :   LinearModel -- main model
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
