"""
@ File Name     :   utils.py
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


def loss_func(pred: np.array, label: np.array):

    loss = ((pred - label) ** 2).sum() / 2 / pred.shape[0]
    loss_list = pred - label
    return loss, loss_list


def optimizer(model, loss_list, train_data, learning_rate):

    para_update = (
        learning_rate * (loss_list * train_data).sum(axis=0) / loss_list.shape[0]
    )
    new_para = model.beta - para_update
    model.beta = new_para
