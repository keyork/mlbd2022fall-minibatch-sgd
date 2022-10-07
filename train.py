"""
@ File Name     :   train.py
@ Time          :   2022/10/04
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   None
@ Function List :   func1() -- func desc1
@ Class List    :   Class1 -- class1 desc1
@ Details       :   None
"""

import argparse
from dataloader import BatchDataLoader
from model import LinearModel
from utils import loss_func, optimizer


def train(config):

    model = LinearModel()
    learning_rate = config.learning_rate
    dataloader = iter(BatchDataLoader("./sgd_data.CSV", config.batch_size))
    for iter_id in range(5000):
        for data, label in dataloader:
            learning_rate = config.learning_rate
            y_pred = model.forward(data)
            loss, loss_list = loss_func(y_pred, label)
            optimizer(model, loss_list, data, learning_rate)
            print(loss)
    print(model.beta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="batch_size")
    parser.add_argument("--learning_rate", type=float, help="learning_rate")
    args = parser.parse_args()

    train(args)
