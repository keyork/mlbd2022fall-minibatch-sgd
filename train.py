"""
@ File Name     :   train.py
@ Time          :   2022/10/04
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   None
@ Function List :   func1() -- func desc1
@ Class List    :   Class1 -- class1 desc1
@ Details       :   python train.py --batch_size 20 --learning_rate 2e-2
"""

import argparse

from data.dataloader import BatchDataLoader
from model.model import LinearModel
from utils.toolbox import LOGGER, str2bool
from utils.sgdtoolbox import bl_search, loss_func, optimizer
from utils.drawtoolbox import draw_loss


def train(config):

    model = LinearModel()
    learning_rate = config.lr
    dataloader = iter(BatchDataLoader("./data/sgd_data.CSV", config.bs))
    loss_log = []
    x_log = []
    for iter_id in range(10):
        iter_loss = 0
        batch_num = 0
        for data, label in dataloader:
            y_pred = model.forward(data)
            loss, loss_list = loss_func(y_pred, label)
            if config.bls:
                learning_rate = bl_search(
                    label, data, learning_rate, 1e-2, 0.1, 0.8, model, loss, loss_list
                )
            optimizer(model, loss_list, data, learning_rate)
            iter_loss += loss
            loss_log.append(loss)
            x_log.append(iter_id + batch_num / config.bs)
            batch_num += 1
        print(iter_loss / batch_num)
    print(model.beta)
    draw_loss(x_log, loss_log, "./img/demo.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=50, help="batch size")
    parser.add_argument("--lr", type=float, default=1, help="learning rate")
    parser.add_argument("--bls", type=str2bool, default=True, help="back line search")
    args = parser.parse_args()

    train(args)
