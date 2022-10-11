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
from utils.drawtoolbox import draw_loss, draw_loss_single


def train(config):
    """train linear model

    Args:
        config (args): config

    Returns:
        x_log: x axis
        loss_log: loss list, y axis
    """
    model = LinearModel()
    learning_rate = config.lr
    dataloader = iter(BatchDataLoader("./data/sgd_data.CSV", config.bs))
    loss_log = []
    x_log = []
    for iter_id in range(config.iter):
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
            x_log.append(iter_id + batch_num * config.bs / dataloader.data_num)
            batch_num += 1
    print(model.beta)
    return x_log, loss_log


def main(config):
    LOGGER.info("Start")
    x_list = []
    loss_list = []
    if config.compbs:
        img_name = config.img_path + "img-bs_compare-iter_{}-bls_{}-lr_{}.png".format(
            config.iter, config.bls, config.lr
        )
        # batch_size_list = [1, 10, 50, 100, 500, 1000, 4000]
        batch_size_list = [1, 20, 4000]
        for batch_size in batch_size_list:
            config.bs = batch_size
            x_log, loss_log = train(config)
            x_list.append(x_log)
            loss_list.append(loss_log)
        draw_loss(
            x_list, loss_list, batch_size_list, img_name, "loss curve & batch size"
        )
    elif config.compbls:
        img_name = config.img_path + "img-bs_{}-iter_{}-bls_compare-lr_{}.png".format(
            config.bs, config.iter, config.lr
        )
        for bls in [True, False]:
            config.bls = bls
            x_log, loss_log = train(config)
            x_list.append(x_log)
            loss_list.append(loss_log)
        draw_loss(
            x_list, loss_list, [True, False], img_name, "loss curve & back line search"
        )
    else:
        img_name = config.img_path + "img-bs_{}-iter_{}-bls_{}-lr_{}.png".format(
            config.bs, config.iter, config.bls, config.lr
        )
        x_log, loss_log = train(config)
        draw_loss_single(x_log, loss_log, img_name)
    LOGGER.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=50, help="batch size")
    parser.add_argument("--lr", type=float, default=1, help="learning rate")
    parser.add_argument("--iter", type=int, default=1, help="iter num")
    parser.add_argument("--bls", type=str2bool, default=True, help="back line search")
    parser.add_argument(
        "--compbs", type=str2bool, default=False, help="compare batch size"
    )
    parser.add_argument(
        "--compbls", type=str2bool, default=False, help="compare back line search"
    )
    parser.add_argument("--img_path", type=str, default="./img/", help="img path")
    parser.add_argument(
        "--data_path", type=str, default="./data/sgd_data.CSV", help="data path"
    )
    args = parser.parse_args()

    main(args)
