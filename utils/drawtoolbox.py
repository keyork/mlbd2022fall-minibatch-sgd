"""
@ File Name     :   drawtoolbox.py
@ Time          :   2022/10/11
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   None
@ Function List :   func1() -- func desc1
@ Class List    :   Class1 -- class1 desc1
@ Details       :   None
"""

import matplotlib.pyplot as plt


def draw_loss(x_list, loss_list, args_list, save_path, title):
    for i in range(len(loss_list)):
        plt.plot(x_list[i], loss_list[i])

    plt.legend((args_list), loc="upper right")
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.title(title)
    plt.savefig(save_path)
    plt.show()


def draw_loss_single(x_list, loss_list, save_path):
    plt.plot(x_list, loss_list)
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.savefig(save_path)
    plt.show()
