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

import seaborn as sns
import matplotlib.pyplot as plt


def draw_loss(x_list, loss_list, save_path):
    sns.lineplot(x=x_list, y=loss_list)
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.show()
    plt.savefig(save_path)


if __name__ == "__main__":
    x_list = [0, 0.5, 1, 1.5, 2, 2.5]
    loss_list = [10, 1, 1, 1, 1, 1]
    draw_loss(x_list, loss_list, "./demo.png")
