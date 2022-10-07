"""
@ File Name     :   dataloader.py
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
import pandas as pd


class BatchDataLoader:
    def __init__(self, data_path, batch_size, shuffle=True):
        raw_data = pd.read_csv(data_path).values
        raw_data[:, 0] = 1
        self.data_num = raw_data.shape[0]
        if shuffle:
            data_idx = np.random.permutation(self.data_num)
            full_data = raw_data[data_idx]
        else:
            full_data = raw_data
        self.data = full_data[:, :4]
        self.label = full_data[:, 4:]
        self.batch_size = batch_size
        self.batch_num = self.data_num / self.batch_size
        if self.data_num % self.batch_size != 0:
            self.batch_num += 1

    def __iter__(self):
        self.batch_id = 0  # start from 0
        return self

    def __next__(self):
        self.batch_id += 1
        if self.batch_id + 1 <= self.batch_num:
            return (
                self.data[
                    self.batch_id
                    * self.batch_size : min(
                        (self.batch_id + 1) * self.batch_size, self.data_num - 1
                    ),
                    :,
                ],
                self.label[
                    self.batch_id
                    * self.batch_size : min(
                        (self.batch_id + 1) * self.batch_size, self.data_num - 1
                    ),
                    :,
                ],
            )
        else:
            raise StopIteration
