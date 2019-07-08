from utils.Generator import Generator
import pandas as pd
import numpy as np
import os
import yaml
import tensorflow as tf
class SkeletonGenerator(Generator):
    def __init__(self, batch_size, dataset_path, skeleton_path, t=0, n_class=0, train=False, dt_type=None, simple_complex=False):
        '''

        :param batch_size:
        :param dataset_path:
        :param skeleton_path:
        :param t:
        :param n_class:
        :param train:
        :param dt_type: data type b= berkeley
        '''
        Generator.__init__(self, batch_size=batch_size, dataset_path=dataset_path, t=t, n_class=n_class, train=train)

        self.skeleton_path = skeleton_path
        self.dt_type = dt_type

        if simple_complex:
            self.data_set = self.data_set[(self.data_set["action"] == 5) & (self.data_set["action"] == 6) & (self.data_set["action"] == 8) & (self.data_set["action"] == 11)]
            self.len_data = len(self.data_set.index)

    def getFlow(self, batch_index):
        if (batch_index + 1) * self.batch_size > len(self.data_set.index):
            data = self.data_set[self.len_data - self.batch_size: self.len_data]
        else:
            data = self.data_set[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]

        x = []
        for id in data.index:
            sampleData = pd.read_pickle(os.path.join(self.skeleton_path, data.loc[id].filename)).reset_index()
            lenData = len(sampleData.index)
            # if self.dt_type == "b":
            #     resample = np.arange(0, lenData, 16)
            # else:
            #     resample = np.arange(0, lenData, int(lenData/self.T))
            resample = np.arange(0, lenData, int(lenData / self.T))
            sampleData = sampleData.loc[resample][0:self.T].values
            x.append(tf.convert_to_tensor(sampleData, dtype=tf.float32))
        x = tf.stack(x, axis=0)
        y = tf.convert_to_tensor(self.lb.transform(data.action.tolist()).astype(float), dtype=tf.float32)

        return x, y

    def suffle(self):
        self.data_set.sample(frac=1.)

