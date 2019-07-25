import random
import numpy as np
import pandas as pd
import math
from sklearn import preprocessing
class Generator:

    def __init__(self, batch_size, dataset_path, t=0, n_class=0, train=False):
        self.batch_size = batch_size
        self.T = t
        self.train = train
        extension = dataset_path.split(".")[-1]
        if extension == "csv":
            self.data_set = pd.read_csv(dataset_path)
        elif extension == "pkl":
            self.data_set = pd.read_pickle(dataset_path)
        self.lb = preprocessing.LabelBinarizer()
        self.lb.fit(range(1, n_class + 1))
        #self.lb.fit([1, 2, 5, 6])
        self.len_data = len(self.data_set.index)
        self.num_batch = math.ceil(self.len_data / self.batch_size)
        self.arr = range(self.num_batch)
        self.n_class = n_class




    def dataSequence(self, min, max):
        if self.train:

            numbers = np.arange(min + 1, max, int((max - min) / self.T))[:self.T]

        else:
            numbers = np.arange(min + 1, max, int((max - min) / self.T))[:self.T]
        return numbers
