from utils.Generator import Generator
import os
import pickle
import tensorflow as tf
from OpenPTrack.SkeletonPreProcessing import SkeletonPreProcessing
import numpy as np

class OpenPTrackGenerator(Generator):

    def __init__(self, batch_size, dataset_path, skeleton_path, t=0, n_class=0, train=False, average=False, mean = None, std = None):
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
        self.average = average
        self.preprocessing = SkeletonPreProcessing()
        if mean is not None:
            self.mean = np.loadtxt(mean, delimiter=",")
        if std is not None:
            self.std = np.loadtxt(std, delimiter=",")

    def normalize(self, skeletons):
        return (skeletons - self.mean) / (self.std + 1.e-13)
    def openSkeleton(self,subject, filename):
        filepath = os.path.join(self.skeleton_path+subject, str(filename)+".pkl")
        pickle_in = open(filepath, "rb")
        skeleton = pickle.load(pickle_in)
        skeleton.joints.normalize()
        #print(skeleton.getFlattenCoordinates())
        return skeleton.getFlattenCoordinates()
    def getFlow(self, batch_index):
        if (batch_index + 1) * self.batch_size > len(self.data_set.index):
            data = self.data_set[self.len_data - self.batch_size: self.len_data]
        else:
            data = self.data_set[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]

        x = []
        labels = []
        weights = []
        for index, row in data.iterrows():

            skeletons = np.array([self.openSkeleton(subject=row.subject, filename=filename) for filename in range(row.time_min, row.time_max)])
            skeletons = self.preprocessing.smoothing(self.normalize(skeletons))
            x.append(tf.convert_to_tensor(skeletons, dtype=tf.float32))
            labels.append(row["class"])
            weights.append(row["weight"])

        x = tf.stack(x, axis=0)
        y = np.array(labels).flatten()
        w = np.array(weights).flatten()

        if self.average:
            y = tf.reduce_mean(tf.reshape(tf.convert_to_tensor(self.lb.transform(y).astype(float), dtype=tf.float32), shape=(self.batch_size, self.T, self.n_class)), axis=1)
        else:
            y = tf.reshape(tf.convert_to_tensor(self.lb.transform(y).astype(float), dtype=tf.float32), shape=(self.batch_size, self.T, self.n_class))
        w = tf.reshape(tf.convert_to_tensor(w, dtype=tf.float32), shape=(self.batch_size, self.T))
        return x, y, w
