from utils.OpenPTrackGenerator import OpenPTrackGenerator
from models.CFGRNN import CFGRNN
import tensorflow as tf
from tensorflow.contrib import summary
import yaml
import os
import math
import numpy as np
from models.SimulatedAnnealing import SimulatedAnnealing
import pickle
import numpy.random as rn

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution()




# --------------------------Configuration-----------------------#
with open("conf/setting.yml") as ymfile:
    cfg = yaml.load(ymfile)

BATCH_SIZE = cfg["CFGRNN_CONFIG"]["BATCH_SIZE"]
VALIDATION_BATCH_SIZE = cfg["CFGRNN_CONFIG"]["VALIDATION_BATCH_SIZE"]

OPENPTRACK_PATH = cfg["DIR_OPENPTRACK"]
TRAINDATASET_PATH = os.path.join(OPENPTRACK_PATH, cfg["OPENPTRACK_CONFIG"]["TRAINDATASET"])
TESTDATASET_PATH = os.path.join(OPENPTRACK_PATH, cfg["OPENPTRACK_CONFIG"]["TESTDATASET"])
MEAN_FILE_PATH = os.path.join(OPENPTRACK_PATH, cfg["OPENPTRACK_CONFIG"]["MEANFILE"])
STD_FILE_PATH = os.path.join(OPENPTRACK_PATH, cfg["OPENPTRACK_CONFIG"]["STDFILE"])

CHECK_POINT_DIR = cfg["OPENPTRACK_CONFIG"]["CHECK_POINT_DIR"]
TENSORBOARD_DIR = cfg["OPENPTRACK_CONFIG"]["TENSORBOARD_DIR"]
T = cfg["CFGRNN_CONFIG"]["T"]
N_CLASS = cfg["OPENPTRACK_CONFIG"]["N_CLASS"]
LR = cfg["LR"]
NUM_ITER = 1000
THRESHOLD_LOSS = 1e+13
PREV_BEST = 0
AVERAGE = True

# --------------------------Generator-----------------------#
skeleton_generator_train = OpenPTrackGenerator(batch_size=BATCH_SIZE, dataset_path=TRAINDATASET_PATH,
                                             skeleton_path=OPENPTRACK_PATH, t=T, n_class=N_CLASS, average=AVERAGE, mean=MEAN_FILE_PATH, std=STD_FILE_PATH)

skeleton_generator_test = OpenPTrackGenerator(batch_size=VALIDATION_BATCH_SIZE, dataset_path=TESTDATASET_PATH,
                                            skeleton_path=OPENPTRACK_PATH, t=T, n_class=N_CLASS, average=AVERAGE, mean=MEAN_FILE_PATH, std=STD_FILE_PATH)

# --------------------------Model-----------------------#
conf = cfg["CFGRNN_CONFIG"]
cfgRNN = CFGRNN(conf, average=AVERAGE)

# --------------------------Simulated Annealing-----------------------#
sa = SimulatedAnnealing()
# LR = sa.random_start()

# --------------------------Optimizer-----------------------#
optimizer = tf.train.AdamOptimizer(learning_rate=LR)

# ---------------------------------------------Check points---------------------------------------------#
check_point = tf.train.Checkpoint(optimizer=optimizer, cfgRNN=cfgRNN, global_step=tf.train.get_or_create_global_step())

manager = tf.contrib.checkpoint.CheckpointManager(
    check_point, directory=CHECK_POINT_DIR, max_to_keep=20)
status = check_point.restore(manager.latest_checkpoint)

# ---------------------------------------------Tensorboard configuration---------------------------------#
summary_writer = summary.create_file_writer(TENSORBOARD_DIR)
training_labels = []
losses = []
validation_losses = [1000, ]
optimize = False
with summary_writer.as_default(), summary.always_record_summaries():
    for i in range(1, NUM_ITER):
        training_loss = 0
        training_acc = 0
        validation_loss = 0
        validation_acc = 0
        h = i % skeleton_generator_train.num_batch
        x, t, w = skeleton_generator_train.getFlow(h)

        with tf.GradientTape() as tape:
            y = cfgRNN.predictOpenPTrack(x)

            y_prob = tf.nn.softmax(y)

            # print(t)




            if AVERAGE:
                training_acc = tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(y_prob, 1), tf.argmax(t, 1)), dtype=tf.float32))
                loss = tf.losses.softmax_cross_entropy(logits=y, onehot_labels=t)

            else:
                training_acc = tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(y_prob, 2), tf.argmax(t, 2)), dtype=tf.float32))
                loss = tf.losses.softmax_cross_entropy(logits=y, onehot_labels=t, weights=w)

            # add loss to vocabulary
            losses.append(loss)
            training_labels.append(np.argmax(t.numpy(), axis=1).tolist())

            # --------------------------Oprimize the learning rate using SA-----------------------#
            lr_bc = optimizer._lr
            T = sa.temperature(0.2, step=i)
            print("The value of T is %f", T)
            if optimize:


                #fraction = i / float(NUM_ITER)
                new_lr = sa.random_neighbour(optimizer._lr, T)
                optimizer._lr = new_lr


            # --------------------------Compute gradient descent-----------------------#
            variables = cfgRNN.trainable_variables

            grads = tape.gradient(loss, variables)

            clipped_grads = [tf.clip_by_norm(g, 1.) for g in grads]
            optimizer.apply_gradients(zip(clipped_grads, variables),
                                      global_step=tf.train.get_or_create_global_step())

        # print("Itteration = %f, Loss =  %f, accuracy = %f" % (
        #     i, loss, training_acc))

        # training_loss += loss

        for j in range(skeleton_generator_test.num_batch):
            x, t, _ = skeleton_generator_test.getFlow(j)

            y = cfgRNN.predictOpenPTrack(x)

            y_prob = tf.nn.softmax(y)
            validation_loss += tf.losses.softmax_cross_entropy(logits=y, onehot_labels=t)

            if AVERAGE:
                validation_acc += tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(y_prob, 1), tf.argmax(t, 1)), dtype=tf.float32))
            else:
                validation_acc += tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(y_prob, 2), tf.argmax(t, 2)), dtype=tf.float32))

        training_loss = loss
        training_acc =  training_acc

        validation_loss = validation_loss / skeleton_generator_test.num_batch
        validation_losses.append(validation_loss)
        validation_acc = validation_acc / skeleton_generator_test.num_batch
        print("Itteration = %f, Loss =  %f, accuracy = %f, validation loss = %f, validation accuracy = %f" % (
            i, training_loss, training_acc, validation_loss, validation_acc))
        summary.scalar("validation_loss", validation_loss, step=tf.train.get_or_create_global_step())
        summary.scalar("training_loss", training_loss, step=tf.train.get_or_create_global_step())

        summary.scalar("training_acc", training_acc, step=tf.train.get_or_create_global_step())
        summary.scalar("validation_acc", validation_acc, step=tf.train.get_or_create_global_step())



        if sa.acceptance_probability(validation_losses[-2], validation_loss, T) > rn.random():
            optimize = False
            print("Current learning rate is %f", optimizer._lr)
        else:
            optimize = True
            optimizer._lr = lr_bc




        if validation_loss < THRESHOLD_LOSS or i == 1:
            manager.save()
            THRESHOLD_LOSS = validation_loss
            PREV_BEST = i

        if (i - PREV_BEST) > 15:
            break

