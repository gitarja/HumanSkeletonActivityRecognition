from utils.SkeletonGenerator import SkeletonGenerator
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
TRAINDATASET_PATH = os.path.join(cfg["DIR_MOCAP"], cfg["CFGRNN_CONFIG"]["TRAINDATASET"])
TESTDATASET_PATH = os.path.join(cfg["DIR_MOCAP"], cfg["CFGRNN_CONFIG"]["TESTDATASET"])
SKELETON_PATH = cfg["DIR_MOCAP_NORMALIZED_RESULT"]
CHECK_POINT_DIR = cfg["CFGRNN_CONFIG"]["CHECK_POINT_DIR"]
TENSORBOARD_DIR = cfg["CFGRNN_CONFIG"]["TENSORBOARD_DIR"]
T = cfg["CFGRNN_CONFIG"]["T"]
N_CLASS = cfg["CFGRNN_CONFIG"]["N_CLASS"]
LR = cfg["LR"]
NUM_ITER = 1000
THRESHOLD_LOSS = 1e+13
PREV_BEST = 0

# --------------------------Generator-----------------------#
skeleton_generator_train = SkeletonGenerator(batch_size=BATCH_SIZE, dataset_path=TRAINDATASET_PATH,
                                             skeleton_path=SKELETON_PATH, t=T, n_class=N_CLASS)
skeleton_generator_train_complex = SkeletonGenerator(batch_size=BATCH_SIZE, dataset_path=TRAINDATASET_PATH,
                                                     skeleton_path=SKELETON_PATH, t=T, n_class=N_CLASS,
                                                     simple_complex=True)
skeleton_generator_test = SkeletonGenerator(batch_size=VALIDATION_BATCH_SIZE, dataset_path=TESTDATASET_PATH,
                                            skeleton_path=SKELETON_PATH, t=T, n_class=N_CLASS)

# --------------------------Model-----------------------#
conf = cfg["CFGRNN_CONFIG"]
cfgRNN = CFGRNN(conf)

# --------------------------Simulated Annealing-----------------------#
sa = SimulatedAnnealing()
LR = sa.random_start()

# --------------------------Optimizer-----------------------#
optimizer = tf.train.RMSPropOptimizer(learning_rate=LR)

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
optimize = True
with summary_writer.as_default(), summary.always_record_summaries():
    for i in range(1, NUM_ITER):
        training_loss = 0
        training_acc = 0
        validation_loss = 0
        validation_acc = 0
        #skeleton_generator_train.suffle()
        # for h in range(skeleton_generator_train.num_batch):
        # if i % 2 == 0 & i < 7:
        #     x, t = skeleton_generator_train_complex.getFlow(h)
        # else:
        h = i % skeleton_generator_train.num_batch
        x, t = skeleton_generator_train.getFlow(h)

        with tf.GradientTape() as tape:
            y = cfgRNN.predict(x)

            y_prob = tf.nn.softmax(y)

            # print(t)

            loss = tf.losses.softmax_cross_entropy(logits=y, onehot_labels=t)
            #add loss to vocabulary
            losses.append(loss)
            training_labels.append(np.argmax(t.numpy(), axis=1).tolist())

            training_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(y_prob, 1), tf.argmax(t, 1)), dtype=tf.float32))

            # --------------------------Oprimize the learning rate using SA-----------------------#
            lr_bc = optimizer._learning_rate
            if optimize:

                T = sa.temperature(0.2, step=i)
                fraction = i / float(NUM_ITER)
                new_lr = sa.random_neighbour(optimizer._learning_rate, fraction)
                optimizer._learning_rate = new_lr


            # --------------------------Compute gradient descent-----------------------#
            variables = cfgRNN.trainable_variables

            grads = tape.gradient(loss, variables)

            clipped_grads = [tf.clip_by_norm(g, 1.) for g in grads]
            optimizer.apply_gradients(zip(clipped_grads, variables),
                                      global_step=tf.train.get_or_create_global_step())
            # record the weights
            # body parts

            # body_combined = tf.concat([cfgRNN.rh.weights[0], cfgRNN.lh.weights[0], cfgRNN.rf.weights[0], cfgRNN.lf.weights[0], cfgRNN.t.weights[0]], 0)
            # unary_combined = tf.concat(
            #     [cfgRNN.mF.weights[0], cfgRNN.mS.weights[0], cfgRNN.forward.weights[0]], 0)
            # binary_combined = tf.concat([cfgRNN.and_op.weights[0], cfgRNN.or_op.weights[0], cfgRNN.touch_op.weights[0], cfgRNN.then_op.weights[0], cfgRNN.with_op.weights[0], cfgRNN.neg_op.weights[0]], 0)
            # unary_combined_rec = tf.concat(
            #     [cfgRNN.mF.weights[1], cfgRNN.mS.weights[1], cfgRNN.forward.weights[1]], 0)
            # binary_combined_rec = tf.concat(
            #     [cfgRNN.and_op.weights[1], cfgRNN.or_op.weights[1], cfgRNN.touch_op.weights[1],
            #      cfgRNN.then_op.weights[1], cfgRNN.with_op.weights[1], cfgRNN.neg_op.weights[1]], 0)
            #
            # summary.histogram("body_combined", body_combined,
            #                   step=tf.train.get_or_create_global_step())
            # summary.histogram("unary_combined", unary_combined,
            #                   step=tf.train.get_or_create_global_step())
            # summary.histogram("binary_combined", binary_combined,
            #                   step=tf.train.get_or_create_global_step())
            # summary.histogram("unary_combined_rec", unary_combined_rec,
            #                   step=tf.train.get_or_create_global_step())
            # summary.histogram("binary_combined_rec", binary_combined_rec,
            #                   step=tf.train.get_or_create_global_step())



        #training_loss += loss

        for j in range(skeleton_generator_test.num_batch):
            x, t = skeleton_generator_test.getFlow(j)

            y = cfgRNN.predict(x)

            y_prob = tf.nn.softmax(y)
            validation_loss += tf.losses.softmax_cross_entropy(logits=y, onehot_labels=t)

            validation_acc += tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(y_prob, 1), tf.argmax(t, 1)), dtype=tf.float32))

        #training_loss = training_loss / skeleton_generator_train.num_batch
        # training_acc = training_acc / skeleton_generator_train.num_batch
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

        # if ((i - PREV_BEST) % 5 == 0 or i % 10 == 0) and i < 30:
        #     print("Learning rate is changed")
        #     # optimizer._lr = optimizer._lr * math.sqrt(0.2)
        #     optimizer._learning_rate = optimizer._learning_rate / math.sqrt(0.3)
        #
        # if ((i - PREV_BEST) % 5 == 0) and i > 30:
        #     print("Learning rate is changed")
        #     # optimizer._lr = optimizer._lr * math.sqrt(0.2)
        #     optimizer._learning_rate = optimizer._learning_rate * math.sqrt(0.2)


        if sa.acceptance_probability(validation_losses[i-2], validation_loss, T) > rn.random():
            optimize = False
        else:
            optimize = True
            optimizer._learning_rate = lr_bc

        if (i % 5) == 0:
            optimize = True

        print("Current learning rate is %f", optimizer._learning_rate)

        if validation_loss < THRESHOLD_LOSS or i == 1:
            manager.save()
            THRESHOLD_LOSS = validation_loss
            PREV_BEST = i

        if (i - PREV_BEST) > 15:
            break

# with open('outfile', 'wb') as fp:
#     pickle.dump(training_labels, fp)
