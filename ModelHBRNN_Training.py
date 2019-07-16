from utils.SkeletonGenerator import SkeletonGenerator
from models.HBRNN import HBRNN
import tensorflow as tf
from tensorflow.contrib import summary
import yaml
import os
from models.SimulatedAnnealing import SimulatedAnnealing
import math
import numpy.random as rn

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution()

# --------------------------Configuration-----------------------#
with open("conf/setting.yml") as ymfile:
    cfg = yaml.load(ymfile)

BATCH_SIZE = cfg["HBRNN_CONFIG"]["BATCH_SIZE"]
VALIDATION_BATCH_SIZE = cfg["HBRNN_CONFIG"]["VALIDATION_BATCH_SIZE"]
TRAINDATASET_PATH = os.path.join(cfg["DIR_MOCAP"], cfg["HBRNN_CONFIG"]["TRAINDATASET"])
TESTDATASET_PATH = os.path.join(cfg["DIR_MOCAP"], cfg["HBRNN_CONFIG"]["TESTDATASET"])
SKELETON_PATH = cfg["DIR_MOCAP_NORMALIZED_RESULT"]
CHECK_POINT_DIR = cfg["HBRNN_CONFIG"]["CHECK_POINT_DIR"]
TENSORBOARD_DIR = cfg["HBRNN_CONFIG"]["TENSORBOARD_DIR"]
T = cfg["HBRNN_CONFIG"]["T"]
N_CLASS = cfg["HBRNN_CONFIG"]["N_CLASS"]
LR = 1.e-3
NUM_ITER = 100
THRESHOLD_LOSS = 1e+13
PREV_BEST = 0

# --------------------------Generator-----------------------#
skeleton_generator_train = SkeletonGenerator(batch_size=BATCH_SIZE, dataset_path=TRAINDATASET_PATH,
                                             skeleton_path=SKELETON_PATH, t=T, n_class=N_CLASS)
skeleton_generator_test = SkeletonGenerator(batch_size=VALIDATION_BATCH_SIZE, dataset_path=TESTDATASET_PATH,
                                            skeleton_path=SKELETON_PATH, t=T, n_class=N_CLASS)

# --------------------------Model-----------------------#
conf = cfg["HBRNN_CONFIG"]
teacher_model = HBRNN.Discriminator(conf)

# --------------------------Simulated Annealing-----------------------#
sa = SimulatedAnnealing()
LR = sa.random_start()

# --------------------------Optimizer-----------------------#
optimizer = tf.train.RMSPropOptimizer(learning_rate=LR)

# ---------------------------------------------Check points---------------------------------------------#
check_point = tf.train.Checkpoint(optimizer=optimizer, teacher_model=teacher_model,
                                  global_step=tf.train.get_or_create_global_step())

manager = tf.contrib.checkpoint.CheckpointManager(
    check_point, directory=CHECK_POINT_DIR, max_to_keep=20)
status = check_point.restore(manager.latest_checkpoint)

# ---------------------------------------------Tensorboard configuration---------------------------------#
summary_writer = summary.create_file_writer(TENSORBOARD_DIR)
validation_losses = [1000, ]
optimize = True
with summary_writer.as_default(), summary.always_record_summaries():
    for i in range(1, NUM_ITER):
        training_loss = 0
        training_acc = 0
        validation_loss = 0
        validation_acc = 0
        skeleton_generator_train.suffle()
        # for h in range(skeleton_generator_train.num_batch):
        h = i % skeleton_generator_train.num_batch
        x, t = skeleton_generator_train.getFlow(h)

        with tf.GradientTape() as tape:
            y = teacher_model.berkeley(x)
            # logits = tf.reduce_sum(tf.reduce_mean([y0, y1], axis=0), axis=1)
            y = tf.reduce_mean(y, axis=1)
            y_prob = tf.nn.softmax(y)

            loss = tf.losses.softmax_cross_entropy(logits=y, onehot_labels=t)

            training_acc += tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(y_prob, 1), tf.argmax(t, 1)), dtype=tf.float32))

            # --------------------------Oprimize the learning rate using SA-----------------------#
            lr_bc = optimizer._learning_rate
            if optimize:
                T = sa.temperature(0.2, step=i)
                fraction = i / float(NUM_ITER)
                new_lr = sa.random_neighbour(optimizer._learning_rate, fraction)
                optimizer._learning_rate = new_lr

            # --------------------------Compute gradient descent-----------------------#
            variables = teacher_model.trainable_variables
            grads = tape.gradient(loss, variables)

            clipped_grads = [tf.clip_by_norm(g, 1.) for g in grads]
            optimizer.apply_gradients(zip(clipped_grads, variables), global_step=tf.train.get_or_create_global_step())
        #training_loss += loss

        for h in range(skeleton_generator_test.num_batch):
            x, t = skeleton_generator_test.getFlow(h)
            y = teacher_model.berkeley(x)
            # logits = tf.reduce_sum(tf.reduce_mean([y0, y1], axis=0), axis=1)
            y = tf.reduce_mean(y, axis=1)
            y_prob = tf.nn.softmax(y)
            validation_loss += tf.losses.softmax_cross_entropy(logits=y, onehot_labels=t)

            validation_acc += tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(y_prob, 1), tf.argmax(t, 1)), dtype=tf.float32))

        # training_loss = training_loss / skeleton_generator_train.num_batch
        # training_acc = training_acc / skeleton_generator_train.num_batch


        training_loss = loss
        training_acc = training_acc
        validation_loss = validation_loss / skeleton_generator_test.num_batch
        validation_losses.append(validation_loss)
        validation_acc = validation_acc / skeleton_generator_test.num_batch
        print("Itteration = %f, Loss =  %f, accuracy = %f, validation loss = %f, validation accuracy = %f" % (
        i, training_loss, training_acc, validation_loss, validation_acc))
        summary.scalar("validation_loss", validation_loss, step=tf.train.get_or_create_global_step())
        summary.scalar("training_loss", training_loss, step=tf.train.get_or_create_global_step())

        summary.scalar("training_acc", training_acc, step=tf.train.get_or_create_global_step())
        summary.scalar("validation_acc", validation_acc, step=tf.train.get_or_create_global_step())

        # if (i - PREV_BEST) % 5 == 0 or i == 15:
        #     print("Learning rate is changed")
        #     optimizer._lr = optimizer._lr * math.sqrt(0.2)

        if sa.acceptance_probability(validation_losses[i-2], validation_loss, T) > rn.random():
            optimize = False
        else:
            optimize = True
            optimizer._learning_rate = lr_bc

        print("Current learning rate is %f", optimizer._learning_rate)

        if validation_loss < THRESHOLD_LOSS or i == 1:
            manager.save()
            THRESHOLD_LOSS = validation_loss
            PREV_BEST = i

        if (i - PREV_BEST) > 15:
            break
