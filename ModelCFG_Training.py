from utils.SkeletonGenerator import SkeletonGenerator
from models.CFGRNN import CFGRNN
import tensorflow as tf
from tensorflow.contrib import summary
import yaml
import os
import math

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
NUM_ITER = 100
THRESHOLD_LOSS = 1e+13
PREV_BEST = 0

# --------------------------Generator-----------------------#
skeleton_generator_train = SkeletonGenerator(batch_size=BATCH_SIZE, dataset_path=TRAINDATASET_PATH,
                                             skeleton_path=SKELETON_PATH, t=T, n_class=N_CLASS)
skeleton_generator_test = SkeletonGenerator(batch_size=VALIDATION_BATCH_SIZE, dataset_path=TESTDATASET_PATH,
                                            skeleton_path=SKELETON_PATH, t=T, n_class=N_CLASS)

# --------------------------Model-----------------------#
conf = cfg["CFGRNN_CONFIG"]
cfgRNN = CFGRNN(conf)

# --------------------------Optimizer-----------------------#
optimizer = tf.train.AdamOptimizer(learning_rate=LR)

# ---------------------------------------------Check points---------------------------------------------#
check_point = tf.train.Checkpoint(optimizer=optimizer, cfgRNN=cfgRNN, global_step=tf.train.get_or_create_global_step())

manager = tf.contrib.checkpoint.CheckpointManager(
    check_point, directory=CHECK_POINT_DIR, max_to_keep=20)
status = check_point.restore(manager.latest_checkpoint)

# ---------------------------------------------Tensorboard configuration---------------------------------#
summary_writer = summary.create_file_writer(TENSORBOARD_DIR)

with summary_writer.as_default(), summary.always_record_summaries():
    for i in range(1, NUM_ITER):
        training_loss = 0
        training_acc = 0
        validation_loss = 0
        validation_acc = 0
        skeleton_generator_train.suffle()
        for h in range(skeleton_generator_train.num_batch):
            x, t = skeleton_generator_train.getFlow(h)

            with tf.GradientTape() as tape:
                y_jump = tf.reduce_mean(cfgRNN.action(x, action="jumping"), axis=1)
                y_jumpJ = tf.reduce_mean(cfgRNN.action(x, action="jumpingJ"), axis=1)
                y_wav = tf.reduce_mean(cfgRNN.action(x, action="waving"), axis=1)
                y_wavR = tf.reduce_mean(cfgRNN.action(x, action="wavingR"), axis=1)
                y_clap = tf.reduce_mean(cfgRNN.action(x, action="clap"), axis=1)
                y_punch = tf.reduce_mean(cfgRNN.action(x, action="punching"), axis=1)
                y_bend = tf.reduce_mean(cfgRNN.action(x, action="bending"), axis=1)
                y_throw = tf.reduce_mean(cfgRNN.action(x, action="throwing"), axis=1)
                y_sitdown = tf.reduce_mean(cfgRNN.action(x, action="sitDown"), axis=1)
                y_standUp = tf.reduce_mean(cfgRNN.action(x, action="standUp"), axis=1)
                y_sitTstand = tf.reduce_mean(cfgRNN.action(x, action="sitDownTstandUp"), axis=1)

                y = tf.concat(
                    [y_jump, y_jumpJ, y_bend, y_punch, y_wav, y_wavR, y_clap, y_throw, y_sitTstand, y_sitdown, y_standUp
                     ], axis=-1)

                y_prob = tf.nn.softmax(y)

                # print(t)

                loss = tf.losses.softmax_cross_entropy(logits=y, onehot_labels=t)

                training_acc += tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(y_prob, 1), tf.argmax(t, 1)), dtype=tf.float32))

                variables = cfgRNN.trainable_variables
                grads = tape.gradient(loss, variables)

                clipped_grads = [tf.clip_by_norm(g, 1.) for g in grads]
                optimizer.apply_gradients(zip(clipped_grads, variables),
                                          global_step=tf.train.get_or_create_global_step())
            training_loss += loss

        for h in range(skeleton_generator_test.num_batch):
            x, t = skeleton_generator_test.getFlow(h)

            y_jump = tf.reduce_mean(cfgRNN.action(x, action="jumping"), axis=1)
            y_jumpJ = tf.reduce_mean(cfgRNN.action(x, action="jumpingJ"), axis=1)
            y_wav = tf.reduce_mean(cfgRNN.action(x, action="waving"), axis=1)
            y_wavR = tf.reduce_mean(cfgRNN.action(x, action="wavingR"), axis=1)
            y_clap = tf.reduce_mean(cfgRNN.action(x, action="clap"), axis=1)
            y_punch = tf.reduce_mean(cfgRNN.action(x, action="punching"), axis=1)
            y_bend = tf.reduce_mean(cfgRNN.action(x, action="bending"), axis=1)
            y_throw = tf.reduce_mean(cfgRNN.action(x, action="throwing"), axis=1)
            y_sitdown = tf.reduce_mean(cfgRNN.action(x, action="sitDown"), axis=1)
            y_standUp = tf.reduce_mean(cfgRNN.action(x, action="standUp"), axis=1)
            y_sitTstand = tf.reduce_mean(cfgRNN.action(x, action="sitDownTstandUp"), axis=1)

            y = tf.concat(
                [y_jump, y_jumpJ, y_bend, y_punch, y_wav, y_wavR, y_clap, y_throw, y_sitTstand, y_sitdown, y_standUp
                 ], axis=-1)

            y_prob = tf.nn.softmax(y)
            validation_loss += tf.losses.softmax_cross_entropy(logits=y, onehot_labels=t)

            validation_acc += tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(y_prob, 1), tf.argmax(t, 1)), dtype=tf.float32))

        training_loss = training_loss / skeleton_generator_train.num_batch
        training_acc = training_acc / skeleton_generator_train.num_batch
        validation_loss = validation_loss / skeleton_generator_test.num_batch
        validation_acc = validation_acc / skeleton_generator_test.num_batch
        print("Itteration = %f, Loss =  %f, accuracy = %f, validation loss = %f, validation accuracy = %f" % (
        i, training_loss, training_acc, validation_loss, validation_acc))
        summary.scalar("validation_loss", validation_loss, step=tf.train.get_or_create_global_step())
        summary.scalar("training_loss", training_loss, step=tf.train.get_or_create_global_step())

        summary.scalar("training_acc", training_acc, step=tf.train.get_or_create_global_step())
        summary.scalar("validation_acc", validation_acc, step=tf.train.get_or_create_global_step())

        if (i - PREV_BEST) % 5 == 0 or i == 15:
            print("Learning rate is changed")
            optimizer._lr = optimizer._lr * math.sqrt(0.2)

        if (i - PREV_BEST) > 15:
            break

        if validation_loss < THRESHOLD_LOSS or i == 1:
            manager.save()
            THRESHOLD_LOSS = validation_loss
            PREV_BEST = i
