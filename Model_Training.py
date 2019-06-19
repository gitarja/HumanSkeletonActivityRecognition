from utils.SkeletonGenerator import SkeletonGenerator
from models.HBRNN import HBRNN
import tensorflow as tf
from tensorflow.contrib import summary
import yaml
import os
import math

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution()


#--------------------------Configuration-----------------------#
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


#--------------------------Generator-----------------------#
skeleton_generator_train = SkeletonGenerator(batch_size=BATCH_SIZE, dataset_path=TRAINDATASET_PATH, skeleton_path=SKELETON_PATH, t=T, n_class=N_CLASS)
skeleton_generator_test = SkeletonGenerator(batch_size=VALIDATION_BATCH_SIZE, dataset_path=TESTDATASET_PATH, skeleton_path=SKELETON_PATH, t=T, n_class=N_CLASS)


#--------------------------Model-----------------------#
conf = cfg["HBRNN_CONFIG"]
teacher_model = HBRNN.Discriminator(conf)


#--------------------------Optimizer-----------------------#
optimizer = tf.train.AdamOptimizer(learning_rate=LR)



#---------------------------------------------Check points---------------------------------------------#
check_point = tf.train.Checkpoint(optimizer=optimizer, teacher_model=teacher_model, global_step=tf.train.get_or_create_global_step())

manager = tf.contrib.checkpoint.CheckpointManager(
    check_point, directory=CHECK_POINT_DIR, max_to_keep=20)
status = check_point.restore(manager.latest_checkpoint)

#---------------------------------------------Tensorboard configuration---------------------------------#
summary_writer = summary.create_file_writer(TENSORBOARD_DIR)

with summary_writer.as_default(), summary.always_record_summaries():
    for i in range(NUM_ITER):
        training_loss = 0
        training_acc = 0
        validation_loss = 0
        validation_acc = 0
        skeleton_generator_train.suffle()
        for h in range(skeleton_generator_train.num_batch):
            x, t = skeleton_generator_train.getFlow(h)

            with tf.GradientTape() as tape:
                y = teacher_model.berkeley(x)
                # logits = tf.reduce_sum(tf.reduce_mean([y0, y1], axis=0), axis=1)
                y = tf.reduce_mean(y, axis=1)
                y_prob = tf.nn.softmax(y)
                loss = tf.losses.softmax_cross_entropy(logits=y, onehot_labels=t)
                #print(loss)

                training_acc += tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(y_prob, 1), tf.argmax(t, 1)), dtype=tf.float32))
                variables = teacher_model.trainable_variables
                grads = tape.gradient(loss, variables)

                clipped_grads = [tf.clip_by_norm(g, 3.) for g in grads]
                optimizer.apply_gradients(zip(clipped_grads, variables), global_step=tf.train.get_or_create_global_step())
            training_loss += loss

        for h in range(skeleton_generator_test.num_batch):
            x, t = skeleton_generator_test.getFlow(h)
            y = teacher_model.berkeley(x)
            # logits = tf.reduce_sum(tf.reduce_mean([y0, y1], axis=0), axis=1)
            y = tf.reduce_mean(y, axis=1)
            y_prob = tf.nn.softmax(y)

            validation_loss += tf.losses.softmax_cross_entropy(logits=y, onehot_labels=t)

            validation_acc += tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(y_prob, 1), tf.argmax(t, 1)), dtype=tf.float32))



        training_loss = training_loss / skeleton_generator_train.num_batch
        training_acc = training_acc / skeleton_generator_train.num_batch
        validation_loss = validation_loss / skeleton_generator_test.num_batch
        validation_acc = validation_acc / skeleton_generator_test.num_batch
        print("Loss =  %f, accuracy = %f, validation loss = %f, validation accuracy = %f" % (training_loss, training_acc, validation_loss, validation_acc))
        summary.scalar("validation_loss", validation_loss, step=tf.train.get_or_create_global_step())
        summary.scalar("training_loss", training_loss, step=tf.train.get_or_create_global_step())

        summary.scalar("training_acc", training_acc, step=tf.train.get_or_create_global_step())
        summary.scalar("validation_acc", validation_acc, step=tf.train.get_or_create_global_step())

        if validation_loss < THRESHOLD_LOSS or i == 1:
            manager.save()
            THRESHOLD_LOSS = validation_loss
            PREV_BEST = i

        if (i+1) % 25 == 0:
            optimizer._lr = optimizer._lr * math.sqrt(0.5)


        if (i - PREV_BEST) > 15:
            break

