from tensorflow import keras as K
import tensorflow as tf

class HBRNN:
    class Discriminator(K.models.Model):

        def __init__(self, conf):
            super().__init__()

            # 1st layer
            self.rnn1_1 = K.layers.Bidirectional(
                K.layers.SimpleRNN(units=conf["RNN_UNITS"][0], recurrent_dropout=conf["DROPOUT"][1], return_sequences=True, name="rnn1_1"), merge_mode="concat")
            self.rnn1_2 = K.layers.Bidirectional(
                K.layers.SimpleRNN(units=conf["RNN_UNITS"][0], recurrent_dropout=conf["DROPOUT"][1], return_sequences=True, name="rnn1_2"), merge_mode="concat")
            self.rnn1_3 = K.layers.Bidirectional(
                K.layers.SimpleRNN(units=conf["RNN_UNITS"][0], recurrent_dropout=conf["DROPOUT"][1], return_sequences=True, name="rnn1_3"), merge_mode="concat")
            self.rnn1_4 = K.layers.Bidirectional(
                K.layers.SimpleRNN(units=conf["RNN_UNITS"][0], recurrent_dropout=conf["DROPOUT"][1], return_sequences=True, name="rnn1_4"), merge_mode="concat")
            self.rnn1_5 = K.layers.Bidirectional(
                K.layers.SimpleRNN(units=conf["RNN_UNITS"][0], recurrent_dropout=conf["DROPOUT"][1], return_sequences=True, name="rnn1_5"), merge_mode="concat")

            # 2nd layer
            self.rnn2_1 = K.layers.Bidirectional(
                K.layers.SimpleRNN(units=conf["RNN_UNITS"][1], recurrent_dropout=conf["DROPOUT"][1], return_sequences=True, name="rnn2_1"), merge_mode="concat")
            self.rnn2_2 = K.layers.Bidirectional(
                K.layers.SimpleRNN(units=conf["RNN_UNITS"][1], recurrent_dropout=conf["DROPOUT"][1], return_sequences=True, name="rnn2_2"), merge_mode="concat")
            self.rnn2_3 = K.layers.Bidirectional(
                K.layers.SimpleRNN(units=conf["RNN_UNITS"][1], recurrent_dropout=conf["DROPOUT"][1], return_sequences=True, name="rnn2_3"), merge_mode="concat")
            self.rnn2_4 = K.layers.Bidirectional(
                K.layers.SimpleRNN(units=conf["RNN_UNITS"][1], recurrent_dropout=conf["DROPOUT"][1], return_sequences=True, name="rnn2_4"), merge_mode="concat")

            # 3rd layer
            self.rnn3_1 = K.layers.Bidirectional(
                K.layers.SimpleRNN(units=conf["RNN_UNITS"][2], recurrent_dropout=conf["DROPOUT"][1], return_sequences=True, name="rnn3_1"), merge_mode="concat")
            self.rnn3_2 = K.layers.Bidirectional(
                K.layers.SimpleRNN(units=conf["RNN_UNITS"][2], recurrent_dropout=conf["DROPOUT"][1], return_sequences=True, name="rnn3_2"), merge_mode="concat")

            # 4th layer
            self.rnn4_1 = K.layers.Bidirectional(K.layers.CuDNNLSTM(units=conf["RNN_UNITS"][3], return_sequences=True, name="rnn4_1"), merge_mode="concat")

            self.concatenate = K.layers.Concatenate(axis=-1)

            self.dense = K.layers.TimeDistributed(K.layers.Dense(units=conf["N_CLASS"], activation="linear"))
            #self.dense = K.layers.Dense(units=conf["N_CLASS"], activation="linear")

            #flatten1 for trunk, and left and right arms
            #flatten2 for left and right arms
            self.flatten1 = K.layers.TimeDistributed(K.layers.Flatten())
            self.flatten2 = K.layers.TimeDistributed(K.layers.Flatten())

            self.dropout = K.layers.Dropout(conf["DROPOUT"][0])



        def berkeley(self, inputs, training=None, mask=None):
            #splits the input
            """"
                  array size (21, 2)
                  [3:21] = trunk
                  [39:57] = left_arm
                  [21:39] = right_arm
                  [57:75] = left_leg
                  [75:93] = right_leg
            """""
            x0 = inputs[:, :, 3:21]
            x1 = inputs[:, :, 39:57]
            x2 = inputs[:, :, 21:39]
            x3 = inputs[:, :, 57:75]
            x4 = inputs[:, :, 75:93]

            # 1st layer
            x1_1 = self.rnn1_1(self.dropout(x0))
            x1_2 = self.rnn1_2(self.dropout(x1))
            x1_3 = self.rnn1_3(self.dropout(x2))
            x1_4 = self.rnn1_4(self.dropout(x3))
            x1_5 = self.rnn1_5(self.dropout(x4))

            # 2nd layer
            x2_1 = self.rnn2_1(self.dropout(self.concatenate([x1_1, x1_3])))
            x2_2 = self.rnn2_2(self.dropout(self.concatenate([x1_2, x1_3])))
            x2_3 = self.rnn2_3(self.dropout(self.concatenate([x1_4, x1_3])))
            x2_4 = self.rnn2_4(self.dropout(self.concatenate([x1_5, x1_3])))

            # 3rd layer
            x3_1 = self.rnn3_1(self.dropout(self.concatenate([x2_1, x2_2])))
            x3_2 = self.rnn3_2(self.dropout(self.concatenate([x2_3, x2_4])))

            # 4th layer
            x4_1 = self.rnn4_1(self.dropout(self.concatenate([x3_1, x3_2])))

            # 5th layer
            x = self.dense(x4_1)

            return x
