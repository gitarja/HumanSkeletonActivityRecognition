import tensorflow as tf
from tensorflow import keras as K


class CFGRNN(K.models.Model):

    def __init__(self, conf, average=False):
        super().__init__()

        # body parts-variable
        '''
        LH = Left Hand
        RH = Right Hand
        LF = Left Food
        RF = Right Food
        T = Trunk
        '''

        '''
        mF = Move Fast
        mS = Move Slow
        n = Not
        '''
        # self.lh = K.layers.TimeDistributed(
        #     K.layers.Dense(units=conf["terminal"]["units"], activation="relu", name="lh"))
        # self.rh = K.layers.TimeDistributed(
        #     K.layers.Dense(units=conf["terminal"]["units"], activation="relu", name="rh"))
        # self.lf = K.layers.TimeDistributed(
        #     K.layers.Dense(units=conf["terminal"]["units"], activation="relu", name="lf"))
        # self.rf = K.layers.TimeDistributed(
        #     K.layers.Dense(units=conf["terminal"]["units"], activation="relu", name="rf"))
        # self.t = K.layers.TimeDistributed(K.layers.Dense(units=conf["terminal"]["units"], activation="relu", name="t"))

        self.lh = K.layers.Bidirectional(
            K.layers.CuDNNLSTM(units=conf["terminal"]["units"], return_sequences=True, name="lh"),
            merge_mode="ave")
        self.rh = K.layers.Bidirectional(
            K.layers.CuDNNLSTM(units=conf["terminal"]["units"], return_sequences=True, name="rh"),
            merge_mode="ave")
        self.lf = K.layers.Bidirectional(
            K.layers.CuDNNLSTM(units=conf["terminal"]["units"], return_sequences=True, name="lf"),
            merge_mode="ave")
        self.rf = K.layers.Bidirectional(
            K.layers.CuDNNLSTM(units=conf["terminal"]["units"], return_sequences=True, name="rf"),
            merge_mode="ave")
        self.t = K.layers.Bidirectional(
            K.layers.CuDNNLSTM(units=conf["terminal"]["units"], return_sequences=True, name="t"),
            merge_mode="ave")

        self.mF = K.layers.Bidirectional(
            K.layers.CuDNNLSTM(units=conf["un_operator"]["units"], return_sequences=True, name="mF"),
            merge_mode="concat")
        self.mS = K.layers.Bidirectional(
            K.layers.CuDNNLSTM(units=conf["un_operator"]["units"], return_sequences=True, name="mS"),
            merge_mode="concat")
        self.forward = K.layers.Bidirectional(
            K.layers.CuDNNLSTM(units=conf["un_operator"]["units"], return_sequences=True, name="forward"),
            merge_mode="concat")

        self.and_op = K.layers.Bidirectional(
            K.layers.CuDNNLSTM(units=conf["bin_operator"]["units"], return_sequences=True, name="and_op"),
            merge_mode="concat")
        self.touch_op = K.layers.Bidirectional(
            K.layers.CuDNNLSTM(units=conf["un_operator"]["units"], return_sequences=True, name="touch_op"),
            merge_mode="concat")
        self.or_op = K.layers.Bidirectional(
            K.layers.CuDNNLSTM(units=conf["bin_operator"]["units"], return_sequences=True, name="or_op"),
            merge_mode="concat")
        self.then_op = K.layers.Bidirectional(
            K.layers.CuDNNLSTM(units=conf["bin_operator"]["units"], return_sequences=True, name="then_op"),
            merge_mode="concat")
        self.with_op = K.layers.Bidirectional(
            K.layers.CuDNNLSTM(units=conf["bin_operator"]["units"], return_sequences=True, name="with_op"),
            merge_mode="concat")
        self.neg_op = K.layers.Bidirectional(
            K.layers.CuDNNLSTM(units=conf["un_operator"]["units"], return_sequences=True, name="neg_op"),
            merge_mode="concat")




        if average:
            self.e = K.layers.Dense(units=1, activation=None, name="equation")
            self.classifier = K.layers.Bidirectional(
                K.layers.CuDNNLSTM(units=conf["classifier"]["units"], return_sequences=False, name="classifier"),
                merge_mode="concat")

        else:
            self.classifier = K.layers.Bidirectional(
                K.layers.CuDNNLSTM(units=conf["classifier"]["units"], return_sequences=True, name="classifier"),
                merge_mode="concat")
            self.e = K.layers.TimeDistributed(K.layers.Dense(units=1, activation=None, name="equation"))

        self.concat = K.layers.Concatenate(axis=-1)

        self.dropout = K.layers.Dropout(conf["DROPOUT"][0])

    def action(self, inputs, action="still"):
        # splits the input
        """"
              array size (21, 2)
              [3:21] = trunk
              [39:57] = left_arm
              [21:39] = right_arm
              [57:75] = left_leg
              [75:93] = right_leg
        """""
        t = inputs[:, :, 4:22]
        lh = inputs[:, :, 40:58]
        rh = inputs[:, :, 22:40]
        lf = inputs[:, :, 58:76]
        rf = inputs[:, :, 75:94]
        if action == "jumping":
            E = self.jumping(lf, rf, t)
        elif action == "jumpingJ":
            E = self.jumpingJ(lf, lh, rf, rh, t)
        elif action == "waving":
            E = self.waving(lh, rh)
        elif action == "wavingR":
            E = self.wavingR(lh, rh)
        elif action == "clap":
            E = self.clap(lh, rh)
        elif action == "punching":
            E = self.punching(lh, rh)
        elif action == "bending":
            E = self.bending(lh, rh, t)
        elif action == "throwing":
            E = self.throwing(lh, rh)
        elif action == "sitDown":
            E = self.sitDown(lf, rf, t)
        elif action == "standUp":
            E = self.standUp(lf, rf, t)
        elif action == "sitDownTstandUp":
            E = self.sitDownTstandUP(lf, rf, t)

        return self.e(self.classifier(E))
        # return E

    def actionOpenpTrack(self, inputs, action="still"):
        """"
              array size (21, 2)
              [3:21] = trunk
              [39:57] = left_arm
              [21:39] = right_arm
              [57:75] = left_leg
              [75:93] = right_leg
        """""
        t = inputs[:, :, 0:9]
        lh = inputs[:, :, 9:18]
        rh = inputs[:, :, 18:27]
        lf = inputs[:, :, 27:36]
        rf = inputs[:, :, 36:45]


        if action == "sitDown":
            E = self.sitDown(lf, rf, t)
        elif action == "walk":
            E = self.walk(lf, rf)
        elif action == "run":
            E = self.run(lf, rf)
        elif action == "movingBox":
            E = self.movingBox(lf, lh, rf, rh)
        elif action == "none":
            E = self.none(lf, lh, rf, rh, t)

        return self.e(self.classifier(E))



    def jumping(self, lf, rf, t):
        lf = self.lf(lf)
        rf = self.rf(rf)
        t = self.t(t)

        E = self.AND(self.AND(self.mF(lf), self.mF(rf)), self.WITH(self.mS(t)))

        return E

    def jumpingJ(self, lf, lh, rf, rh, t):
        lf = self.lf(lf)
        lh = self.lh(lh)
        rf = self.rf(rf)
        rh = self.rh(rh)
        t = self.t(t)

        E = self.AND(self.AND(self.mF(lf), self.mF(rf)),
                     self.AND(self.AND(self.mF(lh), self.mF(rh)), self.WITH(self.mS(t))))

        return E

    def waving(self, lh, rh):
        lh = self.lh(lh)
        rh = self.rh(rh)

        E = self.AND(self.mF(lh), self.mF(rh))

        return E

    def wavingR(self, lh, rh):
        rh = self.rh(rh)
        lh = self.rh(lh)

        E = self.AND(self.mF(rh), self.NEG(self.mF(lh)))

        return E

    def clap(self, lh, rh):
        lh = self.lh(lh)
        rh = self.rh(rh)

        E = self.TOUCH(self.forward(lh), self.forward(rh))

        return E

    def punching(self, lh, rh):
        lh = self.lh(lh)
        rh = self.rh(rh)

        # E = self.AND(self.forward(lh), self.forward(rh))
        E = self.OR(self.THEN(self.forward(lh), self.forward(rh)), self.THEN(self.forward(rh), self.forward(lh)))

        return E

    def throwing(self, lh, rh):
        lh = self.lh(lh)
        rh = self.rh(rh)

        E = self.OR(self.forward(lh), self.forward(rh))

        return E

    def bending(self, lh, rh, t):
        lh = self.lh(lh)
        rh = self.rh(rh)
        t = self.t(t)

        E = self.AND(self.AND(self.forward(lh), self.forward(rh)), self.WITH(self.forward(t)))

        return E

    def sitDown(self, lf, rf, t):
        E = self.NEG(self.standUp(lf, rf, t))

        return E

    def standUp(self, lf, rf, t):
        lf = self.lf(lf)
        rf = self.rf(rf)
        t = self.t(t)

        E = self.AND(self.AND(self.mF(lf), self.mF(rf)), self.mF(t))

        return E

    def sitDownTstandUP(self, lf, rf, t):

        sitDown = self.sitDown(lf, rf, t)
        standUp = self.standUp(lf, rf, t)

        E = self.THEN(sitDown, standUp)

        return E

    def walk(self, lf, rf):
        lf = self.lf(lf)
        rf = self.rf(rf)

        E = self.AND(self.mS(lf), self.mS(rf))

        return E

    def run(self, lf, rf):
        lf = self.lf(lf)
        rf = self.rf(rf)

        E = self.AND(self.mF(lf), self.mF(rf))

        return E

    def movingBox(self, lf, lh, rf, rh):
        lh = self.lh(lh)
        rh = self.rh(rh)

        E = self.AND(self.AND(self.forward(rh), self.forward(lh)), self.walk(lf, rf))

        return E

    def none(self,  lf, lh, rf, rh, t):
        not_walk_run = self.AND(self.NEG(self.walk(lf, rf)), self.NEG(self.run(lf, rf)))
        not_sit_moveBox = self.AND(self.NEG(self.sitDown(lf, rf, t)), self.NEG(self.movingBox(lf, lh, rf, rh)))
        E = self.AND(not_walk_run, not_sit_moveBox)

        return E

    def classify(self, e):

        return self.classifier(e)

    def AND(self, x1, x2):
        return self.and_op(self.concat([x1, x2]))

    def OR(self, x1, x2):
        return self.or_op(self.concat([x1, x2]))

    def TOUCH(self, x1, x2):
        return self.touch_op(self.concat([x1, x2]))

    def THEN(self, x1, x2):
        return self.then_op(self.concat([x1, x2]))

    def WITH(self, x1):
        return self.with_op(x1)

    def NEG(self, x1):
        return self.neg_op(x1)

    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=True):
        y_jump = self.action(x, action="jumping")
        y_jumpJ = self.action(x, action="jumpingJ")
        y_wav = self.action(x, action="waving")
        y_wavR = self.action(x, action="wavingR")
        y_clap = self.action(x, action="clap")
        y_punch = self.action(x, action="punching")
        y_bend = self.action(x, action="bending")
        y_throw = self.action(x, action="throwing")
        y_sitdown = self.action(x, action="sitDown")
        y_standUp = self.action(x, action="standUp")
        y_sitTstand = self.action(x, action="sitDownTstandUp")

        y = tf.concat(
            [y_jump, y_jumpJ, y_bend, y_punch, y_wav, y_wavR, y_clap, y_throw, y_sitTstand, y_sitdown, y_standUp
             ], axis=-1)

        return y


    def predictOpenPTrack(self, x):
        y_walk = self.actionOpenpTrack(x, action="walk")
        y_run = self.actionOpenpTrack(x, action="run")
        y_movingBox = self.actionOpenpTrack(x, action="movingBox")
        y_sitdown = self.actionOpenpTrack(x, action="sitDown")
        y_none = self.actionOpenpTrack(x, action="none")


        y = tf.concat(
            [y_none, y_sitdown, y_run,  y_walk, y_movingBox
             ], axis=-1)

        return y
