import tensorflow as tf
from tensorflow import keras as K


class CFGRNN(K.models.Model):

    def __init__(self, conf):
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
        self.lh = K.layers.Bidirectional(
            K.layers.SimpleRNN(units=conf["terminal"]["units"], return_sequences=True, name="LH"), merge_mode="ave")
        self.rh = K.layers.Bidirectional(
            K.layers.SimpleRNN(units=conf["terminal"]["units"], return_sequences=True, name="RH"), merge_mode="ave")
        self.lf = K.layers.Bidirectional(
            K.layers.SimpleRNN(units=conf["terminal"]["units"], return_sequences=True, name="LF"), merge_mode="ave")
        self.rf = K.layers.Bidirectional(
            K.layers.SimpleRNN(units=conf["terminal"]["units"], return_sequences=True, name="RF"), merge_mode="ave")
        self.t = K.layers.Bidirectional(
            K.layers.SimpleRNN(units=conf["terminal"]["units"], return_sequences=True, name="T"), merge_mode="ave")

        self.mF = K.layers.Bidirectional(
            K.layers.SimpleRNN(units=conf["un_operator"]["units"], return_sequences=True, name="mF"),
            merge_mode="ave")
        self.mS = K.layers.Bidirectional(
            K.layers.SimpleRNN(units=conf["un_operator"]["units"], return_sequences=True, name="mS"),
            merge_mode="ave")
        self.forward = K.layers.Bidirectional(
            K.layers.SimpleRNN(units=conf["un_operator"]["units"], return_sequences=True, name="forward"),
            merge_mode="ave")
        self.opp = K.layers.Bidirectional(
            K.layers.SimpleRNN(units=conf["un_operator"]["units"], return_sequences=True, name="opp"),
            merge_mode="ave")

        self.and_op = K.layers.Bidirectional(
            K.layers.SimpleRNN(units=conf["bin_operator"]["units"], return_sequences=True, name="and_op"),
            merge_mode="ave")
        self.touch_op = K.layers.Bidirectional(
            K.layers.SimpleRNN(units=conf["un_operator"]["units"], return_sequences=True, name="touch_op"),
            merge_mode="ave")
        self.or_op = K.layers.Bidirectional(
            K.layers.SimpleRNN(units=conf["bin_operator"]["units"], return_sequences=True, name="or_op"),
            merge_mode="ave")
        self.then_op = K.layers.Bidirectional(
            K.layers.SimpleRNN(units=conf["bin_operator"]["units"], return_sequences=True, name="then_op"),
            merge_mode="ave")
        self.with_op = K.layers.Bidirectional(
            K.layers.SimpleRNN(units=conf["bin_operator"]["units"], return_sequences=True, name="with_op"),
            merge_mode="ave")

        #self.e = K.layers.TimeDistributed(K.layers.Dense(units=1, activation=None, name="equation"))
        self.e = K.layers.Dense(units=1, activation=None, name="equation")

        #self.classifier = K.layers.TimeDistributed(K.layers.Dense(units=conf["N_CLASS"], activation=None, name="equation"))
        self.classifier =  K.layers.Bidirectional(
            K.layers.SimpleRNN(units=conf["classifier"]["units"], return_sequences=False, name="classifier"),
            merge_mode="ave")

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
        t = inputs[:, :, 3:21]
        lh = inputs[:, :, 39:57]
        rh = inputs[:, :, 21:39]
        lf = inputs[:, :, 57:75]
        rf = inputs[:, :, 75:93]
        if action == "jumping":
            E = self.jumping(lf, lh, rf, rh, t)
        elif action == "jumpingJ":
            E = self.jumpingJ(lf, lh, rf, rh, t)
        elif action == "waving":
            E = self.waving(lh, rh)
        elif action == "wavingR":
            E = self.wavingR(rh)
        elif action == "clap":
            E = self.clap(lh, rh)
        elif action == "punching":
            E = self.punching(lh, rh)
        elif action == "bending":
            E = self.bending(lh, rh, t)
        elif action == "throwing":
            E = self.throwing(lh, rh)
        elif action == "sitDown":
            E = self.sitDown(lh, rh, t)
        elif action == "standUp":
            E = self.standUp(lh, rh, t)
        elif action == "sitDownTstandUp":
            E = self.sitDownTstandUP(lh, rh, t)

        return self.e(self.classifier(E))
        #return E

    def jumping(self, lf, lh, rf, rh, t):
        lf = self.lf(lf)
        lh = self.lh(lh)
        rf = self.rf(rf)
        rh = self.rh(rh)
        t = self.t(t)

        E = self.AND(self.mF(lf), self.AND(self.mS(lh), self.WITH(self.mS(t)), self.mS(rh)), self.mF(rf))

        return E

    def jumpingJ(self, lf, lh, rf, rh, t):
        lf = self.lf(lf)
        lh = self.lh(lh)
        rf = self.rf(rf)
        rh = self.rh(rh)
        t = self.t(t)

        E = self.AND(self.mF(lf), self.AND(self.mF(lh), self.WITH(self.mS(t)), self.mF(rh)), self.mF(rf))

        return E

    def waving(self, lh, rh):
        lh = self.lh(lh)
        rh = self.rh(rh)

        E = self.AND(self.mF(lh), self.mF(rh))

        return E

    def wavingR(self, rh):
        rh = self.rh(rh)

        E = self.WITH(self.mF(rh))

        return E

    def clap(self, lh, rh):
        lh = self.lh(lh)
        rh = self.rh(rh)

        E = self.TOUCH(self.forward(lh), self.forward(rh))

        return E

    def punching(self, lh, rh):
        lh = self.lh(lh)
        rh = self.rh(rh)

        E = self.AND(self.forward(lh), self.forward(rh))

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

        E = self.WITH(self.AND(self.forward(lh), self.forward(rh)), self.forward(t))

        return E

    def sitDown(self, lh, rh, t):
        E = self.opp(self.standUp(lh, rh, t))

        return E

    def standUp(self, lh, rh, t):
        lh = self.lh(lh)
        rh = self.rh(rh)
        t = self.t(t)

        E = self.WITH(self.AND(self.mF(lh), self.mF(rh)), self.mF(t))

        return E

    def sitDownTstandUP(self, lh, rh, t):

        sitDown = self.sitDown(lh, rh, t)
        standUp = self.standUp(lh, rh, t)

        E = self.THEN(sitDown, standUp)

        return E

    def classify(self, e):

        return self.classifier(e)

    def AND(self, x1, x2, end=None):
        end = self.NONE(x1, end)
        return self.and_op(tf.reduce_mean([x1, x2, end], axis=0))

    def OR(self, x1, x2, end=None):
        end = self.NONE(x1, end)
        return self.or_op(tf.reduce_mean([x1, x2, end], axis=0))

    def TOUCH(self, x1, x2, end=None):
        end = self.NONE(x1, end)
        return self.touch_op(tf.reduce_mean([x1, x2, end], axis=0))

    def THEN(self, x1, x2, end=None):
        end = self.NONE(x1, end)
        return self.then_op(tf.reduce_mean([x1, x2, end], axis=0))

    def WITH(self, x1, end=None):
        end = self.NONE(x1, end)
        return self.with_op(tf.reduce_mean([x1, end], axis=0))


    def NONE(self,x, end):
        if end == None:
            return tf.zeros(x.shape)
        return end
