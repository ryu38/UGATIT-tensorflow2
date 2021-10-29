import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow_addons.layers import InstanceNormalization
from enum import Enum


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [kl.InputSpec(ndim=4)]
        super().__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')


class CAM(Layer):

    class Mode(Enum):
        GAP = 1
        GMP = 2

    def __init__(self, mode, sn=False, **kwargs):
        assert(type(mode) is self.Mode)
        self.mode = mode
        self.sn = sn
        self.spectral_norm = CAMSpectralNorm()
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            name='cam_w',
            shape=(input_shape[-1], 1),
            initializer=tf.keras.initializers.glorot_uniform(),
            trainable=True
        )

    def call(self, x, mark=None):

        if self.mode == self.Mode.GAP:
            x_cam = tf.reduce_mean(x, [1, 2])
        else:
            x_cam = tf.reduce_max(x, [1, 2])

        if self.sn:
            w = self.spectral_norm(self.w)
        else:
            w = self.w

        cam_logit = tf.matmul(x_cam, w)
        w = tf.reshape(w, [self.w.shape[0]])
        x = tf.multiply(x, w)
        return x, cam_logit

    def compute_output_shape(self, s):
        return tuple(s), (s[0], 1)


class CAMSpectralNorm(Layer):
    def __init__(self, iteration=1, **kwargs):
        self.iteration = iteration
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.u = self.add_weight(
            name='camsn_u',
            shape=(1, input_shape[-1]),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=False
        )

    def call(self, w, mark=None):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u_hat = self.u
        v_hat = None
        for i in range(self.iteration):
            """
            power iteration
            Usually iteration = 1 will be enough
            """
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        with tf.control_dependencies([self.u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)

        return w_norm

    def compute_output_shape(self, s):
        return tuple(s)


class AdaLIN(Layer):

    def __init__(self, eps=1e-5, **kwargs):
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.rho = self.add_weight(
            name='adalin_rho',
            shape=(input_shape[0][-1]),
            initializer=tf.keras.initializers.constant(1.0),
            constraint=tf.keras.constraints.min_max_norm(min_value=0.0, max_value=1.0),
            trainable=True
        )

    def call(self, inputs, mark=None):
        assert(len(inputs) == 3)
        x, gamma, beta = inputs
        gamma = tf.reshape(gamma, (-1, 1, 1, gamma.shape[-1]))
        beta = tf.reshape(beta, (-1, 1, 1, beta.shape[-1]))

        ins_mean = tf.reduce_mean(x, [1, 2], keepdims=True)
        ins_var = tf.math.reduce_variance(x, [1, 2], keepdims=True)
        x_ins = (x - ins_mean) / tf.sqrt(ins_var + self.eps)

        ln_mean = tf.reduce_mean(x, [1, 2, 3], keepdims=True)
        ln_var = tf.math.reduce_variance(x, [1, 2, 3], keepdims=True)
        x_ln = (x - ln_mean) / tf.sqrt(ln_var + self.eps)

        x = self.rho * x_ins + (1 - self.rho) * x_ln
        x = gamma * x + beta
        return x

    def compute_output_shape(self, s):
        return s[0]


class LIN(Layer):

    def __init__(self, eps=1e-5, **kwargs):
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.rho = self.add_weight(
            name='lin_rho',
            shape=(input_shape[-1]),
            initializer=tf.keras.initializers.constant(1.0),
            constraint=tf.keras.constraints.min_max_norm(min_value=0.0, max_value=1.0),
            trainable=True
        )
        self.gamma = self.add_weight(
            name='lin_gamma',
            shape=(input_shape[-1]),
            initializer=tf.keras.initializers.constant(1.0),
            trainable=True
        )
        self.beta = self.add_weight(
            name='lin_beta',
            shape=(input_shape[-1]),
            initializer=tf.keras.initializers.constant(0.0),
            trainable=True
        )

    def call(self, x, mark=None):
        ins_mean = tf.reduce_mean(x, [1, 2], keepdims=True)
        ins_var = tf.math.reduce_variance(x, [1, 2], keepdims=True)
        x_ins = (x - ins_mean) / tf.sqrt(ins_var + self.eps)

        ln_mean = tf.reduce_mean(x, [1, 2, 3], keepdims=True)
        ln_var = tf.math.reduce_variance(x, [1, 2, 3], keepdims=True)
        x_ln = (x - ln_mean) / tf.sqrt(ln_var + self.eps)

        x = self.rho * x_ins + (1 - self.rho) * x_ln
        x = self.gamma * x + self.beta
        return x

    def compute_output_shape(self, s):
        return tuple(s)


class ResNet(Model):
    def __init__(self, channel, use_bias=True):
        super().__init__()
        self.ref_pad0 = ReflectionPadding2D((1, 1))
        self.conv0 = kl.Conv2D(channel, 3, use_bias=use_bias)
        self.in_norm0 = InstanceNormalization()
        self.relu = kl.ReLU()

        self.ref_pad1 = ReflectionPadding2D((1, 1))
        self.conv1 = kl.Conv2D(channel, 3, use_bias=use_bias)
        self.in_norm1 = InstanceNormalization()

        self.add = kl.Add()

    def call(self, x):
        y = self.ref_pad0(x)
        y = self.conv0(y)
        y = self.in_norm0(y)
        y = self.relu(y)
        y = self.ref_pad1(y)
        y = self.conv1(y)
        y = self.in_norm1(y)
        return self.add([x, y])


class ResNetAdaLIN(Model):
    def __init__(self, channel, use_bias=True):
        super().__init__()
        self.ref_pad0 = ReflectionPadding2D((1, 1))
        self.conv0 = kl.Conv2D(channel, 3, use_bias=use_bias)
        self.adalin0 = AdaLIN()
        self.relu = kl.ReLU()

        self.ref_pad1 = ReflectionPadding2D((1, 1))
        self.conv1 = kl.Conv2D(channel, 3, use_bias=use_bias)
        self.adalin1 = AdaLIN()

        self.add = kl.Add()

    def call(self, inputs):
        assert(len(inputs) == 3)
        x, gamma, beta = inputs

        y = self.ref_pad0(x)
        y = self.conv0(y)
        y = self.adalin0([y, gamma, beta])
        y = self.relu(y)
        y = self.ref_pad1(y)
        y = self.conv1(y)
        y = self.adalin1([y, gamma, beta])
        return self.add([x, y])


class FCGammaBeta(Model):
    def __init__(self, channel, use_bias=True, light=False):
        super().__init__()
        self.light = light
        
        self.fc = [
            kl.Dense(channel, use_bias=use_bias),
            kl.ReLU(),
            kl.Dense(channel, use_bias=use_bias),
            kl.ReLU()
        ]
        self.gamma = kl.Dense(channel, use_bias=use_bias)
        self.beta = kl.Dense(channel, use_bias=use_bias)

    def call(self, x):
        if self.light:
            x = tf.reduce_mean(x, [1, 2])

        for f in self.fc:
            x = f(x)
        return self.gamma(x), self.beta(x)
