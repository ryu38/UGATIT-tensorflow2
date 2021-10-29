from .layers import *

class Generator(Model):
    def __init__(self, channel=64, n_res=6, light=False):
        super().__init__()
        self.channel = channel
        self.n_res = n_res
        self.light = light

        encoder = []
        encoder += [
            ReflectionPadding2D((3, 3)),
            kl.Conv2D(channel, 7, strides=(1, 1)),
            InstanceNormalization(),
            kl.ReLU()
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            encoder += [
                ReflectionPadding2D((1, 1)),
                kl.Conv2D(channel * 2, 3, strides=(2, 2)),
                InstanceNormalization(),
                kl.ReLU()
            ]
            channel = channel * 2
        
        for i in range(n_res):
            encoder += [ResNet(channel)]

        self.encoder = encoder

        self.gap = CAM(CAM.Mode.GAP)
        self.gmp = CAM(CAM.Mode.GMP)

        self.conv_1x1 = kl.Conv2D(channel, 1, strides=(1, 1))
        self.relu = kl.ReLU()

        self.reduce_sum = kl.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True))

        self.fcgb = FCGammaBeta(channel, light=self.light)

        arbs = []
        for i in range(n_res):
            arbs += [ResNetAdaLIN(channel)]
        self.arbs = arbs

        upblock = []
        for i in range(n_downsampling):
            upblock += [
                kl.UpSampling2D((2, 2)),
                ReflectionPadding2D((1, 1)),
                kl.Conv2D(channel // 2, 3),
                LIN(),
                kl.ReLU()
            ]
            channel = channel // 2

        upblock += [
            ReflectionPadding2D((3, 3)),
            kl.Conv2D(3, 7),
            kl.Lambda(lambda x: tf.tanh(x))
        ]

        self.upblock = upblock

    def call(self, x):
        for f in self.encoder:
            x = f(x)

        x_gap, gap_logit = self.gap(x)
        x_gmp, gmp_logit = self.gmp(x)
        cam_logit = kl.Concatenate(axis=-1)([gap_logit, gmp_logit])
        x = kl.Concatenate(axis=-1)([x_gap, x_gmp])

        x = self.conv_1x1(x)
        x = self.relu(x)

        heatmap = self.reduce_sum(x)

        gamma, beta = self.fcgb(kl.Flatten()(x))

        for f in self.arbs:
            x = f([x, gamma, beta])

        for f in self.upblock:
            x = f(x)

        return x, cam_logit, heatmap
