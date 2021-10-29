from .layers import *
from tensorflow_addons.layers import SpectralNormalization

class Discriminator(Model):
    def __init__(self, channel=64, n_layers=5):
        super().__init__()
        self.channel = channel
        self.n_layers = n_layers

        models = []
        models += [
            ReflectionPadding2D((1, 1)),
            SpectralNormalization(kl.Conv2D(channel, 4, strides=(2, 2))),
            kl.LeakyReLU(0.2)
        ]

        for i in range(1, n_layers - 1):
            models += [
                ReflectionPadding2D((1, 1)),
                SpectralNormalization(kl.Conv2D(channel * 2, 4, strides=(2, 2))),
                kl.LeakyReLU(0.2)
            ]
            channel = channel * 2

        models += [
            ReflectionPadding2D((1, 1)),
            SpectralNormalization(kl.Conv2D(channel * 2, 4, strides=(1, 1))),
            kl.LeakyReLU(0.2)
        ]
        channel = channel * 2

        self.models = models

        self.gap = CAM(CAM.Mode.GAP, sn=True)
        self.gmp = CAM(CAM.Mode.GMP, sn=True)

        self.conv_1x1 = kl.Conv2D(channel, 1, strides=(1, 1))
        self.leaky_relu = kl.LeakyReLU(0.2)

        self.reduce_sum = kl.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True))

        self.pad = ReflectionPadding2D((1, 1))
        self.conv = SpectralNormalization(kl.Conv2D(1, 4, strides=(1, 1)))

    def call(self, x):
        for f in self.models:
            x = f(x)
        
        x_gap, gap_logit = self.gap(x)
        x_gmp, gmp_logit = self.gmp(x)
        cam_logit = kl.Concatenate(axis=-1)([gap_logit, gmp_logit])
        x = kl.Concatenate(axis=-1)([x_gap, x_gmp])

        x = self.conv_1x1(x)
        x = self.leaky_relu(x)

        heatmap = self.reduce_sum(x)

        x = self.pad(x)
        x = self.conv(x)

        return x, cam_logit, heatmap
