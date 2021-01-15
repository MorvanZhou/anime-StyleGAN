import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import logging
import sys
import numpy as np


def set_soft_gpu(soft_gpu):
    if soft_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


def save_gan(model, path):
    global z1, z2
    n = 7
    if "z1" not in globals():
        z1 = np.random.normal(0, 1, size=(n, 1, model.latent_dim))
    if "z2" not in globals():
        z2 = np.random.normal(0, 1, size=(n, 1, model.latent_dim))
    n_z1 = 3
    assert n_z1 < model.n_style_block - 1
    noise = np.random.normal(0, 1, [len(z1), model.img_shape[0], model.img_shape[1]])
    inputs = [
        np.ones((len(z1)*n, 1)),
        np.concatenate(
            (z1.repeat(n, axis=0).repeat(n_z1, axis=1),
             np.repeat(np.concatenate([z2 for _ in range(n)], axis=0), model.n_style_block - n_z1, axis=1)),
            axis=1
        ),
        noise.repeat(n, axis=0),
    ]
    z1_inputs = [np.ones((len(z1), 1)), z1.repeat(model.n_style_block, axis=1), noise]
    z2_inputs = [np.ones((len(z2), 1)), z2.repeat(model.n_style_block, axis=1), noise]

    imgs = model.predict(inputs)
    z1_imgs = model.predict(z1_inputs)
    z2_imgs = model.predict(z2_inputs)
    imgs = np.concatenate([z2_imgs, imgs], axis=0)
    rest_imgs = np.concatenate([np.ones([1, model.img_shape[0], model.img_shape[1], model.img_shape[2]], dtype=np.float32), z1_imgs], axis=0)
    for i in range(len(rest_imgs)):
        imgs = np.concatenate([imgs[:i * (n+1)], rest_imgs[i:i + 1], imgs[i * (n+1):]], axis=0)
    imgs = (imgs + 1) / 2

    plt.clf()
    nc, nr = n+1, n+1
    plt.figure(0, (nc*2, nr*2))
    for c in range(nc):
        for r in range(nr):
            i = r * nc + c
            plt.subplot(nr, nc, i + 1)
            plt.imshow(imgs[i])
            plt.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)


def get_logger(date_str):
    log_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = "visual/{}/train.log".format(date_str)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(log_fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    return logger


class InstanceNormalization(keras.layers.Layer):
    def __init__(self, axis=(1, 2), epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.axis = axis
        self.beta, self.gamma = None, None

    def build(self, input_shape):
        shape = [1 for _ in range(len(input_shape))]
        shape[-1] = input_shape[-1]
        self.gamma = self.add_weight(
            name='gamma',
            shape=shape,
            initializer='ones')

        self.beta = self.add_weight(
            name='beta',
            shape=shape,
            initializer='zeros')

    def call(self, x, *args, **kwargs):
        mean = tf.math.reduce_mean(x, axis=self.axis, keepdims=True)
        x -= mean
        variance = tf.reduce_mean(tf.math.square(x), axis=self.axis, keepdims=True)
        x *= tf.math.rsqrt(variance + self.epsilon)
        return x * self.gamma + self.beta