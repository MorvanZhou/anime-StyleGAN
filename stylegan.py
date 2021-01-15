import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import *
from utils import InstanceNormalization
import tensorflow.keras.initializers as initer


class AdaNorm(Layer):
    def __init__(self, axis=(1, 2), epsilon=1e-6):
        super().__init__()
        # NHWC
        self.axis = axis
        self.epsilon = epsilon

    def call(self, x, **kwargs):
        mean = tf.math.reduce_mean(x, axis=self.axis, keepdims=True)
        x -= mean
        variance = tf.reduce_mean(tf.math.square(x), axis=self.axis, keepdims=True)
        x *= tf.math.rsqrt(variance + self.epsilon)
        return x


class AdaMod(Layer):
    def __init__(self):
        super().__init__()
        self.y = None

    def call(self, inputs, **kwargs):
        x, w = inputs
        y = self.y(w)
        o = (y[:, 0] + 1) * x + y[:, 1]
        return o

    def build(self, input_shape):
        x_shape, w_shape = input_shape
        self.y = keras.Sequential([
            Dense(x_shape[-1]*2, input_shape=w_shape[1:], kernel_initializer=initer.HeNormal()),
            Reshape([2, 1, 1, -1]),
        ])  # [2, h, w, c] per feature map


class AddNoise(Layer):
    def __init__(self):
        super().__init__()
        self.s = None
        self.x_shape = None

    def call(self, inputs, **kwargs):
        x, noise = inputs
        noise_ = noise[:, :self.x_shape[1], :self.x_shape[2], :]
        return self.s * noise_ + x

    def build(self, input_shape):
        self.x_shape, _ = input_shape
        self.s = self.add_weight(name="noise_scale", shape=[1, 1, self.x_shape[-1]],
                                 initializer=initer.RandomNormal(0, 0.05))


class Map(Layer):
    def __init__(self, size, num_layers, norm=None):
        super().__init__()
        self.size = size
        self.num_layers = num_layers
        self.norm_name = norm
        self.f = None

    def call(self, inputs, **kwargs):
        w = self.f(inputs)
        return w

    def build(self, input_shape):
        self.f = keras.Sequential()
        for i in range(self.num_layers):
            if i == 0:
                self.f.add(Dense(self.size, input_shape=input_shape[1:], kernel_initializer=initer.HeNormal()))
                continue
            self.f.add(LeakyReLU(0.2))
            if self.norm_name is not None:
                if self.norm_name.lower() == "batch":
                    self.f.add(BatchNormalization())
                elif self.norm_name.lower() == "instance":
                    self.f.add(InstanceNormalization((1,)))       # instance norm increases model collapse
            self.f.add(Dense(self.size, kernel_initializer=initer.HeNormal()))


class Style(Layer):
    def __init__(self, filters, upsampling=True):
        super().__init__()
        self.filters = filters
        self.upsampling = upsampling
        self.ada_mod, self.ada_norm, self.add_noise, self.up, self.conv, self.conv_expend = None, None, None, None, None, None

    def call(self, inputs, **kwargs):
        x, w, noise = inputs
        # x = self.conv_expend(x)     #TODO: may help for styling
        x = self.ada_mod((x, w))
        if self.up is not None:
            x = self.up(x)
        x = self.conv(x)
        x = LeakyReLU(0.2)(x)
        x = self.add_noise((x, noise))
        x = self.ada_norm(x)
        return x

    def build(self, input_shape):
        self.ada_mod = AdaMod()
        self.ada_norm = AdaNorm()
        if self.upsampling:
            self.up = UpSampling2D((2, 2), interpolation="bilinear")
        self.add_noise = AddNoise()
        # self.conv_expend = Conv2D(self.filters*2, 1, 1, kernel_initializer=initer.HeNormal())
        self.conv = Conv2D(self.filters, 3, 1, "same", kernel_initializer=initer.HeNormal())


def get_generator(latent_dim, img_shape):
    n_style_block = 0
    const_size = _size = 4
    while _size <= img_shape[1]:
        n_style_block += 1
        _size *= 2

    z = keras.Input((n_style_block, latent_dim,), name="z")
    noise_ = keras.Input((img_shape[0], img_shape[1]), name="noise")
    ones = keras.Input((1,), name="ones")

    w = Map(size=128, num_layers=5)(z)
    noise = tf.expand_dims(noise_, axis=-1)
    const = keras.Sequential([
        Dense(const_size * const_size * 200, use_bias=False, name="const", kernel_initializer=initer.HeNormal()),
        Reshape((const_size, const_size, -1)),
    ], name="const")(ones)

    x = AddNoise()((const, noise))
    x = AdaNorm()(x)
    for i in range(n_style_block):
        x = Style(200, upsampling=False if i == 0 else True)((x, w[:, i], noise))
    o = Conv2D(img_shape[-1], 5, 1, "same", activation=keras.activations.tanh)(x)

    g = keras.Model([ones, z, noise_], o, name="generator")
    return g, n_style_block


class StyleGAN(keras.Model):
    def __init__(self, img_shape, latent_dim,
                 summary_writer=None, lr=0.0002, beta1=0.5, beta2=0.99, lambda_=10, wgan=2):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.lambda_ = lambda_
        self.wgan = wgan

        self.g, self.n_style_block = get_generator(latent_dim, img_shape)
        self.g.summary()
        self.d = self._get_discriminator()
        self.d.summary()

        self.opt = keras.optimizers.Adam(lr, beta_1=beta1, beta_2=beta2)

        self.summary_writer = summary_writer
        self._train_step = 0

    def call(self, inputs, training=None, mask=None):
        if isinstance(inputs[0], np.ndarray):
            inputs = [tf.convert_to_tensor(i) for i in inputs]
        return self.g.call(inputs, training=training)

    def _get_discriminator(self):
        def add_block(filters, do_norm=True):
            model.add(Conv2D(filters, 4, strides=2, padding='same'))
            if do_norm: model.add(InstanceNormalization())
            model.add(LeakyReLU(alpha=0.2))

        model = keras.Sequential([Input(self.img_shape)], name="d")
        # [n, 64, 64, 3]
        add_block(32, do_norm=False)   # -> 32^2
        add_block(64)                   # -> 16^2
        add_block(128)                  # -> 8^2
        add_block(256)                  # -> 4^2
        # add_block(512)                  # 2^2
        model.add(Flatten())
        # model.add(GlobalAveragePooling2D())
        model.add(Dense(256))
        model.add(Dense(1))
        return model

    # gradient penalty
    def gp(self, real_img, fake_img):
        e = tf.random.uniform((len(real_img), 1, 1, 1), 0, 1)
        noise_img = e * real_img + (1. - e) * fake_img  # extend distribution space
        with tf.GradientTape() as tape:
            tape.watch(noise_img)
            o = self.d(noise_img)
        g = tape.gradient(o, noise_img)  # image gradients
        g_norm2 = tf.sqrt(tf.reduce_sum(tf.square(g), axis=[1, 2, 3]))  # norm2 penalty
        gp = tf.square(g_norm2 - 1.)
        return tf.reduce_mean(gp)

    @staticmethod
    def w_distance(real, fake):
        # the distance of two data distributions
        return tf.reduce_mean(real) - tf.reduce_mean(fake)

    def get_inputs(self, n):
        if np.random.rand() < 0.5:
            available_z = [tf.random.normal((n, 1, self.latent_dim)) for _ in range(2)]
            z = tf.concat(
                [available_z[np.random.randint(0, len(available_z))] for _ in range(self.n_style_block)], axis=1)
        else:
            z = tf.repeat(tf.random.normal((n, 1, self.latent_dim)), self.n_style_block, axis=1)

        noise = tf.random.normal((n, self.img_shape[0], self.img_shape[1]))
        return [tf.ones((n, 1)), z, noise]

    def train_d(self, img):
        n = len(img)

        with tf.GradientTape() as tape:
            gimg = self.call(self.get_inputs(n), training=False)
            gp = self.gp(img, gimg)
            pred_fake = self.d.call(gimg, training=True)
            pred_real = self.d.call(img, training=True)
            w_distance = -self.w_distance(pred_real, pred_fake)  # maximize W distance
            gp_loss = self.lambda_ * gp
            loss = gp_loss + w_distance
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))

        if self._train_step % 300 == 0 and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("d/w_distance", w_distance, step=self._train_step)
                tf.summary.scalar("d/gp", gp, step=self._train_step)
        return gp, w_distance

    def train_g(self, n):
        with tf.GradientTape() as tape:
            gimg = self.call(self.get_inputs(n), training=True)
            pred_fake = self.d.call(gimg, training=False)
            w_distance = tf.reduce_mean(-pred_fake)  # minimize W distance
        grads = tape.gradient(w_distance, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))

        if self._train_step % 300 == 0 and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("g/w_distance", w_distance, step=self._train_step)
                if self._train_step % 1000 == 0:
                    tf.summary.image("gimg", (gimg + 1) / 2, max_outputs=5, step=self._train_step)

        return w_distance

    def step(self, img):
        gw = self.train_g(len(img) * 2)
        for _ in range(self.wgan):
            dgp, dw = self.train_d(img)
        self._train_step += 1
        return gw, dgp, dw