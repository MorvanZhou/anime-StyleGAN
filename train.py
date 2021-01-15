import os
import time
from dataset import load_tfrecord
from stylegan import StyleGAN
import utils
import argparse
import datetime
import tensorflow as tf
import numpy as np

tf.random.set_seed(1)
np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int)
parser.add_argument("-e", "--epoch", dest="epoch", default=101, type=int)
parser.add_argument("--latent", dest="latent", default=100, type=int)
parser.add_argument("--soft_gpu", dest="soft_gpu", action="store_true", default=False)
parser.add_argument("-lr", "--learning_rate", dest="lr", default=0.0002, type=float)
parser.add_argument("-b1", "--beta1", dest="beta1", default=0., type=float)
parser.add_argument("-b2", "--beta2", dest="beta2", default=0.99, type=float)
parser.add_argument("-s", "--image_size", dest="image_size", default=64, type=int)
parser.add_argument("-w", "--wgan", help="number of time for training wgan discriminator per generator", default=2, type=int)
parser.add_argument("--lambda_", dest="lambda_", default=10, type=float)
parser.add_argument("--data_dir", dest="data_dir", default="./data")

args = parser.parse_args()

date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def train(gan, d):
    _dir = "visual/{}/model".format(date_str)
    checkpoint_path = _dir + "/cp-{epoch:04d}.ckpt"
    os.makedirs(_dir, exist_ok=True)
    t0 = time.time()
    for ep in range(args.epoch):
        for t, img in enumerate(d.ds):
            gw, dgp, dw = gan.step(img)
            if t % 1000 == 0:
                utils.save_gan(gan, "visual/%s/ep%03dt%06d.png" % (date_str, ep, t))
                t1 = time.time()
                logger.info("ep={:03d} t={:04d} | time={:05.1f} | gw={:.3f} | dw={:.3f} | dgp={:.3f}".format(
                    ep, t, t1-t0, gw.numpy(), dw.numpy(), dgp.numpy()))
                t0 = t1

        gan.save_weights(checkpoint_path.format(epoch=ep))


def init_logger(date_str, m):
    logger = utils.get_logger(date_str)
    logger.info(str(args))
    logger.info("model parameters: g={}, d={}".format(
        m.g.count_params(), m.d.count_params()))

    try:
        tf.keras.utils.plot_model(m.g, show_shapes=True, expand_nested=True, dpi=150,
                                  to_file="visual/{}/net_g.png".format(date_str))
        tf.keras.utils.plot_model(m.d, show_shapes=True, expand_nested=True, dpi=150,
                                  to_file="visual/{}/net_d.png".format(date_str))
    except Exception as e:
        print(e)
    return logger


if __name__ == "__main__":
    utils.set_soft_gpu(args.soft_gpu)

    summary_writer = tf.summary.create_file_writer('visual/{}'.format(date_str))
    d = load_tfrecord(args.batch_size//2, args.data_dir)
    m = StyleGAN(
        img_shape=(args.image_size, args.image_size, 3), latent_dim=args.latent, summary_writer=summary_writer,
        lr=args.lr, beta1=args.beta1, beta2=args.beta2, lambda_=args.lambda_, wgan=args.wgan)
    logger = init_logger(date_str, m)
    train(m, d)


