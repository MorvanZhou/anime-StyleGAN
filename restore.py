import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from stylegan import StyleGAN

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("-o", "--output_path", type=str, default="demo/stylemix.png")
parser.add_argument("-n", "--n_z1", type=int, default=1, help="number of z1 style blocks")
parser.add_argument("--z_seed", default=1, type=int, help="z latent random seed")
parser.add_argument("--n_seed", default=1, type=int, help="noise random seed")
parser.add_argument("--latent_dim", type=int, default=128)
parser.add_argument("-s", "--image_size", dest="image_size", default=64, type=int)

args = parser.parse_args()

IMG_SHAPE = (args.image_size, args.image_size, 3)


def generate(generator, n_style_block):
    np.random.seed(args.z_seed)
    z1 = np.random.normal(0, 1, size=(1, 1, args.latent_dim))
    z2 = np.random.normal(0, 1, size=(1, 1, args.latent_dim))

    np.random.seed(args.n_seed)
    noise = np.random.normal(0, 1, [len(z1), IMG_SHAPE[0], IMG_SHAPE[1]])

    n_z1 = args.n_z1
    inputs = [
        np.ones((len(z1), 1)),
        np.concatenate(
            (z1.repeat(n_z1, axis=1),
             np.repeat(z2, n_style_block - n_z1, axis=1)),
            axis=1
        ),
        noise,
    ]
    z1_inputs = [
        np.ones((len(z1), 1)),
        z1.repeat(n_style_block, axis=1),
        noise
    ]
    z2_inputs = [
        np.ones((len(z2), 1)),
        z2.repeat(n_style_block, axis=1),
        noise
    ]
    imgs = generator.predict(inputs)
    z1_img = generator.predict(z1_inputs)
    z2_img = generator.predict(z2_inputs)

    imgs = (imgs.squeeze(0) + 1) / 2
    z1_img = (z1_img.squeeze(0) + 1) / 2
    z2_img = (z2_img.squeeze(0) + 1) / 2

    plt.figure(0, (20, 6))
    plt.subplot(131)
    plt.imshow(z1_img)
    plt.title("style:~{}".format(n_z1))
    plt.axis("off")
    plt.subplot(132)
    plt.imshow(imgs)
    plt.title("mixed")
    plt.axis("off")
    plt.subplot(133)
    plt.title("style:{}~".format(n_z1))
    plt.imshow(z2_img)
    plt.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path)


if __name__ == "__main__":
    print(args)
    gan = StyleGAN(img_shape=IMG_SHAPE, latent_dim=args.latent_dim)
    gan.load_weights(args.model_path)
    generate(gan.g, gan.n_style_block)