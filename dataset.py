import matplotlib.pyplot as plt
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    values = value if isinstance(value, (list, tuple)) else [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _img_array_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value.ravel()))


def _bytes_img_process(img_str):
    imgs = tf.io.decode_jpeg(img_str)
    imgs = tf.image.resize(imgs, (64, 64))
    return imgs


class Anime:
    def __init__(self, batch_size, data_dir="data", sub_dir="images"):
        self.img_dir = os.path.join(data_dir, sub_dir)
        self.tfrecord_dir = os.path.join(data_dir, "tfrecord-anime-stylegan")
        self.batch_size = batch_size
        self.ds = None

    def _image_example(self, img):
        feature = {
            "img": _bytes_feature(img),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _parse_img(self, example_proto):
        feature = tf.io.parse_single_example(example_proto, features={
            "img": tf.io.FixedLenFeature([], tf.string)
        })
        imgs = _bytes_img_process(feature["img"])
        return tf.cast(imgs, tf.float32) / 255 * 2 - 1

    def load_tf_recoder(self):
        paths = [os.path.join(self.tfrecord_dir, p) for p in os.listdir(self.tfrecord_dir)]
        raw_img_ds = tf.data.TFRecordDataset(paths)
        self.ds = raw_img_ds.shuffle(1024).map(
            self._parse_img, num_parallel_calls=tf.data.experimental.AUTOTUNE,
        ).batch(
            self.batch_size, drop_remainder=True
        ).prefetch(
            tf.data.experimental.AUTOTUNE
        )

    def to_tf_recoder(self):
        fs = os.listdir(self.img_dir)
        n = len(fs)//4
        chunks = [fs[i:i + n] for i in range(0, len(fs), n)]
        for i, chunk in enumerate(chunks):
            path = os.path.join(self.tfrecord_dir, "{}.tfrecord".format(i))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print("parsing " + path)
            with tf.io.TFRecordWriter(path) as writer:
                for img_name in chunk:
                    try:
                        img = open(os.path.join(self.img_dir, img_name), "rb").read()
                    except Exception as e:
                        break
                    tf_example = self._image_example(img)
                    writer.write(tf_example.SerializeToString())


def show_sample(data_dir):
    d = load_tfrecord(10, data_dir)
    images = next(iter(d.ds))
    images = (images.numpy() + 1) / 2
    for i in range(2):
        for j in range(5):
            n = i*5+j
            plt.subplot(2, 5, n+1)
            plt.imshow(images[n])
            plt.xticks(())
            plt.yticks(())
    plt.show()


def parse_tfreord(data_dir):
    d = Anime(1, data_dir)
    d.to_tf_recoder()


def load_tfrecord(batch_size, data_dir):
    d = Anime(batch_size, data_dir)
    d.load_tf_recoder()
    return d


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", type=str)
    args = parser.parse_args()

    t0 = time.time()
    parse_tfreord(args.data_dir)
    # ds = load_celebA_tfrecord(20, args.data_dir)
    # t1 = time.time()
    # print("load time", t1-t0)
    # count = 0
    # while True:
    #     for img, label in ds:
    #         # if _ % 200 == 0:
    #         count+=1
    #         if count % 500==0: print(img.shape, label.shape)
    #         if count == 10000:
    #             break
    #     if count == 10000:
    #         break
    #
    # print("runtime", time.time()-t1)
    show_sample(args.data_dir)
