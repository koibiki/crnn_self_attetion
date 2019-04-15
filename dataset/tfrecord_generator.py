import os.path as osp

import cv2
import tensorflow as tf

from utils.img_utils import resize_image

save_dir = "/home/chengli/data/fine_data_tf"


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class DataGenerator(object):

    @classmethod
    def generator_by_tuple(cls, t):
        cls.generator(*t)

    @classmethod
    def generator(cls, index, lang_dict, data):
        with tf.python_io.TFRecordWriter(osp.join(save_dir, "tfdata_{:d}.tfrecords".format(index))) as tfrecord_writer:
            for i in range(len(data)):
                imread = cv2.imread(data[i], cv2.IMREAD_GRAYSCALE)

                if imread is None:
                    continue

                imread = resize_image(imread, 100, 32)

                label = data[i].split("/")[-1].split("_")[1]

                imread_bytes = imread.tobytes()

                label = [lang_dict.word2idx[l] for l in label]
                # create features
                feature = {'train/image': _bytes_feature([imread_bytes]),
                           'train/label': _int64_feature(label)}
                # create example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                # serialize protocol buffer to string
                tfrecord_writer.write(example.SerializeToString())
        print("finish write tfdata_{:d}.tfrecords.")
