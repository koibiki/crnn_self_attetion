import os.path as osp
import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import *

from dataset.parse_tfrecords import parse_item
from lang_dict.lang_dict import LanguageDict
from utils.img_utils import resize_image
from config import cfg


class DataProvider(object):

    def __init__(self):
        self.lang_dict = LanguageDict()

    @staticmethod
    def _create_dataset_from_file(root, files):
        readlines = []
        for file in files:
            with open(osp.join(root, file), "r") as f:
                readlines += f.readlines()

        img_paths = []
        for img_name in tqdm(readlines, desc="read dir"):
            img_name = img_name.rstrip().strip()
            img_name = img_name.split(" ")[0]
            img_path = osp.join(root, img_name)
            img_paths.append(img_path)
        img_paths = img_paths[:1000000]
        labels = [img_path.split("/")[-1].split("_")[-2] for img_path in tqdm(img_paths, desc="generator label")]
        return img_paths, labels

    def _map_func(self, img_path_tensor, label):
        imread = cv2.imread(img_path_tensor.decode('utf-8'), cv2.IMREAD_GRAYSCALE)
        if imread is None:
            imread = cv2.imread("./sample/0_APPS_0.png", cv2.IMREAD_GRAYSCALE)
            label = "APPS"
        imread = resize_image(imread, 100, 32)
        imread = np.expand_dims(imread, axis=-1)
        imread = np.array(imread, np.float32) / 255.
        label_idx = [self.lang_dict.word2idx[s] for s in str(label)]
        label_idx = label_idx + [0 for _ in range(25 - len(label_idx))]
        return imread, label_idx, len(label)

    def generate_train_input_fn(self):
        # root = "/media/holaverse/aa0e6097-faa0-4d13-810c-db45d9f3bda8/holaverse/work/00ocr/crnn_data/fine_data"
        root = "/home/chengli/data/fine_data"
        train_files = ["annotation_train.txt"]
        batch_size = cfg.TRAIN.BATCH_SIZE

        def _input_fn():
            train_img_paths, train_labels = self._create_dataset_from_file(root, train_files)

            dataset = tf.data.Dataset.from_tensor_slices((train_img_paths, train_labels)) \
                .map(lambda item1, item2: tf.py_func(self._map_func, [item1, item2], [tf.float32, tf.int64, tf.int64])) \
                .shuffle(100)
            dataset = dataset.repeat()
            dataset = dataset.prefetch(32 * batch_size)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_one_shot_iterator()
            images, labels, labels_len = iterator.get_next()

            features = {'images': images}
            return features, (labels, labels_len)

        return _input_fn


class TfDataProvider(object):

    def __init__(self, lang_dict):
        self.lang_dict = lang_dict

    def generate_train_input_fn(self):
        def _input_fn():
            root = "/home/chengli/data/fine_data_tf"
            batch_size = cfg.TRAIN.BATCH_SIZE
            tf_records_path = [osp.join(root, path) for path in os.listdir(root)]
            dataset = tf.data.TFRecordDataset(tf_records_path)
            dataset = dataset.map(parse_item, num_parallel_calls=12).shuffle(32)
            dataset = dataset.repeat()
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(2 * batch_size)
            iterator = dataset.make_one_shot_iterator()
            images, labels = iterator.get_next()

            features = {'images': images}
            return features, labels

        return _input_fn
