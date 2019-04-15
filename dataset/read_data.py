import os
import os.path as osp
import tensorflow as tf

from dataset.parse_tfrecords import parse_item

root = "/home/chengli/data/fine_data_tf"
tf_records_path = os.listdir(root)
dataset = tf.data.TFRecordDataset([osp.join(root, path) for path in tf_records_path])
dataset = dataset.map(parse_item, num_parallel_calls=12).batch(5)
dataset = dataset.repeat()
dataset = dataset.prefetch(32 * 2)
iterator = dataset.make_one_shot_iterator()
next = iterator.get_next()

with tf.Session() as sess:
    for i in range(1000):
        imgs, labels = sess.run(next)
        print(labels)
