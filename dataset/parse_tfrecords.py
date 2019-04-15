import tensorflow as tf


def parse_item(example_proto):
    dics = {
        'train/image': tf.FixedLenFeature([], tf.string),
        'train/label': tf.VarLenFeature(tf.int64),
    }
    parsed_example = tf.parse_single_example(serialized=example_proto, features=dics)
    image = tf.decode_raw(parsed_example['train/image'], out_type=tf.uint8)
    image = tf.reshape(image, shape=[32, 100, 1])
    image = tf.cast(image, dtype=tf.float32) / 255.
    label = parsed_example['train/label']
    label = tf.cast(label, dtype=tf.int32)
    return image, label
