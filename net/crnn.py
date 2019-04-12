import tensorflow as tf
import tensorflow.contrib.slim as slim


class CrnnNet(object):

    def __init__(self):
        self.feature_net = CnnFeature()
        self.self_attention = SelfAttention()

    def __call__(self, inputs, mode, batch_size, num_classes):
        with tf.variable_scope("crnn"):
            cnn_feature = self.feature_net(inputs, mode == tf.estimator.ModeKeys.TRAIN)

            squeeze = tf.squeeze(input=cnn_feature, axis=[1], name='squeeze')

            attention = self.self_attention(squeeze, mode == tf.estimator.ModeKeys.TRAIN)

            concat = tf.concat([squeeze, attention], axis=-1)

            raw_logits = tf.layers.dense(concat, num_classes, activation=None, name="raw_logits")

            decoded_logits = tf.transpose(raw_logits, (1, 0, 2), name='decoded_logits')  # [width, batch, n_classes]

        return raw_logits, decoded_logits


class CnnFeature(tf.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, inputs, training, **kwargs):
        with tf.variable_scope("cnn"):
            conv1 = tf.layers.conv2d(
                inputs=inputs, filters=64, kernel_size=(3, 3), padding="same",
                activation=tf.nn.relu, name='conv1')

            pool1 = tf.layers.max_pooling2d(
                inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')

            conv2 = tf.layers.conv2d(
                inputs=pool1, filters=64, kernel_size=(3, 3), padding="same",
                activation=tf.nn.relu, name='conv2')

            pool2 = tf.layers.max_pooling2d(
                inputs=conv2, pool_size=[2, 2], strides=2, padding="same", name='pool2')

            conv3 = tf.layers.conv2d(
                inputs=pool2, filters=128, kernel_size=(3, 3), padding="same",
                activation=tf.nn.relu, name='conv3')

            conv4 = tf.layers.conv2d(
                inputs=conv3, filters=128, kernel_size=(3, 3), padding="same",
                activation=tf.nn.relu, name='conv4')

            pool3 = tf.layers.max_pooling2d(
                inputs=conv4, pool_size=[2, 1], strides=[2, 1], padding="same", name='pool3')

            conv5 = tf.layers.conv2d(
                inputs=pool3, filters=256, kernel_size=(3, 3), padding="same",
                activation=tf.nn.relu, name='conv5')

            bnorm1 = tf.layers.batch_normalization(conv5, training=training)

            conv6 = tf.layers.conv2d(
                inputs=bnorm1, filters=256, kernel_size=(3, 3), padding="same",
                activation=tf.nn.relu, name='conv6')

            bnorm2 = tf.layers.batch_normalization(conv6, training=training)

            pool4 = tf.layers.max_pooling2d(
                inputs=bnorm2, pool_size=[2, 1], strides=[2, 1], padding="same", name='pool4')

            conv7 = tf.layers.conv2d(
                inputs=pool4, filters=512, kernel_size=[2, 2], strides=[2, 1], padding="same",
                activation=tf.nn.relu, name='conv7')

        return conv7


class SelfAttention(tf.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, inputs, training, *args, **kwargs):
        with tf.variable_scope('self_attention'):
            q = tf.layers.dense(inputs, 512, activation=None, name="query")
            k = tf.layers.dense(inputs, 512, activation=None, name="key")
            v = tf.layers.dense(inputs, 512, activation=None, name="value")

            logits = tf.matmul(q, k, transpose_b=True)
            logits = slim.bias_add(logits)
            weights = tf.nn.softmax(logits, name="attention_weights")
            if training:
                weights = tf.nn.dropout(weights, 0.5)
            attention_output = tf.matmul(weights, v)
        return attention_output
