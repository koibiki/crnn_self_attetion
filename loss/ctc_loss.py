import tensorflow as tf

"""
copy from https://github.com/tensorflow/models/blob/master/research/deep_speech/deep_speech.py
"""


def calculate_edit_distance(label_length, labels, decoded):
    sparse_labels = transfer2sparse(label_length, labels)
    sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), sparse_labels))
    return sequence_dist


def calculate_ctc_loss(label_length, ctc_input_length, labels, logits):
    """Computes the ctc loss for the current batch of predictions."""
    ctc_input_length = tf.to_int32(ctc_input_length)
    sparse_labels = transfer2sparse(label_length, labels)
    return tf.reduce_mean(tf.nn.ctc_loss(labels=sparse_labels, inputs=logits, sequence_length=ctc_input_length))


def transfer2sparse(label_length, labels):
    label_length = tf.to_int32(label_length)
    sparse_labels = tf.to_int32(tf.keras.backend.ctc_label_dense_to_sparse(labels, label_length))
    return sparse_labels
