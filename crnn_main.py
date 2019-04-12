import tensorflow as tf
import numpy as np
import os
from dataset.dataset import DataProvider
from loss.ctc_loss import calculate_ctc_loss, calculate_edit_distance
from net.crnn import CrnnNet
from config import cfg

print(tf.__version__)

tf.logging.set_verbosity(tf.logging.INFO)

provider = DataProvider()
train_input_fn = provider.generate_train_input_fn()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_feature_columns():
    feature_columns = {
        'images': tf.feature_column.numeric_column('images', (32, 100, 1)),
    }
    return feature_columns


def model_fn(features, labels, mode, params):
    labels_tensor = labels[0]
    labels_len_tensor = labels[1]

    ctc_sequence_length = cfg.SEQ_LENGTH * np.ones(cfg.TRAIN.BATCH_SIZE)

    # Create the input layers from the features
    feature_columns = list(get_feature_columns().values())

    images = tf.feature_column.input_layer(
        features=features, feature_columns=feature_columns)

    images = tf.reshape(images, shape=(-1, 32, 100, 1))

    crnn = CrnnNet()
    raw_logits, decoded_logits = crnn(images, mode, cfg.TRAIN.BATCH_SIZE, cfg.NUM_CLASSES)

    predicted_indices = tf.argmax(input=raw_logits, axis=1, name="raw_pred_tensor")
    probabilities = tf.nn.softmax(raw_logits, name='softmax_tensor')

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(inputs=decoded_logits,
                                                      sequence_length=ctc_sequence_length,
                                                      merge_repeated=False)

    dense = tf.sparse_to_dense(decoded[0].indices, [cfg.TRAIN.BATCH_SIZE, cfg.SEQ_LENGTH], decoded[0].values, -1)

    dense_pred = tf.cast(dense, dtype=tf.int32, name="dense_out")

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        global_step = tf.train.get_or_create_global_step()

        ctc_loss = calculate_ctc_loss(labels_len_tensor, ctc_sequence_length, labels_tensor, decoded_logits)
        ctc_loss = tf.identity(ctc_loss, name='ctc_loss')

        edit_distance = calculate_edit_distance(labels_len_tensor, labels_tensor, decoded)
        edit_distance = tf.identity(edit_distance, name='sequence_dist')

        tf.summary.scalar('ctc_entropy', ctc_loss)

        tf.summary.scalar('sequence_dist', tf.reduce_mean(edit_distance))

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(ctc_loss, global_step=global_step)
            return tf.estimator.EstimatorSpec(
                mode, loss=ctc_loss, train_op=train_op)

        # if mode == tf.estimator.ModeKeys.EVAL:
        #     eval_metric_ops = {
        #         'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)
        #     }
        #     return tf.estimator.EstimatorSpec(
        #         mode, loss=loss, eval_metric_ops=eval_metric_ops)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': predicted_indices,
            'probabilities': probabilities,
            'dense_pred': dense_pred
        }
        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)


tensors_to_log = {"ctc_loss": "ctc_loss", "sequence_dist": "sequence_dist"}

logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10, at_end=True)

run_config = tf.estimator.RunConfig(
    save_checkpoints_steps=1000,
    tf_random_seed=512,
    model_dir="./checkpoints",
    keep_checkpoint_max=3,
    log_step_count_steps=10,
)

estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

estimator.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])
