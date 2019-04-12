from typing import List

import cv2
import numpy as np
import tensorflow as tf
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def sparse_tuple_from(sequences, lang_dict, dtype=np.int32):
    """
        Inspired (copied) from https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
    """

    indices = []
    values = []
    sequences = [seq.decode() for seq in sequences]
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), [i for i in range(len(seq))]))
        values.extend([lang_dict.word2idx[s] for s in seq])

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def sparse_tensor_to_str(lang_dict, sparse_tensor: tf.SparseTensor) -> List[str]:
    """
    :param sparse_tensor: prediction or ground truth label
    :return: String value of the sparse tensor
    """
    indices = sparse_tensor.indices
    values = sparse_tensor.values
    # Translate from consecutive numbering into ord() values

    dense_shape = sparse_tensor.dense_shape

    number_lists = np.ones(dense_shape, dtype=values.dtype) * -1
    str_lists = []
    for i, index in enumerate(indices):
        number_lists[index[0], index[1]] = values[i]
    for number_list in number_lists:
        str_lists.append(ground_truth_to_word(lang_dict, number_list))
    return str_lists


def dense_to_str(lang_dict, dense):
    str_lists = []
    for number_list in dense:
        str_lists.append(ground_truth_to_word(lang_dict, number_list))
    return str_lists


def sparse_to_str(lang_dict, indices, values, dense_shape):
    number_lists = np.ones(dense_shape, dtype=values.dtype) * -1
    str_lists = []
    for i, index in enumerate(indices):
        number_lists[index[0], index[1]] = values[i]
    for number_list in number_lists:
        str_lists.append(ground_truth_to_word(lang_dict, number_list))
    return str_lists


def ground_truth_to_word(lang_dict, ground_truth):
    try:
        return ''.join([lang_dict.idx2word[int(i)] for i in ground_truth if int(i) != -1])
    except Exception as ex:
        print(ground_truth)
        print(ex)
        input()


def compute_accuracy(ground_truth: List[str], predictions: List[str],
                     display: bool = True) -> np.float32:
    """ Computes accuracy
    TODO: this could probably be optimized

    :param ground_truth:
    :param predictions:
    :param display: Whether to print values to stdout
    :return:
    """
    accuracy = []

    for index, label in enumerate(ground_truth):
        prediction = predictions[index]
        total_count = len(label)
        correct_count = 0
        try:
            for i, tmp in enumerate(label):
                if tmp == prediction[i]:
                    correct_count += 1
        except IndexError:
            continue
        finally:
            try:
                accuracy.append(correct_count / total_count)
            except ZeroDivisionError:
                if len(prediction) == 0:
                    accuracy.append(1)
                else:
                    accuracy.append(0)

    accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    if display:
        pass
        # print('Mean accuracy is {:5f}'.format(accuracy))

    return accuracy
