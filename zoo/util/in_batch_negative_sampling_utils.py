import logging

import tensorflow as tf
from keras import backend

from zoo.zoo_constants import ZooConstants


def _cartesian_expand(l, r):
    s = tf.reshape(tf.linalg.matmul(tf.reshape(l, [-1, 1]), tf.reshape(r, [-1, 1]), transpose_b=True), [-1])
    return s


def _generate_indices_v2(batch_size, n_repeats):
    repeats = batch_size
    if n_repeats > 0:
        repeats = n_repeats
    # if config.get(ZooConstants.IN_BATCH_NEGATIVE_SAMPLING_REPEATS, -1) > 0:
    #     repeats = config.get(ZooConstants.IN_BATCH_NEGATIVE_SAMPLING_REPEATS, -1)

    repeats = tf.where(tf.greater(repeats, batch_size), batch_size, repeats)

    #  0,0,.0,1,1.1,..,n-1,.,n-1
    repeat_indices = _cartesian_expand(
        tf.reshape(tf.range(batch_size), [-1, 1])
        , tf.ones([repeats, 1], dtype=tf.int32)
    )

    #  0,1,2,.,n-1,0,1,2,.,n-1,.,0,1,2,.,n-1
    shuffle_indices = _cartesian_expand(
        tf.ones([batch_size, 1], dtype=tf.int32)
        , tf.reshape(tf.compat.v1.random_shuffle(tf.range(batch_size))[:repeats], [-1, 1])
    )

    mask = tf.not_equal(repeat_indices, shuffle_indices)

    # repeat_indices = tf.where(mask, ,repeat_indices)

    shuffle_indices = tf.where(mask, shuffle_indices, tf.compat.v1.mod(repeat_indices + 1, batch_size))
    # tf.boolean_mask(shuffle_indices, mask)

    return shuffle_indices, repeat_indices


def generate_negative_sampling_cartesian(emb, emb_augmented, config):
    batch_size = tf.shape(emb)[0]
    shuffle_indices_negative, repeat_indices_negative = _generate_indices_v2(
        batch_size, config.get(ZooConstants.IN_BATCH_NEGATIVE_SAMPLING_REPEATS, -1))

    negative_sample_num = tf.shape(shuffle_indices_negative)[0]

    shuffle_indices = tf.concat([tf.range(batch_size), shuffle_indices_negative], axis=0)
    repeat_indices = tf.concat([tf.range(batch_size), repeat_indices_negative], axis=0)

    learning_phase = backend.learning_phase()
    logging.info('learning_phase: {}'.format(learning_phase))
    shuffle_indices = tf.cond(learning_phase, lambda: shuffle_indices, lambda: tf.range(batch_size))
    repeat_indices = tf.cond(learning_phase, lambda: repeat_indices, lambda: tf.range(batch_size))

    emb_expanded = tf.gather(emb, shuffle_indices)
    emb_augmented_expanded = tf.gather(emb_augmented, repeat_indices)

    label = tf.cond(
        learning_phase, lambda: tf.concat([tf.ones([batch_size]), 0 * tf.ones([negative_sample_num])], axis=0),
        lambda: tf.ones([batch_size]))

    return emb_expanded, emb_augmented_expanded, label


def generate_negative_sampling_cartesian_2(emb, emb_augmented, n_repeats):
    batch_size = tf.shape(emb)[0]
    shuffle_indices_negative, repeat_indices_negative = _generate_indices_v2(
        batch_size, n_repeats)

    negative_sample_num = tf.shape(shuffle_indices_negative)[0]

    shuffle_indices = tf.concat([tf.range(batch_size), shuffle_indices_negative], axis=0)
    repeat_indices = tf.concat([tf.range(batch_size), repeat_indices_negative], axis=0)

    learning_phase = backend.learning_phase()
    logging.info('learning_phase: {}'.format(learning_phase))
    shuffle_indices = tf.cond(learning_phase, lambda: shuffle_indices, lambda: tf.range(batch_size))
    repeat_indices = tf.cond(learning_phase, lambda: repeat_indices, lambda: tf.range(batch_size))

    emb_expanded = tf.gather(emb, shuffle_indices)
    emb_augmented_expanded = tf.gather(emb_augmented, repeat_indices)

    label = tf.cond(
        learning_phase, lambda: tf.concat([tf.ones([batch_size]), 0 * tf.ones([negative_sample_num])], axis=0),
        lambda: tf.ones([batch_size]))

    return emb_expanded, emb_augmented_expanded, label

def generate_negative_sampling_cartesian_3(emb, emb_augmented, n_repeats, shuffle_indices=None, repeat_indices=None):
    learning_phase = backend.learning_phase()
    batch_size = tf.shape(emb)[0]

    if shuffle_indices is None or repeat_indices is None:
        shuffle_indices_negative, repeat_indices_negative = _generate_indices_v2(
            batch_size, n_repeats)

        negative_sample_num = tf.shape(shuffle_indices_negative)[0]

        shuffle_indices = tf.concat([tf.range(batch_size), shuffle_indices_negative], axis=0)
        repeat_indices = tf.concat([tf.range(batch_size), repeat_indices_negative], axis=0)

        logging.info('learning_phase: {}'.format(learning_phase))
        shuffle_indices = tf.cond(learning_phase, lambda: shuffle_indices, lambda: tf.range(batch_size))
        repeat_indices = tf.cond(learning_phase, lambda: repeat_indices, lambda: tf.range(batch_size))

    emb_expanded = tf.gather(emb, shuffle_indices)
    emb_augmented_expanded = tf.gather(emb_augmented, repeat_indices)

    batch_size_expanded = tf.shape(shuffle_indices)[0]

    label = tf.cond(
        learning_phase, lambda: tf.concat([tf.ones([batch_size]), 0 * tf.ones([batch_size_expanded - batch_size])], axis=0),
        lambda: tf.ones([batch_size]))

    return emb_expanded, emb_augmented_expanded, label, shuffle_indices, repeat_indices


def construct_pairwise_sample(logit, label, session_id, sample_weight):
    batch_size = tf.shape(logit)[0]
    shuffle_indices_negative, repeat_indices_negative = _generate_indices_v2(
        batch_size, -1)

    shuffle_indices = tf.concat([tf.range(batch_size), shuffle_indices_negative], axis=0)
    repeat_indices = tf.concat([tf.range(batch_size), repeat_indices_negative], axis=0)

    learning_phase = backend.learning_phase()
    logging.info('learning_phase: {}'.format(learning_phase))
    shuffle_indices = tf.cond(learning_phase, lambda: shuffle_indices, lambda: tf.range(batch_size))
    repeat_indices = tf.cond(learning_phase, lambda: repeat_indices, lambda: tf.range(batch_size))

    logit_l = tf.gather(logit, shuffle_indices)
    logit_r = tf.gather(logit, repeat_indices)

    label_l = tf.gather(label, shuffle_indices)
    label_r = tf.gather(label, repeat_indices)

    session_id_l = tf.gather(session_id, shuffle_indices)
    session_id_r = tf.gather(session_id, repeat_indices)

    mask = tf.reshape(tf.logical_and(
        tf.less(tf.abs(session_id_l - session_id_r), 1e-6)
        , tf.greater(tf.abs(label_l - label_r), 0)), shape=[-1])

    logit_diff = tf.boolean_mask(logit_l - logit_r, mask)
    label_diff = tf.boolean_mask(tf.cast(label_l - label_r > 0, dtype=tf.float32), mask)
    session_id_l = tf.boolean_mask(session_id_l, mask)
    session_id_r = tf.boolean_mask(session_id_r, mask)

    return logit_diff, label_diff, session_id_l, session_id_r


def construct_pairwise_sample_v2(logit, label, session_id, sample_weight):
    batch_size = tf.shape(logit)[0]
    shuffle_indices_negative, repeat_indices_negative = _generate_indices_v2(
        batch_size, -1)

    shuffle_indices = tf.concat([tf.range(batch_size), shuffle_indices_negative], axis=0)
    repeat_indices = tf.concat([tf.range(batch_size), repeat_indices_negative], axis=0)

    learning_phase = backend.learning_phase()
    logging.info('learning_phase: {}'.format(learning_phase))
    shuffle_indices = tf.cond(learning_phase, lambda: shuffle_indices, lambda: tf.range(batch_size))
    repeat_indices = tf.cond(learning_phase, lambda: repeat_indices, lambda: tf.range(batch_size))

    logit_l = tf.gather(logit, shuffle_indices)
    logit_r = tf.gather(logit, repeat_indices)

    label_l = tf.gather(label, shuffle_indices)
    label_r = tf.gather(label, repeat_indices)

    session_id_l = tf.gather(session_id, shuffle_indices)
    session_id_r = tf.gather(session_id, repeat_indices)

    sample_weight_l = tf.gather(sample_weight, shuffle_indices)
    sample_weight_r = tf.gather(sample_weight, repeat_indices)
    sample_weight_lgt = tf.cast(sample_weight_l > sample_weight_r, dtype=tf.float32)
    sample_weight_pair = sample_weight_lgt * sample_weight_l + (1.0 - sample_weight_lgt) * sample_weight_r

    mask = tf.reshape(tf.logical_and(
        tf.less(tf.abs(session_id_l - session_id_r), 1e-6)
        , tf.greater(tf.abs(label_l - label_r), 0)), shape=[-1])

    logit_diff = tf.boolean_mask(logit_l - logit_r, mask)
    label_diff = tf.boolean_mask(tf.cast(label_l - label_r > 0, dtype=tf.float32), mask)
    session_id_l = tf.boolean_mask(session_id_l, mask)
    session_id_r = tf.boolean_mask(session_id_r, mask)
    sample_weight_pair = tf.boolean_mask(sample_weight_pair, mask)

    return logit_diff, label_diff, session_id_l, session_id_r, sample_weight_pair