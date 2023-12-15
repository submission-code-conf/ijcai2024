import keras
import tensorflow as tf
from keras.layers import Dense


def add_suffix_2_key(d, suffix):
    assert d is not None
    if isinstance(d, dict):
        return dict([('{}_{}'.format(k, suffix), v) for k, v in d.items()])
    elif isinstance(d, list):
        return ['{}_{}'.format(k, suffix) for k in d]
    else:
        return '{}_{}'.format(d, suffix)

def create_mlp_layer(input, n_units):
    w = tf.get_variable(name='w', shape=(input.shape[1], n_units),
                        initializer=tf.random_normal_initializer(0, 0.01))
    bias = tf.get_variable(name='bias', shape=(n_units), initializer=tf.zeros_initializer())

    z = tf.matmul(input, w) + bias
    h = tf.nn.relu(z)
    return h, z


def create_mlp_layers(input, units_list, scope=''):
    h = input
    for i in range(len(units_list)):
        with tf.compat.v1.variable_scope('{}_mlp_layer_{}'.format(scope, i)):
            h, z = create_mlp_layer(h, units_list[i])
    return h, z

def calc_cosine_similarity(l, r):
    l = tf.nn.l2_normalize(l, axis=-1)
    r = tf.nn.l2_normalize(r, axis=-1)
    similarity = tf.reduce_sum(tf.multiply(l, r), axis=-1)
    return similarity


def create_mlp_layer_v2(hidden_units_list, output_dim, output_activation, scope_name):
    with tf.compat.v1.variable_scope(scope_name):
        l = keras.Sequential(
            [Dense(hidden_units_list[i], activation='relu') for i in range(len(hidden_units_list))]
            + [Dense(output_dim, activation=output_activation)])
    return l


