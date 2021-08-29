from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv1D, GlobalMaxPooling1D, Activation, MaxPooling1D
from tensorflow.keras.activations import exponential
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import concatenate
import tensorflow as tf


def pearson(true, pred):
    x = tf.convert_to_tensor(true)
    y = tf.cast(pred, x.dtype)
    xmean = tf.math.reduce_mean(x)
    ymean = tf.math.reduce_mean(y)

    xm = x - xmean
    ym = y - ymean

    xnorm = tf.norm(xm, ord=2)
    ynorm = tf.norm(ym, ord=2)

    r = tf.tensordot(xm / xnorm, ym / ynorm, axes=2)
    return r


def base_block(input_layer, num_filters, filter_size, active='exp', maxp='global'):
    conv = Conv1D(filters=num_filters, kernel_size=filter_size, strides=1, kernel_initializer='glorot_uniform',
                  kernel_regularizer=l2(5e-3), padding='same', bias_initializer='RandomNormal')(input_layer)

    if active == 'exp':
        bn = Activation(exponential)(conv)
    else:
        bn = Activation(active)(conv)

    if maxp == 'global':
        pool = GlobalMaxPooling1D()(bn)
    elif maxp == 'local':
        pool = MaxPooling1D(pool_size=3)(bn)

    return pool


def base_model(in_sh, num_filters, filter_size, fc):
    input_seq = Input(shape=in_sh)

    base = base_block(input_seq, num_filters, filter_size[0], active='relu')
    hidden = Dense(fc)(base)
    hidden = Activation('relu')(hidden)
    output = Dense(1)(hidden)
    output = Activation('linear')(output)
    model = Model(inputs=input_seq, outputs=output)

    return model


def gen_model(in_sh, num_filters, filter_size, fc, lr):
    model = base_model(in_sh, num_filters, filter_size, fc)

    opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.99, epsilon=1e-8, decay=1e-5)

    model.compile(loss='mse', optimizer=opt, metrics=[pearson])

    return model
