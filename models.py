from keras.models import Model
from keras.layers import Dense, Input, Conv1D, Activation, Dropout
from keras.layers.pooling import GlobalMaxPooling1D, MaxPool1D
from keras.layers.merge import concatenate
from keras import optimizers
from keras.regularizers import l2


def first_features(input_layer, f, s):

    conv_layer = Conv1D(filters=f, kernel_size=s, strides=1, kernel_initializer='RandomNormal',
                        activation='relu', kernel_regularizer=l2(5e-3), padding='same', use_bias=True,
                        bias_initializer='RandomNormal')(input_layer)
    pool = GlobalMaxPooling1D()(conv_layer)

    return pool

def model_base(in_sh1, filter_size, lr, fc, opt_func='Adam'):

    in1 = Input(shape=in_sh1)
    flat = first_features(in1, filter_size, 12)

    hidden1 = Dense(fc)(flat)
    hidden1 = Activation('relu')(hidden1)
    output = Dense(1)(hidden1)
    output = Activation('linear')(output)
    model = Model(inputs=in1, outputs=output)

    if 'sgd' == opt_func:
        opt = optimizers.SGD(lr=lr, decay=1e-4, momentum=0.99, nesterov=True)
    else:
        opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.99, epsilon=1e-8, decay=1e-5)

    model.compile(loss='mse', optimizer=opt, metrics=['mae'])

    return model


def model_split(in_sh1, in_sh2, filter_size, lr, fc, opt_func='Adam'):

    in1 = Input(shape=in_sh1)
    flat1 = first_features(in1, filter_size, 12)
    in2 = Input(shape=in_sh2)
    flat2 = first_features(in2, 128, 6)
    in3 = Input(shape=in_sh2)
    flat3 = first_features(in3, 128, 6)


    concate = concatenate([flat2, flat1, flat3])

    hidden1 = Dense(fc)(concate)
    hidden1 = Activation('relu')(hidden1)
    output = Dense(1)(hidden1)
    output = Activation('linear')(output)
    model = Model(inputs=[in1, in2, in3], outputs=output)

    if 'sgd' in opt_func:
        opt = optimizers.SGD(lr=lr, decay=1e-4, momentum=0.99, nesterov=True)
    else:
        opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.99, epsilon=1e-8, decay=1e-5)

    model.compile(loss='mse', optimizer=opt, metrics=['mae'])

    return model


def model_loop(in_sh1, in_sh2, in_sh3, filter_size, lr, fc, opt_func='Adam'):

    in1 = Input(shape=in_sh1)
    flat1 = first_features(in1, filter_size, 12)
    in2 = Input(shape=in_sh2)
    flat2 = first_features(in2, 128, 6)
    in3 = Input(shape=in_sh2)
    flat3 = first_features(in3, 128, 6)
    in4 = Input(shape=in_sh3)

    concate = concatenate([flat2, flat1, flat3, in4])

    hidden1 = Dense(fc)(concate)
    hidden1 = Activation('relu')(hidden1)
    output = Dense(1)(hidden1)
    output = Activation('linear')(output)
    model = Model(inputs=[in1, in2, in3, in4], outputs=output)

    if 'sgd' in opt_func:
        opt = optimizers.SGD(lr=lr, decay=1e-4, momentum=0.99, nesterov=True)
    else:
        opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.99, epsilon=1e-8, decay=1e-5)

    model.compile(loss='mse', optimizer=opt, metrics=['mae'])

    return model
