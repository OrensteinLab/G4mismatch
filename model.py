from keras.models import Model
from keras.layers import Dense, Input, Conv1D, Activation, Dropout
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras import backend as K

def first_features(input_layer,f,s):

    conv_layer = Conv1D(filters=f, kernel_size=s, strides=1, kernel_initializer='RandomNormal',
                        activation='relu', kernel_regularizer=l2(5e-3), padding='same', use_bias=True,
                        bias_initializer='RandomNormal')(input_layer)
    pool = GlobalMaxPooling1D()(conv_layer)

    return pool



def model(input_shape, filters, opt_func, lr, fc):
    input_layer = Input(shape=input_shape)

    flat1 = first_features(input_layer, filters,12)
    hidden1 = Dense(fc)(flat1)
    hidden1 = Activation('relu')(hidden1)
    output = Dense(1)(hidden1)
    output = Activation('linear')(output)
    model = Model(inputs=input_layer, outputs=output)


    if 'sgd' in opt_func:
        opt = optimizers.SGD(lr=lr, decay=1e-4, momentum=0.99, nesterov=True)
    else:
        opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.99, epsilon=1e-8, decay=1e-5)

    model.compile(loss='mse', optimizer=opt, metrics=['mae'])

    return model

