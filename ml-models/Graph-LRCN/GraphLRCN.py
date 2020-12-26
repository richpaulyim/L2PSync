"""
A collection of models we'll use to attempt to classify videos.
"""
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D,\
BatchNormalization, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l2
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D, AveragePooling2D)
from collections import deque
import sys

class GraphLSTM():

    def __init__(self, nb_classes, seq_length, saved_model, model, lrate):
        """
        `model` = lrcn
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load """

        # Set defaults.
        self.saved_model = saved_model
        self.seq_length = seq_length
        self.nb_classes = nb_classes
        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'lrcn':
            print("Loading CNN-LSTM model.")
            self.input_shape = (seq_length, 15, 15, 1)
            self.model = self.lrcn()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        optimizer = Adam(lr=lrate)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer,
                           metrics=metrics)

        print(self.model.summary())

    def lrcn(self):
        """Build a CNN into RNN.
        Starting version from:
            https://github.com/udacity/self-driving-car/blob/master/
                steering-models/community-models/chauffeur/models.py

        Heavily influenced by VGG-16:
            https://arxiv.org/abs/1409.1556

        Also known as an LRCN:
            https://arxiv.org/pdf/1411.4389.pdf
        """
        def add_default_block(model, kernel_filters, init, reg_lambda):

            # conv
            model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
                                             kernel_initializer=init, kernel_regularizer=l2(l=reg_lambda))))
            model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Activation('relu')))
            # conv
            model.add(TimeDistributed(Conv2D(kernel_filters, (2, 2), padding='same',
                                             kernel_initializer=init, kernel_regularizer=l2(l=reg_lambda))))
            model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Activation('relu')))
            # max pool
            model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1))))

            return model

        initialiser = 'glorot_uniform'
        reg_lambda  = 0.001

        model = Sequential() 

        # first (non-default) block
        model.add(TimeDistributed(Conv2D(8, (6, 6), strides=(1, 1), padding='same',
                                         kernel_initializer=initialiser, kernel_regularizer=l2(l=reg_lambda)),
                                  input_shape=self.input_shape))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))

        model.add(TimeDistributed(Conv2D(8, (5, 5), kernel_initializer=initialiser, kernel_regularizer=l2(l=reg_lambda))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))

        model.add(TimeDistributed(MaxPooling2D((4, 4), strides=(2, 2))))

        # 2nd-5th (default) blocks
        model = add_default_block(model, 8,  init=initialiser, reg_lambda=reg_lambda)
        #model = add_default_block(model, 128, init=initialiser, reg_lambda=reg_lambda)
        #model = add_default_block(model, 256, init=initialiser, reg_lambda=reg_lambda)
        #model = add_default_block(model, 512, init=initialiser, reg_lambda=reg_lambda)

        # LSTM output head
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(256, return_sequences=False, dropout=0.5))
        model.add(Dense(1, activation='sigmoid'))

        print('model builds')
        return model



