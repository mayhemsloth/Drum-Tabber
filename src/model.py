#================================================================
#
#   File name   : model.py
#   Author      : Thomas Hymel
#   Created date: 9-2-2020
#   GitHub      : https://github.com/mayhemsloth/Drum-Tabber
#   Description : Functions for creation of model for automatic drum tabbing
#
#================================================================

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, ReLU, LeakyReLU, Dropout, Flatten, Concatenate, Add
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, ZeroPadding2D, AveragePooling2D

from src.configs import *

'''  ~~~~~ Initial Drum-Tabber model goals~~~~~
The Drum-Tabber model will be modeled after (or heavily informed by) the results shown in this paper:
Drum Transcription via Joint Beat and Drum Modeling Using Convolutional Recurrent Neural Networks (2017)
http://archives.ismir.net/ismir2017/paper/000123.pdf

Initially the main thing I want to build is a "context aware" CNN model. That is, the model takes in sequences of windows for
every window that exists in the spectrogram. This sequence of windows is called "context" in the paper, and I will use this
terminology here as well. I want to make the context of each window to be a hyperparamater/configs constant, so I should
code the context CNN model with this in mind (N_CONTEXT_PRE and _POST), including the fact that if you let the total context
window length go to 1, it simplifies to a no-context CNN-only model.

The reason that I want to build this context aware CNN model is because the paper shows it did pretty well compared to
a context aware+recurrent NN model, and it is much easier for me to code this model first before getting into recurrent NN
layers. Assuming you only use CNNs, the problem entirely transforms into a "image recognition" problem, and I am currently
more familiar with that type of problem than anything using RNNs (although I do want to build a CRNN based model
at some point). RNN are used for long-term structure in time sequenced data, but in reality a context CNN model MAY
have enough informational context to learn the "structure" of a drum onset, but not of the long term drum patterns
(which I particularly don't care about, but only care if it helps the model understand the drum onsets better, which
I am hypothesizing will not)
'''

''' ~~~~~ Model input/outputs and number of classes ~~~~~
Regardless of the model type, the inputs are going to be the same (log melspectrogram with 1-3 channels and varying numbers
of mels / inclusion of first-time-derivative as determined by the configs.py file constants) Additionally, depending
on the configs, the outputs will all be the same as well. The "simplest" model that I will code for is one that predicts
7 classes: beats, downbeats, BD, SD, at (all toms), ac (all cymbals), and HH (x only), but the model code should be
flexible enough to accommodate any number of classes as determined by the configs parameters. That is, the number
of classes that the model is prepped for needs to be gathered from some external source that already has
knowledge of the encoded df.
'''


'''
: This class extension was in the tutorial code, I need to look into if this is actually what needs to be done
: Need to understand what this code is doing!

    class BatchNormalization(BatchNormalization):
        # "Frozen state" and "inference mode" are two separate concepts.
        # `layer.trainable = False` is to freeze the layer, so the layer will use
        # stored moving `var` and `mean` in the "inference mode", and both `gama`
        # and `beta` will not be updated !
        def call(self, x, training=False):
            if not training:
                training = tf.constant(False)
            training = tf.logical_and(training, self.trainable)
            return super().call(x, training)
'''

def conv2D_block(input, filter_num, kernel_shape, strides_arg = (1,1), activation = 'relu', padding_arg = 'same'):
    '''
    Helper builder function to define a Conv2D block, complete with BatchNorm and activation that follows the Conv2D layer

    Args:
        input [tensorflow tensor]: the tensorflow tensor that is the next input of the current CNN
        filter_num [int]: how many filters are present in this Conv2D
        kernel_shape [int tuple]: the shape of each filter.
        strides_arg [int tupe]: Default (1,1), the shape of the strides that will be used
        activ [str]: Default 'relu', will determine the activation function of this Conv2D block

    Returns:
        tensorflow tensor: the tensor representing the output after being passed through the conv layers
    '''


    output = Conv2D(filters = filter_num, kernel_size = kernel_shape, strides = strides_arg,
                    padding = padding_arg,  data_format = 'channels_last', use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001) )(input)

    output = BatchNormalization()(output)

    if activation == 'relu':
        output = ReLU()(output)
    elif activation == 'leaky_relu':
        output = LeakyReLU(alpha=0.2)(output)
    elif activation == 'none':   # don't apply any activation. In the case of residual block
        pass

    return output

def residual_c_block(input, filter_nums, strides_arg = (1,1), activation = 'relu', padding_arg='same'):

    input_skip = input
    filter1, filter2 = filter_nums

    # first block (1x1 conv)
    output = conv2D_block(input, filter1, kernel_shape=(1,1), strides_arg = strides_arg, activation = activation)

    # second block (3x3 conv)
    output = conv2D_block(output, filter1, kernel_shape=(3,3), strides_arg = strides_arg, activation = activation)

    # third block (1x1) (no activation)
    output = conv2D_block(output, filter2, kernel_shape=(1,1), strides_arg = strides_arg, activation='none')

    # shortcut block (no activation)
    input_skip = conv2D_block(input_skip, filter2, kernel_shape=(1,1), strides_arg = strides_arg, activation='none')

    # Add and Activate
    output = Add()([output, input_skip])
    if activation == 'relu':
        output = ReLU()(output)
    elif activation == 'leaky_relu':
        output = LeakyReLU(alpha=0.2)(output)

    return output

def residual_i_block(input, filter_nums, strides_arg = (1,1), activation = 'relu', padding_arg='same'):
    ''' note that the filter2 needs to be the same number of channels as the input'''
    input_skip = input
    filter1, filter2 = filter_nums

    # first block (1x1 conv)
    output = conv2D_block(input, filter1, kernel_shape=(1,1), strides_arg = strides_arg, activation = activation)

    # second block (3x3 conv)
    output = conv2D_block(output, filter2, kernel_shape=(3,3), strides_arg = strides_arg, activation = activation)

    # Add and Activate
    output = Add()([output, input_skip])
    if activation == 'relu':
        output = ReLU()(output)
    elif activation == 'leaky_relu':
        output = LeakyReLU(alpha=0.2)(output)

    return output

def create_DrumTabber(n_features, n_classes, activ = 'relu', training = False):
    '''
    Main function that creates the tf layers that describe the DrumTabber NN model and outputs a keras.Model object

    Args:
        n_features [int]: the number of features (rows) in the initial spectrogram 2D array
        n_classes [int]: the number of classes to be classified in this model
        activ [str]: Default 'relu'. Changes all Conv2D layers and Dense layers (minus the output layer) to this activation.
                    : Other options include 'leaky_relu'
        training [bool]: Default False, pass True if the

    Returns:
        keras.Model: a Model that has the correct input and output layers
    '''

    # TODO: Handle the Context-CNN case where the Context == 0. Need to handle the Conv2D filters differently
    if MODEL_TYPE == 'Context-CNN':
        #  'channels_last' ordering: will be using the shape of layers as (batch_size, n_features, n_context, channels = 1)
        input_layer = Input(shape = (n_features, (N_CONTEXT_PRE+1+N_CONTEXT_POST), 1, ), dtype = 'float32')  # creates a None in first dimension for the batch size

        # 2 x Block Conv2D-32-3x3-BN-ReLU
        output = conv2D_block(input_layer, 32, (3,3), activation = activ)
        output = conv2D_block(output, 32, (3,3), activation = activ)

        # DropOut
        output = Dropout(rate = 0.2)(output, training = training)

        # MaxPool
        output = MaxPool2D(pool_size = (5,5), strides=None, padding = 'same')(output)

        # 2 x Block Conv2D-64-3x3-BN-ReLU
        output = conv2D_block(output, 64, (3,3), activation = activ)
        output = conv2D_block(output, 64, (3,3), activation = activ)

        # DropOut
        output = Dropout(rate = 0.2)(output, training = training)

        # MaxPool
        output = MaxPool2D(pool_size = (3,3), strides=None, padding = 'same')(output)

        # Flatten to prepare for Dense
        output = Flatten()(output)

        # 1 x 512 FC Dense + activation
        output = Dense(512, activation = activ)(output)
        output = BatchNormalization()(output)

        # FC Dense sigmoid activation
        output = Dense(n_classes, activation = 'sigmoid')(output)


    elif MODEL_TYPE == 'TimeFreq-CNN':
        '''
        "TimeFreq-CNN" stands for Time+Frequency Convolutional NN. IMO the previous literature borrowed TOO many things
        from the computer vision community. For one, most CNN filters/strides are square because a picture, inherently,
        has "equivalent" x- and y-axis that should be treated as equal because they both represent spatial dimensions.
        This is NOT TRUE for spectrograms. In a spectrogram, the x-axis represents time and the y-axis represents the
        frequencies present (the power of each frequency bin). As such, square filters don't make much sense.
        However if we believe filters can learn general enough features, than square features, after multiple conv blocks,
        should be able to learn general features. We can help this process along by treating the time and frequency axes as
        differently from each other in certain regards.

        The "TimeFreq-CNN" architecture relies on "context" structure and thus shares similar code to Context-CNN
        model type in other parts of the code base.
        '''

        #  'channels_last' ordering: will be using the shape of layers as (batch_size, n_features, n_context, channels = 1)
        input_layer = Input(shape = (n_features, (N_CONTEXT_PRE+1+N_CONTEXT_POST), 1, ), dtype = 'float32')  # creates a None in first dimension for the batch size

        # ZeroPadding 2D layer
        zero_padded = ZeroPadding2D(padding = (0,0))(input_layer)

        # Frequency branch: the convs that are looking for features across wide range of freqencies (tall filters)
        freq_branch = conv2D_block(zero_padded, 64,  kernel_shape = (15,3), strides_arg = (1,1), activation = activ)
        # Frequency branch: residual i block
        freq_branch = residual_i_block(freq_branch, filter_nums=(32,64), strides_arg = (1,1), activation = activ)
        # Frequency branch: MaxPool
        freq_branch = MaxPool2D(pool_size = (3,3), strides=None, padding = 'same')(freq_branch)


        # Time branch: the convs that are looking for features across wide range of time (wide filters)
        time_branch = conv2D_block(zero_padded, 64,  kernel_shape = (3,9), strides_arg = (1,1), activation = activ) # note that input layer goes here as well
        # Time branch: residual i block
        time_branch = residual_i_block(time_branch, filter_nums=(32,64), strides_arg = (1,1), activation = activ)
        # Time branch: MaxPool
        time_branch = MaxPool2D(pool_size = (3,3), strides=None, padding = 'same')(time_branch)



        # Combine the branches, concatenating along the channels
        timefreq = Concatenate()([freq_branch, time_branch])
        # convolve the timefreq, expanding the num_channels using a res_block
        timefreq = conv2D_block(timefreq, 128,  kernel_shape = (3,3), strides_arg = (1,1), activation = activ)
        timefreq = residual_c_block(timefreq, filter_nums=(128,256), strides_arg = (1,1), activation = activ)
        #timefreq = residual_i_block(timefreq, filter_nums=(128,256), strides_arg = (1,1), activation = activ)

        # Asymmetric Pool2D
        timefreq = MaxPool2D(pool_size = (3,1), strides=None, padding = 'same')(timefreq)

        # 1x1 conv the TimeFreq: "downsampling" the number of channels
        timefreq = conv2D_block(timefreq, 64, kernel_shape = (1,1), strides_arg = (1,1), activation = activ)

        # Asymmetric Pool2D
        timefreq = MaxPool2D(pool_size = (3,3), strides=None, padding = 'same')(timefreq)

        # Flatten to prepare for Dense
        output = Flatten()(timefreq)

        # 1 x 256 FC Dense + activation
        output = Dense(256, activation = activ)(output)
        output = BatchNormalization()(output)

        # FC Dense sigmoid activation
        output = Dense(n_classes, activation = 'sigmoid')(output)

    # TODO: handle the other model type cases
    else:
        print('create_DrumTabber: Please choose a valid MODEL_TYPE in configs')
        return None

    return tf.keras.Model(inputs = input_layer, outputs = output)

'''
: need a compute_loss function
'''
