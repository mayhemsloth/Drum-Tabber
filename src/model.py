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
from tensorflow.keras.layers import Conv2D, Input, ReLU, LeakyReLU, Dense, BatchNormalization, MaxPool2D, Dropout, Flatten

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

def conv2D_block(input, filter_num, kernel_shape, activation = 'relu'):
    '''
    Helper builder function to define a Conv2D block, complete with BatchNorm and activation that follows the Conv2D layer

    Args:
        input [tensorflow tensor]: the tensorflow tensor that is the next input of the current CNN
        filter_num [int]: how many filters are present in this Conv2D
        kernel_shape [int tuple]: the shape of each filte. Usually will be (3,3) unless there is experiments with it / no context
        activ [str]: Default 'relu', will determine the activation function of this Conv2D block

    Returns:
        tensorflow tensor: the tensor representing the output after being passed through the conv layers
    '''


    output = Conv2D(filters = filter_num, kernel_size = kernel_shape, strides = (1,1),
                    padding = 'same',  data_format = 'channels_last', use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001) )(input)

    output = BatchNormalization()(output)

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
        input_layer = Input(shape = (n_features, N_CONTEXT_PRE + 1 + N_CONTEXT_POST, 1, ), dtype = 'float32')  # creates a None in first dimension for the batch size

        # First BatchNormalization to standardize all data before putting it into the model.
        output = BatchNormalization()(input_layer)

        # 2 x Block Conv2D-32-3x3-BN-ReLU
        output = conv2D_block(output, 32, (3,3), activation = activ)
        # output = conv2D_block(output, 32, (3,3), activation = activ)

        # DropOut
        output = Dropout(rate = 0.1)(output, training = training)

        # MaxPool
        output = MaxPool2D(pool_size = (3,3), strides=None, padding = 'same')(output)

        # 2 x Block Conv2D-64-3x3-BN-ReLU
        # output = conv2D_block(output, 64, (3,3), activation = activ)
        # output = conv2D_block(output, 64, (3,3), activation = activ)

        # DropOut
        output = Dropout(rate = 0.1)(output, training = training)

        # MaxPool
        output = MaxPool2D(pool_size = (3,3), strides=None, padding = 'same')(output)

        # Flatten to prepare for Dense
        output = Flatten()(output)

        # 2 x 256 FC Dense + relu activation
        output = Dense(256, activation = activ)(output)
        output = BatchNormalization()(output)
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





'''
TAKEN FROM CODE OF RICHARD VOGL from "TOWARDS MULTI-INSTRuMENT DRUM TRANSCRIPTION" 2018 paper
Used to see an implementation of the peak picking code referenced in the paper
'''
'''
    def peak_picking(activations, threshold, smooth=None, pre_avg=0, post_avg=0,
                     pre_max=1, post_max=1):
        """
        Perform thresholding and peak-picking on the given activation function.

        Parameters
        ----------
        activations : numpy array
            Activation function.
        threshold : float
            Threshold for peak-picking
        smooth : int or numpy array, optional
            Smooth the activation function with the kernel (size).
        pre_avg : int, optional
            Use `pre_avg` frames past information for moving average.
        post_avg : int, optional
            Use `post_avg` frames future information for moving average.
        pre_max : int, optional
            Use `pre_max` frames past information for moving maximum.
        post_max : int, optional
            Use `post_max` frames future information for moving maximum.

        Returns
        -------
        peak_idx : numpy array
            Indices of the detected peaks.

        See Also
        --------
        :func:`smooth`

        Notes
        -----
        If no moving average is needed (e.g. the activations are independent of
        the signal's level as for neural network activations), set `pre_avg` and
        `post_avg` to 0.
        For peak picking of local maxima, set `pre_max` and  `post_max` to 1.
        For online peak picking, set all `post_` parameters to 0.

        References
        ----------
        .. [1] Sebastian Böck, Florian Krebs and Markus Schedl,
               "Evaluating the Online Capabilities of Onset Detection Methods",
               Proceedings of the 13th International Society for Music Information
               Retrieval Conference (ISMIR), 2012.

        """
        # smooth activations
        activations = smooth_signal(activations, smooth)
        # compute a moving average
        avg_length = pre_avg + post_avg + 1
        if avg_length > 1:
            # TODO: make the averaging function exchangeable (mean/median/etc.)
            avg_origin = int(np.floor((pre_avg - post_avg) / 2))
            if activations.ndim == 1:
                filter_size = avg_length
            elif activations.ndim == 2:
                filter_size = [avg_length, 1]
            else:
                raise ValueError('`activations` must be either 1D or 2D')
            mov_avg = uniform_filter(activations, filter_size, mode='constant',
                                     origin=avg_origin)
        else:
            # do not use a moving average
            mov_avg = 0
        # detections are those activations above the moving average + the threshold
        detections = activations * (activations >= mov_avg + threshold)
        # peak-picking
        max_length = pre_max + post_max + 1
        if max_length > 1:
            # compute a moving maximum
            max_origin = int(np.floor((pre_max - post_max) / 2))
            if activations.ndim == 1:
                filter_size = max_length
            elif activations.ndim == 2:
                filter_size = [max_length, 1]
            else:
                raise ValueError('`activations` must be either 1D or 2D')
            mov_max = maximum_filter(detections, filter_size, mode='constant',
                                     origin=max_origin)
            # detections are peak positions
            detections *= (detections == mov_max)
        # return indices
        if activations.ndim == 1:
            return np.nonzero(detections)[0]
        elif activations.ndim == 2:
            return np.nonzero(detections)
        else:
            raise ValueError('`activations` must be either 1D or 2D')
'''
