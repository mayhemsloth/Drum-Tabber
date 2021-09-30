#================================================================
#
#   File name   : timeseriestransformer.py
#   Author      : Thomas Hymel
#   Created date: 9-10-2021
#   GitHub      : https://github.com/mayhemsloth/Drum-Tabber
#   Description : class defintions for the time series transformer framework code
#
#================================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, MultiHeadAttention, Dropout

def create_mask(arr, r= 0.15, lm=4, mask_type="seq", random_type_list = None, all_batch_same=False):
    '''
    High level function that creates a mask for a batch of input sequences

    Args:
        arr [np.ndarray]: the batch array. Expected shape=(batch, features, len_seq)
        r [float]: Default 0.15. The overall percentage of masked values in each example
        lm [int]: Default 4. The mean masked length (lm = length_mask)
        mask_type [str]: Default 'seq'. Masking can be 'seq' (random time sequences), 'feature' (entire rows), 'time' (entire columns), 'forecast' (last columns), 'noise' (random throughout)
        random_type_list [list of str]: Default None. If NOT none, then the mask type will be randomly chosen from any valid mask type in this list
        all_batch_same [bool]: Default False. If True, the mask from the first example will override all other dimensions - thus all examples in the batch will have the same mask

    Returns:
        nd.array: an array of boolean masks of shape (batch, features, len_seq), where True means the value should be masked, and False means leave it alone
    '''

    assert 0 <= r <= 1, "Choose an appropriate masked rate, 0 <= r <=1"
    assert lm >=1 , "Choose an appropriate mean length of masked sequences, an integer >=1"

    if r == 0:   # case where no masking is requested, so there are no True values in the mask, regardless of requested masking type
        return np.zeros_like(arr, dtype=bool)

    batch_size, feature_size, len_seq = arr.shape
    rng = np.random.default_rng()   # for random number generation

    if random_type_list is not None:                # case where mask type is desired to be randomly determined
        list_of_mask_types = ['seq', 'feature', 'time', 'forecast', 'noise']
        valid_list = [mt for mt in random_type_list if mt in list_of_mask_types]   # preserves relative probabilities supplied by multiple instances of any valid mask type
        assert len(valid_list) > 0, "No valid mask type options in the passed list of mask types. Must contain at least one of ['seq', 'feature', 'time', 'forecast', 'noise']"
        mask_type = rng.choice(valid_list)  # choose one from the valid list of mask types

    if mask_type == "seq":    # case where mask will be sequences of 0s and 1s (as booleans)
    # Idea is to create a single list (of size mask_list_size) of sequences of 0s and 1s, and then reconstruct back into the correct size

        mask_list_size = batch_size*feature_size*len_seq    # total number of mask/unmask values to create
        lu = np.max((1, (lm)*(1-r)/r))                        # mean length of unmasked values, dictated by r and lm, max ensures lu >=1
        prob_m, prob_u = 1/lm, 1/lu                         # set up probability for Geometric sampling
        max_sampling_num = max(np.ceil(1.0), 2*np.ceil(mask_list_size // (lu+lm))).astype(np.int)     # number of times to sample, casted as an int

        for _ in range(5):  # ensures we randomly get enough length of sequences to cover all mask_list_size
            dist_mask = rng.geometric(prob_m, size=max_sampling_num)         # sample masked length distribution
            dist_unmasked = rng.geometric(prob_u, size=max_sampling_num)     # sample unmasked length distribution
            dist_summed = dist_mask + dist_unmasked  # creates summed "pairs" of un/masked sequence lengths
            if np.sum(dist_summed) > mask_list_size:  # successfully randomly sampled the appropriate length distrubitions
                break

        dist_index = np.argmax(np.cumsum(dist_summed, axis=0) > mask_list_size, axis=0).astype(int)+1    # finds the first index whose cumsum is greater than the numebr of values we need

        masked_unmasked_zipped = np.stack((dist_mask[:dist_index], dist_unmasked[:dist_index]), axis=-1) # zips the "pairs" together

        core_bool_array = np.tile([True, False], dist_index)   # [True, False] because masked sequences comes first always (based on order of mask, unmasked zipped)

        # repeat the bools core_bool_array the appropriate number of times to produce sequences of False/True
        mask_unraveled = np.repeat(core_bool_array, masked_unmasked_zipped.flatten())
        mask_unraveled = np.roll(mask_unraveled, rng.integers(0,mask_unraveled.shape[0] ))  # roll the list randomly so starting mask value is randomized

        mask = mask_unraveled[:mask_list_size].reshape(arr.shape)  # grab the correct number of values and reshape

    elif mask_type == "feature":  # case where entire feature rows get masked
    # Idea is to calculate the number of feature rows to mask, then randomly choose a subset from num of rows
        num_rows_to_mask = int(r*feature_size)       # based on r and number of total features there are
        mask = np.zeros_like(arr, dtype=bool)
        if num_rows_to_mask > 0:                     # if feature rows need to be masked
            row_indices = list(range(feature_size))
            for idx in range(batch_size):   # TODO: vectorize this shit. Super annoying I can't figure it out with indexing, but didn't want to spend more time on it
                rows_to_mask = rng.choice(a=row_indices, size=[num_rows_to_mask], replace=False)
                mask[idx, rows_to_mask, :] = True    # for this idx example in batch, set True to only those rows randomly selected, for all columns (:)

    elif mask_type == "time":    # case where random time columns are masked
    # Idea is similar to feature mask production, calculate the number of time columns to mask, then randomly choose a subset from num of rows
        num_cols_to_mask = int(r*len_seq)
        mask = np.zeros_like(arr, dtype=bool)
        if num_cols_to_mask > 0:
            col_indices = list(range(len_seq))
            for idx in range(batch_size):   # TODO: vectorize this shit.
                cols_to_mask = rng.choice(a=col_indices, size=[num_cols_to_mask], replace=False)
                mask[idx, :, cols_to_mask] = True   # for this idx example in batch, set True to only those cols randomly selected, for all rows (:)

    elif mask_type == "forecast":   # case where the final ~r column values (ie, the "future") are masked
        mask = np.zeros_like(arr, dtype=bool)
        for idx in range(batch_size):
            mask_subbatch = rng.random(size=[feature_size*len_seq]) < r     # produces a boolean array with r proportion of True values
            mask_subbatch = np.reshape(np.sort(mask_subbatch), newshape=(len_seq, feature_size)).T  # can probably do this better but I don't know how
            mask[idx] = mask_subbatch

    elif mask_type == "noise":       # classic random noise mask - equivalent to a Dropout layer
        mask = rng.random(size=arr.shape) < r

    else:
        print(f'{mask_type} mask type is not valid, returning a mask with no masked values')
        mask = np.zeros_like(arr, dtype=bool)

    if all_batch_same:    # option for the entire batch to have the same mask
        mask = np.stack([mask[0] for _ in range(batch_size)], axis=0)   # first example mask replaces all others (each mask is random anyway, first isn't special))

    return mask


class TimeSeriesTransformer(tf.keras.Model):
    def __init__(self, d_features_in, d_out, len_seq, d_model, n_heads=4, n_encoder_layers=3,
                 d_ffn=256, activ = 'gelu', mha_bias = False, attention_dropout_p = 0.1, ffn_dropout_p = 0.1,
                 self_supervised_training = False, output_type = 'regr', custom_head = None
                 ):
        '''
        TimeSeriesTransformer is a transformer-based framework that takes in time series inputs and
        can perform either regression or classification problems based on self-supervised and
        fine-tune supervised training.

        Args:
            d_features_in [int]: the number of features in the time series inputs (assumes the time series is a 1D vector!)
            d_out [int]: the number of target classes or values to predict
            len_seq [int]: the length of the time series data sequence given to the model
            d_model [int]: core dimension of the model: number of learned features at each attention layer
            n_heads [int]: number of heads in each of the multi-head attention layers
            n_encoder_layers [int]: number of encoder layer blocks that make up the entire transformer encoder section
            d_ffn [int]: dimension of the feed forward network inside the encoder layer
            activ [str]: activation function used in the transformer network. Tensorflow activation functions only
            mha_bias [bool]: If True, include the bias term in the multi-head attention layer
            attention_dropout_p [float]: dropout probability for the attention layer
            ffn_dropout_p [float]: dropout probability for the feed forward network layer
            self_supervised_training [bool]: Default False. If True, in self-supervise training regime. If False, in fine-tune regime.
            output_type [str]: Default 'regr'. Determines the output for the fine-tune layer
            custom_head [tf.layer]: Default None. List of tf.layers to be used as the custom fine tune head

        Returns:
            TimeSeriesTransformer: a custom tf.keras.Model class
        '''

        super(TimeSeriesTransformer, self).__init__()
        self.self_supervised_training = self_supervised_training
        self.output_type = output_type

        # layer objects
        self.embed_layer = EmbeddingLayer(d_features_in, len_seq, d_model)  # learnable Embedding Layer

        self.pe_layer = PositionalEncodingLayer(d_model, len_seq)  # learnable positional encoding layer

        self.encoder = TransformerEncoder(d_model, len_seq, n_encoder_layers = n_encoder_layers, n_heads=n_heads, d_ffn=d_ffn, activ = activ, mha_bias = mha_bias,
                  attention_dropout_p = attention_dropout_p, ffn_dropout_p = ffn_dropout_p)

        self.self_supervised_head = SelfSupervisedLayer(d_features_in, len_seq, d_model)

        self.fine_tune_head = self.create_fine_tune_layer(output_type, d_out, custom_head)

    def create_fine_tune_layer(self, output_type, d_out, custom_head = None):
        '''
        Helper function to create the fine tune layer/head

        Args:
            output_type [str]: One of the following options:
                'regr' - for regression problem, predicting values (single or multi-variable output)
                'softmax' - for single label classification (single or multi-class)
                'multilabel' - for multi-label classification (multi-class necessarily)
            d_out [int]: the number of target classes or values to predict
            custom_head [tf.layer]: Default None. Allows a list of layers to be passed in to be used as the final head for prediction

        Returns:
            list of tf.layer: the appropriate stack of layers for the desired problem type to affix onto the last layer

        '''

        if custom_head is not None:  # case where a custom head (stack of layers) is passed to be used as the custom head
            fine_tune_layers = custom_head

        else:     # standard Dense layer is used to convert from flattened encoder feature representation to final predictions
            fine_tune_layers = [Flatten()]  # always need a flatten layer initially
            if output_type == 'regr':         dense_activ = None    # no activation because we want values directly
            elif output_type == 'softmax':    dense_activ = 'softmax'    # force exclusive probabilities
            elif output_type == 'multilabel': dense_activ = 'sigmoid'    # guess a probability individually for each class
            fine_tune_layers.append(Dense(d_out, activation = dense_activ))      # Dense layer with d_out classes, activated by the appropriate function

        return fine_tune_layers

    def call(self, sequence):
        '''
        Forward pass of a sequence through the entire model

        Args:
            sequence [np.ndarray]: Raw input sequence of shape=(batch, features, time step)
        Returns:
            tf.array: output sequence
        '''

        # Embedding layer transformation
        output = self.embed_layer(sequence)

        # Positional Encoding
        output = self.pe_layer(output)

        # Encoder
        output = self.encoder(output)

        # Final Head Layer
        if self.self_supervised_training:    # self-supervised training regime
            output = self.self_supervised_head(output)
            output = tf.transpose(output, perm=(0, 2, 1))  # transpose back to shape=(batch, features, time step)
        else:      # in fine-tuning regime
            for ft_layer in self.fine_tune_head:
                output = ft_layer(output)        # calling the fine tune layers on output transforms it into a prediction

        return output

    def build_graph(self, raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, d_features_in, len_seq, d_model, activ=None):
        '''
        The linear layer responsible for taking in the data (of dimensions d_features_in) and
        casting it to the model's core dimension space
        TODO: Functionality to include a different type of EmbeddingLayer - like ROCKET
        derived features, or time dilating a longer sequence via 1D Convs to shorter time sequence

        Args:
            d_features_in [int]: the number of features in the time series inputs (assumes the time series is a 1D vector!)
            len_seq [int]: the length of the time series data sequence given to the model
            d_model [int]: core dimension of the model: number learned features at each attention layer
            activ [str]: Default 'none', activation used for the Embedding layer (none for linear)

        '''
        super(EmbeddingLayer, self).__init__()
        self.model_size = d_model
        self.input_features_size = d_features_in
        self.input_len = len_seq

        # main layer, expected input shape for embedding = (batch, features, time steps)
        # will transpose input during call step due to how Dense works: dots across the last dimension
        self.embed = Dense(self.model_size, input_shape = (self.input_len, self.input_features_size), activation = activ) # None activation is linear matrix

    def call(self, sequence):
        '''
        Args:
            sequence [np.ndarray] shape=(batch, features, time step): Raw input sequence
        Returns:
            tf.array, shape=(batch, time step, model_size_features): the embedded sequence with same number of time steps as input sequence, but different feature space dimension
        '''
        return self.embed(tf.transpose(sequence, perm=(0, 2, 1)) )

class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, len_seq):
        '''
        Layer responsible for learning the positional encoding

        Args:
        '''
        super(PositionalEncodingLayer, self).__init__()
        self.model_size = d_model
        self.input_len = len_seq

        # pe initialization is random.normal with mean = 0.0, stddev = 0.2
        self.pe = tf.Variable(initial_value = tf.random.normal((self.input_len, self.model_size), mean=0.0, stddev=0.2),
                              trainable = True, name = 'Positional_Encoding_weights')

    def call(self, sequence):
        '''
        Adds together the trainable positonal encoding layer and the sequence.

        Args:
            sequence [tf.array] shape=(batch, time step, d_model): EmbeddingLayer output

        Returns:
            tf.array: the embed layer added to the positional encoding layer
        '''

        # note that tf.add automatically casts in the correct dimension despite not having a batch dimension
        # Don't quite understand how it figures it out but it seems to work fine
        return tf.math.add(sequence, self.pe)

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, len_seq, n_heads=4, d_ffn=256, activ = 'gelu', mha_bias = False,
                  attention_dropout_p = 0.1, ffn_dropout_p = 0.1):
        super(TransformerEncoderLayer, self).__init__()

        assert (d_model % n_heads) == 0, f'Model dimension (d_model = {d_model}) must be divisible by number of heads (n_heads = {n_heads})'

        # using the MultiHeadAttention layer from TF Keras
        self.self_attention = MultiHeadAttention(num_heads=n_heads, key_dim=d_model, value_dim=d_model, dropout = attention_dropout_p, use_bias=mha_bias)

        # Attention residual connection - Add and Normalize layers
        self.attention_dropout = Dropout(attention_dropout_p)
        self.attention_batchnorm = BatchNormalization()

        # Feed-Forward layers
        self.ffn1 = Dense(d_ffn, activation = activ)  # expand out to d_ffn number of units
        self.ffn2 = Dense(d_model)                    # contract back to d_model number of units

        # Feed-Forward residual connection - Add and Normalize layers
        self.ffn_dropout = Dropout(ffn_dropout_p)
        self.ffn_batchnorm = BatchNormalization()

    def call(self, sequence, attention_mask=None):
        '''
        Applies one forward pass of the Transformer Encoder layer

        Args:
            sequence [tf.array] shape=(batch, time_step, d_model): Output of positional encoding, or a previous Transformer Encoder Layer
            attention_mask [tf.array] shape=(batch, time_step, time_step) or (time_step, time_step): mask to hide future self-attention (if applicable)

        '''
        # Calculate attention output and scores
        attention_output, attention_scores = self.self_attention(sequence, sequence, return_attention_scores = True)

        # Attention residual connection (dropout of output, then add original, then batchnorm)
        sequence = self.attention_batchnorm(tf.math.add(sequence, self.attention_dropout(attention_output)))

        # Calculate feed-forward using feed-forward layers ()
        ff_output = self.ffn2(self.ffn1(sequence))

        # Feed-forward residual connection (dropout of output, then add original, then batchnorm)
        sequence = self.ffn_batchnorm(tf.math.add(sequence, self.ffn_dropout(ff_output)))

        return sequence # shape=(batch, time_step, d_model)

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,  d_model, len_seq, n_encoder_layers = 3, n_heads=4, d_ffn=256, activ = 'gelu', mha_bias = False,
                  attention_dropout_p = 0.1, ffn_dropout_p = 0.1):
        '''
        Mainly a wrapper for the desired number of encoder layers in the transformer architecture
        '''
        super(TransformerEncoder, self).__init__()

        # makes a list of the desired number of encoder layers
        self.encoder_layers = [TransformerEncoderLayer(d_model=d_model, len_seq=len_seq, n_heads=n_heads,
                                                       d_ffn=d_ffn, activ = activ, mha_bias = mha_bias,
                                                        attention_dropout_p = attention_dropout_p,
                                                       ffn_dropout_p = ffn_dropout_p) for _ in range(n_encoder_layers)]

    def call(self, sequence, attention_mask = None):
        '''
        Applies sequential forward passes through all transformer encoder layers

        Args:
            sequence [tf.array] shape=(batch, time_step, d_model): Output of positional encoding, or a previous Transformer Encoder Layer
            attention_mask [tf.array] shape=(batch, time_step, time_step) or (time_step, time_step): mask to hide future self-attention (if applicable)

        '''
        # TODO utilize the attention mask
        temp = sequence
        for encoder_layer in self.encoder_layers:    # process layer by layer
            temp = encoder_layer(temp, attention_mask = attention_mask)
        return temp

class SelfSupervisedLayer(tf.keras.layers.Layer):
    def __init__(self, d_features_in, len_seq, d_model, activ=None):
        '''
        The default head layer attached for self-supervised layer. Mainly a wrapper for a linear layer

        Args:
            d_features_in [int]: the number of features in the time series inputs (assumes the time series is a 1D vector!)
            len_seq [int]: the length of the time series data sequence given to the model
            d_model [int]: core dimension of the model: number learned features at each attention layer
            activ [str]: Default 'none', activation used for the self supervised layer (none for linear)
        '''
        super(SelfSupervisedLayer, self).__init__()

        self.self_supervised_layer = Dense(d_features_in, activation = activ)

    def call(self, sequence):
        '''
        Applies the self supervised layer forward pass
        '''

        return self.self_supervised_layer(sequence)
