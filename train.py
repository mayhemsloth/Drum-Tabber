#================================================================
#
#   File name   : train.py
#   Author      : Thomas Hymel
#   Created date: 9-2-2020
#   GitHub      : https://github.com/mayhemsloth/Drum-Tabber
#   Description : Main python file to call to train an automatic drum tabber model
#
#================================================================

import os
import shutil
import random
import numpy as np
import pandas as pd
import librosa as lb
import tensorflow as tf

from src.configs import *
from src.dataset import Dataset
from src.utils import MusicAlignedTab, create_FullSet_df, clean_labels, collapse_class, one_hot_encode, create_configs_dict
from src.utils import save_drum_tabber_model, detect_peaks
from src.model import create_DrumTabber

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    '''
    Main training function used to initiate training of the Drum-Tabber model
    '''

    # set GPU usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass

    # create logs directory and the tf.writer log
    if os.path.exists(TRAIN_LOGDIR):
        shutil.rmtree(TRAIN_LOGDIR)
    writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    val_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

    if TRAIN_FULLSET_MEMORY:       # able to create and load the entire FullSet into memory
        FullSet = create_FullSet_df(SONGS_PATH)
        if CLEAN_DATA:
            FullSet = clean_labels(FullSet)
            MusicAlignedTab.labels_summary(FullSet)
        FullSet = collapse_class(FullSet_df = FullSet,
                                    keep_dynamics = KEEP_DYNAMICS,
                                    keep_bells = KEEP_BELLS,
                                    keep_toms_separate = KEEP_TOMS_SEPARATE,
                                    hihat_classes = HIHAT_CLASSES,
                                    cymbal_classes = CYMBAL_CLASSES)
        MusicAlignedTab.labels_summary(FullSet)   # prints a labels summary out to screen
        FullSet_encoded = one_hot_encode(FullSet)
        configs_dict = create_configs_dict(FullSet_encoded)
        print('train.py main(): FullSet_encoded created!')
    else:
        FullSet_encoded = None

    # create the iterable Dataset objects for training
    train_set = Dataset('train', FullSet_encoded)
    val_set = Dataset('val', FullSet_encoded)

    # determine the weights of the different class labels to be put into the weighted_cross_entropy loss function
    target_counts = np.zeros(shape=(configs_dict['num_classes']), dtype=np.float32)
    total_windows = 0
    for dset in [train_set, val_set]:
        for _, target in dset:
            n_class, n_window, n_channel = target.shape   # target.shape = (n_classes, n_windows, n_channels)
            target_counts += np.sum(np.count_nonzero(target, axis=1), axis=1)   # sums all nonzero_counts across all windows and channels
            total_windows += n_window*n_channel
    target_freq = target_counts/total_windows    # the percentage of total counts for each class in each window
    print('train.py main(): ', target_freq)

    # epochs/steps variables (note that a "step" is currently setup as one song)
    steps_per_epoch = len(train_set) # how many songs there are in the training set
    global_steps = tf.Variable(0, trainable=False, dtype=tf.int64)   # specifically used as args for later tf functions
    warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
    total_steps = TRAIN_EPOCHS * steps_per_epoch

    # load the model to be trained, based on configs.py options
    # TODO: code the create_DrumTabber function, which initializes to randomized weights
    drum_tabber = create_DrumTabber(n_features = configs_dict['num_features'],
                                    n_classes = configs_dict['num_classes'],
                                    activ = 'relu',
                                    training = True)  # initial randomized weights of a keras model
    print('train.py main(): drum_tabber model created!')
    print('train.py main(): ', drum_tabber.summary())

    if TRAIN_FROM_CHECKPOINT and len(TRAIN_CHECKPOINT_MODEL_NAME) != 0:
        try:
            drum_tabber.load_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_CHECKPOINT_MODEL_NAME))
        except ValueError:
            print("Shapes are incompatible, using default initial randomized weights")

    # load the optimizer, using Adam
    optimizer = tf.keras.optimizers.Adam()
    if warmup_steps != 0:
        optimizer.lr.assign( ( ( (global_steps+1) / warmup_steps) * TRAIN_LR_INIT ).numpy())
    else:
        optimizer.lr.assign( TRAIN_LR_INIT )

    # Model Input and Target massaging FUNCTIONS
    def spectro_to_input_array(spectrogram, model_type):
        '''
        Expands a 2D song's channel's spectrogram into slices of the correct shape to be input into the model

        Args:
            spectrogram [np.array]: 2D spectrogram of the current song and channel
            model_type [str]:

        Returns:
            np.array: 3D spectrogram object of the entire song with batch size in the first dimension. e.g., input_array[0,:,:] is the entire first input (a 2D array)
        '''

        n_features, n_windows = spectrogram.shape

        # TODO: Finish the other model type options when they become available
        if model_type == 'Context-CNN':

            pre_context, post_context = N_CONTEXT_PRE, N_CONTEXT_POST
            input_width = pre_context + 1 + post_context
            min_value = np.min(spectrogram)

            # assign into this np.array filled with the min values of the spectrogram (silence)
            input_array = np.full(shape = (n_windows, n_features, input_width), fill_value = min_value)

            for idx in range(n_windows):
                if idx - pre_context < 0:    # in a window where you would slice before the beginning
                    start = pre_context-idx
                    input_array[idx, :, start:] = spectrogram[:, 0:idx+post_context+1]
                elif idx + post_context+1 > n_windows: # in a window where you would slice past the end
                    end = post_context+1 - (n_windows - idx)
                    input_array[idx, :, :input_width-end] = spectrogram[:, idx-pre_context: n_windows ]
                else:    # in a "normal" middle window where you slice into the spectrogram normally
                    input_array[idx, :,:] = spectrogram[:, idx-pre_context : idx+post_context+1]

            return input_array

    def target_to_target_array(target, model_type):
        '''
        Expands a 2D target array into slices of the correct shape to be the output of the model

        Args:
            target [np.array]: for this song and one channel, the one-hot target array of shape (n_classes, n_windows)
            model_type [str]:

        Returns:
            np.array: 2D spectrogram object of the entire targets with batch size in the first dimension. e.g., target_array[0,:] is the entire first target (a 1D array of the one-hot-encoded classes)
        '''

        n_classes, n_windows = target.shape

        # TODO: Finish the other model type options when they become available
        if model_type == 'Context-CNN':
            target_array = target.T  # only need the transpose because the target array is 2D (n_classes, n_windows)
            # and we want (n_windows, n_classes)

        return target_array

    def compute_loss(prediction, target_array, model_type, target_freq):
        '''
        Computes the loss for the model type given

        Args:
            prediction [np.array]:
            target_array [np.array]:
            model_type [str]:
            target_freq [np.array]:

        Returns:
            tf.Tensor: Tensor of the same shape as logits with component-wise losses calculated
        '''

        if model_type == 'Context-CNN':
            # losses = tf.nn.sigmoid_cross_entropy_with_logits(labels = target_array.astype(np.float32), logits = prediction)
            losses = tf.nn.weighted_cross_entropy_with_logits(labels = target_array.astype(np.float32),
                                                              logits = prediction,
                                                              pos_weight = 1/target_freq)

        # TODO: implement loss type for the RCNN type
        else:
            print("compute_loss: No other model type compute loss has been implemented")

        return losses

    # Train and Validation Step FUNCTIONS
    # TODO: Combine these two functions into one and then send in a training/val flag for the differentials???
    def train_song_step(spectrogram, target):
        '''
        Updates the model from the information of one song.
        Note that multiple model update steps can (and will) occur in one call of train_song_step

        Args:
            spectrogram [np.array]: for this song, the spectrogram array of shape (n_features, n_windows, n_channels)
            target [np.array]: for this song, the one-hot target array of shape (n_classes, n_windows, n_channels)

        Returns:
            variables about the training step (to be displayed)
        '''

        # full spectrogram shape dimensions
        n, m, n_channels = spectrogram.shape

        song_loss = 0

        # treat each channel individually as a single "song"
        for channel in range(n_channels):

            # converts the current spectrogram into the correct input array
            input_array = np.expand_dims(spectro_to_input_array(spectrogram[:,:,channel], MODEL_TYPE), axis=-1) # adding a channel dim at the end so that it is 4D for the model input
            target_array = target_to_target_array(target[:,:,channel], MODEL_TYPE)
            num_examples = input_array.shape[0]    # total number of examples in this song/channel

            # the number of model updates, based on the batch size and number of inputs
            num_updates = int(np.ceil(num_examples/TRAIN_BATCH_SIZE))
            channel_loss = 0

            for idx in range(num_updates):
                total_loss = 0
                start_batch_slice = idx*TRAIN_BATCH_SIZE
                end_batch_slice = (idx+1)*TRAIN_BATCH_SIZE

                # start recording the functions that are applied to autodifferentiate later
                with tf.GradientTape() as tape:
                    if end_batch_slice > num_examples:  # in the case where we are in the last batch, so we concat end of input_array/target_array with beginning samples of remaining length
                        prediction = drum_tabber(np.concatenate( (input_array[start_batch_slice:, ...], input_array[ 0:end_batch_slice-num_examples , ...]) , axis=0), training = True)
                        '''
                        NOTE ON THE RAMIFICATIONS OF THIS DECISION ON HOW TO SUPPLY ADDITIONAL SAMPLES FOR CORRECT BATCH SIZE:
                        To correct the batch size for the end samples, I chose to append the start of the song samples onto the end to correct the batch number size
                        The "beginning" of each song will be oversampled one additional time for every epoch that occurs.
                        However, due to data augmentation, the "beginning" of each song will be randomly chosen at SHIFT_CHANCE probability
                        For this reason I feel that the model won't be trained on the exact same oversampled data too much, and
                        thus why I decided to code it this way instead of some other way (randomly taking samples).
                        Additionally, when I implement the Recurrent NN part, the order of the samples matter. This method
                        preserves relative time order between the different samples.
                        '''
                        losses = compute_loss(prediction, np.concatenate( (target_array[start_batch_slice:, :], target_array[0: end_batch_slice - num_examples, :])  , axis=0), MODEL_TYPE, target_freq)
                    else:
                        prediction = drum_tabber(input_array[ start_batch_slice : end_batch_slice , ...], training = True)   # the forward pass though the current model, with training = True
                        losses = compute_loss(prediction, target_array[start_batch_slice : end_batch_slice, :], MODEL_TYPE, target_freq)

                    total_loss += tf.math.reduce_mean(losses)   # gets the average of all the classes
                    # apply gradients to update the model, the backward pass
                    gradients = tape.gradient(total_loss, drum_tabber.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, drum_tabber.trainable_variables))
                channel_loss += total_loss

            channel_loss = channel_loss / num_updates
            song_loss += channel_loss

        # after the full song is done, update learning rate, using warmup and cosine decay
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = (global_steps / warmup_steps) * TRAIN_LR_INIT   # linearly increase lr until out of warmup steps
        else: # out of warmup epochs, so we use cosine decay for learning rate
            lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*(
                (1 + tf.cos( ( (global_steps - warmup_steps) / (total_steps - warmup_steps) ) * np.pi)))
        optimizer.lr.assign(lr.numpy())

        # write summary data
        # TODO: understand what the writer TF is doing
        '''
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", loss, step=global_steps)
        writer.flush()
        '''

        song_loss = song_loss/n_channels

        return global_steps.numpy(), optimizer.lr.numpy(), song_loss.numpy()

    def validation_song_step(spectrogram, target):
        '''
        Calculates losses for one song in the validation set to see if the model got better during this training epoch

        Args:
            spectrogram [np.array]: for this song, the spectrogram array of shape (n_features, n_windows, n_channels)
            target [np.array]: for this song, the one-hot target array of shape (n_classes, n_windows, n_channels)

        Returns:
            variables about the training step (to be displayed)
        '''

        # full spectrogram shape dimensions
        n, m, n_channels = spectrogram.shape

        song_loss = 0

        # treat each channel individually as a single "song"
        for channel in range(n_channels):

            # converts the current spectrogram into the correct input array
            input_array = np.expand_dims(spectro_to_input_array(spectrogram[:,:,channel], MODEL_TYPE), axis=-1)
            target_array = target_to_target_array(target[:,:,channel], MODEL_TYPE)
            num_examples = input_array.shape[0]

            # the number of model updates, based on the batch size and number of inputs
            num_updates = int(np.ceil(num_examples/VAL_BATCH_SIZE))
            channel_loss = 0

            for idx in range(num_updates):
                total_loss = 0
                start_batch_slice = idx*VAL_BATCH_SIZE
                end_batch_slice = (idx+1)*VAL_BATCH_SIZE

                # start recording the functions that are applied to autodifferentiate
                with tf.GradientTape() as tape:
                    if end_batch_slice > num_examples:  # in the case where we are in the last batch, so we concat end of input_array/target_array with beginning samples of remaining length
                        prediction = drum_tabber(np.concatenate( (input_array[start_batch_slice:, :, :], input_array[ 0:end_batch_slice-num_examples , :, :]) , axis=0), training = False)
                        losses = compute_loss(prediction, np.concatenate( (target_array[start_batch_slice:, :], target_array[0: end_batch_slice - num_examples, :])  , axis=0), MODEL_TYPE, target_freq)
                    else:
                        prediction = drum_tabber(input_array[ start_batch_slice : end_batch_slice , :, :], training = False)   # the forward pass though the current model, with training = True
                        losses = compute_loss(prediction, target_array[start_batch_slice : end_batch_slice, :], MODEL_TYPE, target_freq)

                total_loss += tf.math.reduce_mean(losses)   # gets the average of all the classes
                channel_loss += total_loss

            # after each channel, print the total channel mean loss
            # print(f'Channel {channel} loss: {channel_loss/num_updates}')
            channel_loss = channel_loss / num_updates
            song_loss += channel_loss

        song_loss = song_loss / n_channels

        return song_loss.numpy()

    best_val_loss = 1000.0    # start with a high validation loss
    n_val_songs = len(val_set)

    # loop over the number of epochs
    for epoch in range(TRAIN_EPOCHS):
        print(f'Starting Epoch {epoch+1}/{TRAIN_EPOCHS}')
        for spectrogram, target in train_set:   # outputs a full song's spectrogram and target, over the entire dataset
            # do a train step with the current spectrogram and target
            loss_results = train_song_step(spectrogram, target)
            current_step = (loss_results[0] % steps_per_epoch)
            if current_step == 0 : current_step = steps_per_epoch  # fixes the modulo returning 0 issue for display
            print('Epoch:{:2} Song{:3}/{}, lr:{:.6f}, song_loss:{:8.6f}'.format(epoch+1, current_step, steps_per_epoch, loss_results[1], loss_results[2]))

        total_val = 0
        for spectrogram, target in val_set:
            # do a validation step with the current spectrogram and target
            results = validation_song_step(spectrogram, target)
            total_val += results
        print('\n\nEpoch: {:2} val_loss:{:8.6f} \n\n'.format(epoch+1, total_val/n_val_songs))

        if TRAIN_SAVE_CHECKPOINT_ALL_BEST:
            save_model_path = os.path.join(TRAIN_CHECKPOINTS_FOLDER, MODEL_TYPE + configs_dict['month_date']+"total_val_loss_{:8.6f}".format(total_val/n_val_songs))
            drum_tabber.save_weights(filepath=save_model_path, overwrite = True)
        if TRAIN_SAVE_CHECKPOINT_MAX_BEST and (total_val / n_val_songs) < best_val_loss:
            save_model_path = os.path.join(TRAIN_CHECKPOINTS_FOLDER, MODEL_TYPE + configs_dict['month_date'])
            best_val_loss = total_val / n_val_songs
            drum_tabber.save_weights(filepath=save_model_path, overwrite = True)
        if not TRAIN_SAVE_CHECKPOINT_ALL_BEST and not TRAIN_SAVE_CHECKPOINT_MAX_BEST:
            save_model_path = os.path.join(TRAIN_CHECKPOINTS_FOLDER, MODEL_TYPE + configs_dict['month_date'])
            drum_tabber.save_weights(filepath=save_model_path, overwrite = True)

    print('Congrats on making it through all training epochs!')
    print('Saving the current drum_tabber model in memory and configs_dict to storage')
    saved_model_name = '{}-E{}-VL{:.5f}'.format(configs_dict['model_type'], TRAIN_EPOCHS, best_val_loss).replace('.','_')
    save_drum_tabber_model(drum_tabber = drum_tabber, model_name = saved_model_name, saved_models_path = SAVED_MODELS_PATH, configs_dict = configs_dict)

    return None

if __name__ == '__main__':
    main()
