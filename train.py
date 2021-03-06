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

def main(custom_model_name = None):
    '''
    Main training function used to initiate training of the Drum-Tabber model. Tons of options are available in the configs.py file constants

    Args:
        custom_model_name [str]: Default None. If not None, forces the model after training to be saved under this custom name.

    Returns:
        None (Trains a model, writes to log that can be read by TensorBoard, saves the final model and configs dictionary describing it)
    '''

    # check configs compatibility with labeling window
    assert NEGATIVE_WINDOW_FRACTION + POSITIVE_WINDOW_FRACTION > HOP_SIZE/WINDOW_SIZE, 'Total labeling window fraction is not greater than HOP_SIZE/WINDOW_SIZE. Will not properly label data. Change WINDOW_FRACTION or HOP_SIZE and WINDOW_SIZE'


    # set GPU usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass

    # create logs directory and the tf.writer log
    logdir = os.path.join(TRAIN_LOGDIR, custom_model_name) if custom_model_name is not None else os.path.join(TRAIN_LOGDIR, 'unnamed_model')
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    train_writer = tf.summary.create_file_writer(os.path.join(logdir,'train'))
    val_writer = tf.summary.create_file_writer(os.path.join(logdir,'val'))

    if TRAIN_FULLSET_MEMORY:       # able to create and load the entire FullSet into memory
        FullSet = create_FullSet_df(SONGS_PATH)
        if CLEAN_DATA:
            FullSet = clean_labels(FullSet)
            MusicAlignedTab.labels_summary(FullSet)  # prints a labels summary out to screen
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
        for _, target, _2, _3 in dset:
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

        ''' NORMALIZE SPECTROGRAMS '''
        if model_type in ['TL-DenseNet121', 'TL-DenseNet169', 'TL-DenseNet201']:
            if INCLUDE_FO_DIFFERENTIAL:
                # TODO: normalize the top half of the n_features dimension. Bottom half is already normalized via the FO_DIFF concatenation step
                pass
            elif SHIFT_TO_DB:    # not including the fto differential, so just normalize the spectrogram directly if we shifted to db
                spectrogram = (spectrogram + 40)/40  # DenseNet requires "pixel values" [-1,1], and spectrogram was [-80,0]

        elif model_type in ['Context-CNN', 'TimeFreq-CNN']:
            if INCLUDE_FO_DIFFERENTIAL:
                # TODO: normalize the top half of the n_features dimension. Bottom half is already normalized via the FO_DIFF concatenation step
                pass
            elif SHIFT_TO_DB:
                spectrogram = (spectrogram + 80)/80   # Change to [0,1] values

        ''' RESHAPE TO INPUT ARRAY '''
        # TODO: Finish the other model type options when they become available
        if model_type in ['Context-CNN', 'TimeFreq-CNN', 'TL-DenseNet121', 'TL-DenseNet169', 'TL-DenseNet201']:

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

            # create the correct number of channels to be made for the input array stacking
            if model_type in ['Context-CNN', 'TimeFreq-CNN']:
                channel_number = 1
            elif model_type in ['TL-DenseNet121', 'TL-DenseNet169', 'TL-DenseNet201']:
                channel_number = 3  # simulating RGB color channels

            final_input_array = np.stack([input_array for _ in range(channel_number)], axis=-1)  # create the channel dimension for input

        return final_input_array

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
        if model_type in ['Context-CNN', 'TimeFreq-CNN',  'TL-DenseNet121', 'TL-DenseNet169', 'TL-DenseNet201']:
            target_array = target.T  # only need the transpose because the target array is 2D (n_classes, n_windows)
            # and we want (n_windows, n_classes)

        return target_array

    # loss and error metric computation FUNCTIONS
    def compute_error_metrics(peaks, label_ref_df, model_type, tolerance_window, hop_size, sr):
        '''
        Computes the various error metrics associated with the peaks against the original sample start assigned to each tab_df label example

        NOTE on variable name consistency: in this function I refer to "spectrogram windows" as "frames" so as not to be confused with
        "tolerance window" which refers to the amount of +- that a peak can be to satisfy a sample start label

        Args:
            peaks [np.array]: 0s and 1s array of shape (n_frames, n_classes) that has a detected peak where there is a 1 and not a peak at 0s
            label_ref_df [Dataframe]:
            model_type [str]:
            tolerance_window [int]:
            hope_size [int]:
            sr [int]:

        Returns:
            Dataframe: Dataframe of error metrics for the current song
        '''

        n_frames , n_classes = peaks.shape

        # TODO: fix the label_ref_df so that it "wraps around" properly like the batches do

        # create the error_df filled with 0s
        class_names = [x for x in list(label_ref_df.columns) if '_' in x]  # same code used to find the class names in configs_dict so should be consistent
        assert n_classes == len(class_names), 'For some reason the passed label_ref_df class is different than the number of classes in peaks'
        error_metrics_names = ['P', 'N', 'TP', 'TN', 'FP', 'FN', 'EX']   # error metrics and order
        error_df = pd.DataFrame(0, index = class_names, columns = error_metrics_names)  # Dataframe filled with 0s. Columns are error metrics. Rows are class names

        frame_starts = np.array([hop_size*idx for idx in range(n_frames)]) # calculates the frame starts in sample num of the frames for peaks array
        tol_window_sample = int((tolerance_window/1000.0) * sr) # converts tolerance_window (in unit of ms) to unit of sample number
        sample_starts = label_ref_df['sample start'].to_numpy(copy=True)
        num_rows_in_df = len(sample_starts)
        labels = label_ref_df[class_names].to_numpy(copy=True)   # gets a (num_rows_in_df, n_class) numpy array of 0s and 1s corresponding to correct labels

        # total positive and negative labels assignment into the error_df
        error_df['P'] = np.sum(labels==1, axis=0)
        error_df['N'] = np.sum(labels==0, axis=0)

        # create a np boolean mask array of shape ( num_rows_in_df, n_frames  ). This determines which frames to check from peaks, for each row in label_ref_df
        bool_mask = np.stack([ ((samp-tol_window_sample) <= frame_starts) & (frame_starts <= (samp+tol_window_sample)) for samp in sample_starts] , axis=0)

        for row_idx in range(num_rows_in_df):  # looping over each sample start (row) example, one at a time
            peaks_to_compare = peaks[bool_mask[row_idx,:],:]   # for this row example, here are the valid peaks with which to calculate error metrics
            bool_peaks_in_frames = np.any(peaks_to_compare==1, axis=0)    # finds if there is a peak, per class (returns a shape=(n_class,) array)
            row_labels = labels[row_idx,:]                               # getting this row's labels, containing all the classes
            error_df['TP'] += np.logical_and(row_labels==1, bool_peaks_in_frames)
            error_df['TN'] += np.logical_and(row_labels==0, np.logical_not(bool_peaks_in_frames))  # for all of these, take advantage of autocasting boolean
            error_df['FP'] += np.logical_and(row_labels==0, bool_peaks_in_frames)     # to int (false=0, true=1) when doing an operation
            error_df['FN'] += np.logical_and(row_labels==1, np.logical_not(bool_peaks_in_frames))

        # calculate the number of extraneous peaks that were unaccounted for in the peaks array
        # these peaks are located OUTSIDE of the tolerance window of ANY sample start num, and thus were never counted in TP or FN
        total_peaks_per_class = np.sum(peaks, axis=0)
        error_df['EX'] = total_peaks_per_class.astype(int) - error_df['P'].to_numpy(copy=True)

        return error_df

    def compute_loss(prediction, target_array, model_type, target_freq):
        '''
        Computes the loss for the model type given

        Args:
            prediction [tf.Tensor]:
            target_array [np.array]:
            model_type [str]:
            target_freq [np.array]:

        Returns:
            tf.Tensor: Tensor of the same shape as logits with component-wise losses calculated
        '''

        if model_type in ['Context-CNN', 'TimeFreq-CNN',  'TL-DenseNet121', 'TL-DenseNet169', 'TL-DenseNet201']:
            #losses = tf.nn.sigmoid_cross_entropy_with_logits(labels = target_array.astype(np.float32), logits = prediction)
            losses = tf.nn.weighted_cross_entropy_with_logits(labels = target_array.astype(np.float32),
                                                              logits = prediction,
                                                              pos_weight = 1/target_freq)

        # TODO: implement loss type for the RCNN type
        else:
            print("compute_loss: No other model type compute loss has been implemented")

        return losses

    # Train and Validation Step FUNCTION
    def song_step(spectrogram, target, label_ref_df, set_type, song_name):
        '''
        Controls both training and validation step in model training

        If training: updates the model from information of one song.
        Note that multiple model update steps can (and will) occur in one call of song_step

        If validation: skips the model updating part of the function

        Args:
            spectrogram [np.array]: for this song, the spectrogram array of shape (n_features, n_windows, n_channels)
            target [np.array]: for this song, the one-hot target array of shape (n_classes, n_windows, n_channels)
            label_ref_df [Dataframe]: for this song, a dataframe containing the labels and the 'sample start' column used to find accuracy
            set_type [str]: either 'train' or 'val' to determine which type of song_step to do
            song_name [str]: name of the song that the song step function is currently working on

        Returns:
            float: the global training step that we are currently on
            float: the learning rate after being updated by this training step
            float: the song loss of the song that was just processed
            DataFrame: the error dataframe that contains the tabulated True/False Positive/Negatives by class for this song
        '''

        # full spectrogram shape dimensions
        n, m, n_channels = spectrogram.shape

        class_names = [x for x in list(label_ref_df.columns) if '_' in x]

        if set_type == 'train':
            batch_size = TRAIN_BATCH_SIZE   # set correct batch size here for rest of function
            training_update = True          # set correct training update bool when feeding forward through the model
            writer_for_step = train_writer
        elif set_type =='val':
            batch_size = VAL_BATCH_SIZE
            training_update = False
            writer_for_step = val_writer

        song_loss = 0.0                       # the total running song loss for all classes
        song_loss_by_class = tf.zeros(shape=len(class_names))  # saves all the losses split out by class for entire song
        error_df_list = []

        # treat each channel individually as a single "song"
        for channel in range(n_channels):

            # converts the current spectrogram into the correct input array
            input_array = spectro_to_input_array(spectrogram[:,:,channel], MODEL_TYPE) # exactly the current output for input into model. Check function for dimensions of input_array
            target_array = target_to_target_array(target[:,:,channel], MODEL_TYPE)
            num_examples = input_array.shape[0]    # total number of examples in this song/channel

            # the number of model updates, based on the batch size and number of inputs
            num_updates = int(np.ceil(num_examples/batch_size))
            channel_loss = 0.0
            channel_loss_by_class = tf.zeros(shape=len(class_names))  # saves all the losses split out by class
            max_samples = int(num_updates*batch_size*HOP_SIZE)

            # making an empty array to concatenate later for building up the error metrics array after converting to peaks
            prediction_list = []

            # go through batches, make predictions, calculate losses, calculate gradients, and update the model
            for idx in range(num_updates):
                start_batch_slice = idx*batch_size
                end_batch_slice = (idx+1)*batch_size

                # start recording the functions that are applied to autodifferentiate later
                with tf.GradientTape() as tape:
                    if end_batch_slice > num_examples:  # in the case where we are in the last batch, so we concat end of input_array/target_array with beginning samples of remaining length
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
                        prediction = drum_tabber(np.concatenate( (input_array[start_batch_slice:, ...], input_array[ 0:end_batch_slice-num_examples , ...]) , axis=0), training = training_update)
                        losses = compute_loss(prediction, np.concatenate( (target_array[start_batch_slice:, :], target_array[0: end_batch_slice - num_examples, :])  , axis=0), MODEL_TYPE, target_freq)
                    else:
                        prediction = drum_tabber(input_array[ start_batch_slice : end_batch_slice , ...], training = training_update)   # the forward pass though the current model, with training = True
                        losses = compute_loss(prediction, target_array[start_batch_slice : end_batch_slice, :], MODEL_TYPE, target_freq)

                    batch_loss = tf.math.reduce_mean(losses)   # gets the average of all the classes, reduces down to scalar
                    batch_loss_by_class = tf.math.reduce_mean(losses, axis=0)  # gets the average of each class for the batch

                    if set_type == 'train':   # if we are in a train set, update the model
                        # apply gradients to update the model, the backward pass
                        gradients = tape.gradient(batch_loss, drum_tabber.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, drum_tabber.trainable_variables))

                prediction_list.append(prediction.numpy())
                channel_loss += batch_loss
                channel_loss_by_class += batch_loss_by_class

            full_chan_peaks = detect_peaks(np.concatenate(prediction_list, axis=0)) # concat the list of prediction arrays, and then feed it into detect_peaks function

            chan_error_df = compute_error_metrics(full_chan_peaks, label_ref_df, MODEL_TYPE, TOLERANCE_WINDOW, HOP_SIZE, SAMPLE_RATE)
            error_df_list.append(chan_error_df)

            channel_loss = channel_loss / num_updates                     # normalize the channel_loss by number of total batches done
            channel_loss_by_class = channel_loss_by_class / num_updates   # normalize by total batches done
            ''' THESE STEPS BELOW ARE SIGNIFICANT TIME SINKS DUE TO WRITING TO DISK. COMMENT THEM OUT FOR NOW
            # write a bunch of information to the current writer_for_step, including channel loss by class
            with writer_for_step.as_default():
                tf.summary.scalar('channel_loss/{}/{}/all_classes'.format(song_name,channel), channel_loss, step = epoch)
                for idx, class_name in enumerate(class_names):
                    tf.summary.scalar('channel_loss/{}/{}/{}'.format(song_name,channel,class_name), channel_loss_by_class[idx],  step = epoch)
            writer_for_step.flush()
            '''

            song_loss += channel_loss
            song_loss_by_class += channel_loss_by_class

        song_loss = song_loss/n_channels
        song_loss_by_class = song_loss_by_class/n_channels
        error_df = sum(error_df_list).copy() # the sum calls the + operator, which is overloaded for DataFrames to add element-wise values!

        # after the full song+channels is done, update learning rate, using warmup and cosine decay. Additionally, write summary data
        if set_type == 'train':
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = (global_steps / warmup_steps) * TRAIN_LR_INIT   # linearly increase lr until out of warmup steps
            else: # out of warmup epochs, so we use cosine decay for learning rate
                lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*(
                    (1 + tf.cos( ( (global_steps - warmup_steps) / (total_steps - warmup_steps) ) * np.pi)))
            optimizer.lr.assign(lr.numpy())
            # write summary data
            with train_writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("song_loss", song_loss, step=global_steps)
                tf.summary.scalar('song_loss/{}'.format(song_name), song_loss, step=epoch)
            train_writer.flush()

        return global_steps.numpy(), optimizer.lr.numpy(), song_loss.numpy(), error_df, song_loss_by_class.numpy()

    def display_error_metrics(error_df):
        '''
        Prints the error metrics to the screen from the error metrics in the Dataframe

        Args:
            error_df [Dataframe]: error metrics Dataframe; could be a summation of various songs or not

        Returns:
            None (prints to screen)
        '''
        acc = (error_df['TP'] + error_df['TN']) /  (error_df['TP'] + error_df['TN'] + error_df['FP'] + error_df['FN'])
        f1 = (2*error_df['TP']) / (2*error_df['TP'] + error_df['FP'] + error_df['FN'])
        print('Error_df: \n{}\n'.format(error_df))
        # print('Accuracy: \n{}\n'.format(acc))
        print('F1 Score: \n{}\n'.format(f1))

        return None

    best_val_loss = 1000.0    # start with a high validation loss
    best_val_loss_epoch = 0
    n_train_songs, n_val_songs = len(train_set), len(val_set)
    final_epoch_error_df_list = []

    # loop over the number of epochs
    for epoch in range(TRAIN_EPOCHS):
        print(f'Starting Epoch {epoch+1}/{TRAIN_EPOCHS}')
        train_songs_loss = 0.0
        for spectrogram, target, label_ref_df, song_name in train_set:   # outputs a full song's spectrogram and target and label reference df, over the entire dataset
            # do a train step with the current spectrogram and target
            glob_steps, current_lr, song_loss, error, sl_by_class = song_step(spectrogram, target, label_ref_df, train_set.set_type, song_name)
            current_step = (glob_steps % steps_per_epoch) if (glob_steps % steps_per_epoch) != 0 else steps_per_epoch # fixes the modulo returning 0 issue for display
            train_songs_loss += song_loss
            if epoch+1 == TRAIN_EPOCHS:   # in the last epoch, want to get the sum of all training error statistics.
                final_epoch_error_df_list.append(error)
            print('Epoch:{:2} Song{:3}/{}, lr:{:.6f}, song_loss:{:8.6f}, {}'.format(epoch+1, current_step, steps_per_epoch, current_lr, song_loss, song_name))
            if (glob_steps % 50) == 0:
                sl_by_class_dict = { configs_dict['class_names_dict'][idx] : sl_by_class[idx] for idx in range(sl_by_class.shape[0]) }
                print(f'Song Loss by class for {song_name}: {sl_by_class_dict}')

        total_val = 0.0
        val_error_list = []   # combining all the validation songs into one big error_metrics_df later
        for spectrogram, target, label_ref_df, song_name in val_set:
            # do a validation step with the current spectrogram and target
            _, _2, song_loss, error_, sl_by_class = song_step(spectrogram, target, label_ref_df, val_set.set_type, song_name)
            total_val += song_loss
            val_error_list.append(error_)
            if epoch+1 == TRAIN_EPOCHS:   # in the last epoch, want to get the sum of all training + validation error statistics.
                final_epoch_error_df_list.append(error_)
        print('\n\nEpoch: {:2} val_loss:{:8.6f} \n\n'.format(epoch+1, total_val/n_val_songs))
        # additional error metrics for the entire validation set (not normalized by the number of validation songs)
        if ((epoch % 5) == 0) or (epoch+1 == TRAIN_EPOCHS):
            val_error = sum(val_error_list)
            print('Validation error metrics:\n')
            display_error_metrics(val_error)

        with train_writer.as_default():
            tf.summary.scalar('epoch_loss/songs_loss', train_songs_loss/n_train_songs, step = epoch)
        train_writer.flush()
        with val_writer.as_default():
            tf.summary.scalar('epoch_loss/songs_loss', total_val/n_val_songs, step = epoch)
        val_writer.flush()

        # execute the saving model checkpoint options depending on configs
        if TRAIN_SAVE_CHECKPOINT_ALL:
            save_model_path = os.path.join(TRAIN_CHECKPOINTS_FOLDER, MODEL_TYPE + configs_dict['month_date']+"total_val_loss_{:8.6f}".format(total_val/n_val_songs))
            drum_tabber.save_weights(filepath=save_model_path, overwrite = True)
        if (total_val / n_val_songs) < best_val_loss:   # this epoch val_loss was the best so far
            best_val_loss = total_val / n_val_songs
            best_val_loss_epoch = epoch
            if TRAIN_SAVE_CHECKPOINT_MAX_BEST:
                save_model_path = os.path.join(TRAIN_CHECKPOINTS_FOLDER, MODEL_TYPE + configs_dict['month_date'])
                drum_tabber.save_weights(filepath=save_model_path, overwrite = True)
        if not TRAIN_SAVE_CHECKPOINT_ALL and not TRAIN_SAVE_CHECKPOINT_MAX_BEST:
            save_model_path = os.path.join(TRAIN_CHECKPOINTS_FOLDER, MODEL_TYPE + configs_dict['month_date'])
            drum_tabber.save_weights(filepath=save_model_path, overwrite = True)

    print('\nCongrats on making it through all {} training epochs!\n The best validation loss was {:8.6f} in epoch {}'.format(TRAIN_EPOCHS, best_val_loss, best_val_loss_epoch))
    print('Here is the final epoch error metrics for all songs are once:\n')
    final_epoch_error_df = sum(final_epoch_error_df_list)
    display_error_metrics(final_epoch_error_df)
    print('Saving the current drum_tabber model in memory and configs_dict to storage')

    saved_model_name = '{}-E{}-VL{:.5f}'.format(configs_dict['model_type'], TRAIN_EPOCHS, best_val_loss).replace('.','_')
    if custom_model_name != None:
        saved_model_name = custom_model_name
    save_drum_tabber_model(drum_tabber = drum_tabber, model_name = saved_model_name, saved_models_path = SAVED_MODELS_PATH, configs_dict = configs_dict)

    return None

if __name__ == '__main__':
    main()
