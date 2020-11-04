#================================================================
#
#   File name   : train.py
#   Author      : Thomas Hymel
#   Created date: 9-2-2020
#   GitHub      : https://github.com/mayhemsloth/Drum-Tabber
#   Description : Main python file to call to train an automatic drum tabber model
#
#================================================================
'''
: import all the dependency functions and modules here.
: If you import the configs file you get access to all the variables in there and the functions can use it
'''
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
        configs_dict = create_configs_dict(FullSet_encoded, TRAIN_CONFIGS_SAVE_PATH)
    else:
        FullSet_encoded = None

    # create the iterable Dataset objects for training
    train_set = Dataset('train', FullSet_encoded)
    val_set = Dataset('val', FullSet_encoded)

    # epochs/steps variables (note that a "step" is currently setup as one song)
    steps_per_epoch = len(train_set) # how many songs there are in the training set
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)   # specifically used as args for later tf functions
    warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
    total_steps = TRAIN_EPOCHS * steps_per_epoch

    # load the model to be trained, based on configs.py options
    # TODO: code the create_DrumTabber function, which initializes to randomized weights
    drum_tabber = create_DrumTabber(n_features = configs_dict['num_features'],
                                    n_classes = configs_dict['num_classes'],
                                    training = True)  # initial randomized weights of a tf/keras model

    if TRAIN_FROM_CHECKPOINT:
        try:
            drum_tabber.load_weights('path_to_training_checkpoints_and_model_name')
        except ValueError:
            print("Shapes are incompatible, using default initial randomized weights")

    # load the optimizer, using Adam
    optimizer = tf.keras.optimizers.Adam()
    optimizer.lr.assign( ( (global_steps / warmup_steps) * TRAIN_LR_INIT ).numpy())

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


    # Train and Validation Step FUNCTIONS
    # TODO: Finish coding the train and validation step functions
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

        # treat each channel individually as a single "song"
        for channel in range(n_channels):

            # converts the current spectrogram into the correct input array
            input_array = np.expand_dims(spectro_to_input_array(spectrogram[:,:,channel], MODEL_TYPE), axis=-1) # adding a channel dim at the end so that it is 4D for the model input
            target_array = np.expand_dims(target_to_target_array(target[:,:,channel], MODEL_TYPE), axis=-1)  # adding a channel dim at the end so that it is 4D for the model input
            num_examples = input_array.shape[0]    # total number of examples in this song/channel

            # the number of model updates, based on the batch size and number of inputs
            num_updates = int(np.ceil(num_examples/TRAIN_BATCH_SIZE))

            for idx in range(num_updates):
                losses = 0
                start_batch_slice = idx*TRAIN_BATCH_SIZE
                end_batch_slice = (idx+1)*TRAIN_BATCH_SIZE

                # start recording the functions that are applied to autodifferentiate later
                with tf.GradientTape() as tape:
                    if end_batch_slice > num_examples:  # in the case where we are in the last batch, so we concat end of input_array/target_array with beginning samples of remaining length
                        prediction = drum_tabber(np.concatenate( (input_array[start_batch_slice:, ...], input_array[ 0:end_batch_slice-num_examples , ...]) , axis=0), training = True)
                        '''
                        NOTE ON THIS DECISION AND RAMIFICATIONS ON HOW TO SUPPLY ADDITIONAL SAMPLES FOR CORRECT BATCH SIZE:
                        To correct the batch size for the end samples, I chose to append the start of the song samples onto the end to correct the batch number size
                        The "beginning" of each song will be oversampled one additional time for every epoch that occurs.
                        However, due to data augmentation, the "beginning" of each song will be randomly chosen at SHIFT_CHANCE probability
                        For this reason I feel that the model won't be trained on the exact same oversampled data too much, and
                        thus why I decided to code it this way instead of some other way (randomly taking samples).
                        Additionally, when I implement the Recurrent NN part, the order of the samples matter. This method
                        preserves relative time order between the different samples.
                        '''
                        # TODO: code the compute_loss function
                        # losses = compute_loss(prediction, np.concatenate( (target_array[start_batch_slice:, :], target_array[0: end_batch_slice - num_examples, :])  , axis=0), other_args)
                    else:
                        prediction = drum_tabber(input_array[ start_batch_slice : end_batch_slice , ...], training = True)   # the forward pass though the current model, with training = True
                        # losses = compute_loss(prediction, target_array[start_batch_slice : end_batch_slice, :], other_args)

                    # apply gradients to update the model, the backward pass
                    gradients = tape.gradient(losses, drum_tabber.trainable_variables)
                    optmizer.apply_gradients(zip(gradients, drum_tabber.trainable_variables))

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

        return global_steps.numpy(), optimizer.lr.numpy(), losses.numpy()

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

        # treat each channel individually as a single "song"
        for channel in range(n_channels):

            # converts the current spectrogram into the correct input array
            input_array = spectro_to_input_array(spectrogram[:,:,channel], MODEL_TYPE)
            target_array = target_to_target_array(target[:,:,channel], MODEL_TYPE)
            num_examples = input_array.shape[0]

            # the number of model updates, based on the batch size and number of inputs
            num_updates = int(np.ceil(num_examples/VAL_BATCH_SIZE))

            for idx in range(num_updates):
                losses = 0
                start_batch_slice = idx*VAL_BATCH_SIZE
                end_batch_slice = (idx+1)*VAL_BATCH_SIZE

                # start recording the functions that are applied to autodifferentiate later
                with tf.GradientTape() as tape:
                    if end_batch_slice > num_examples:  # in the case where we are in the last batch, so we concat end of input_array/target_array with beginning samples of remaining length
                        prediction = drum_tabber(np.concatenate( (input_array[start_batch_slice:, :, :], input_array[ 0:end_batch_slice-num_examples , :, :]) , axis=0), training = False)
                        # TODO: code the compute_loss function
                        # losses = compute_loss(prediction, np.concatenate( (target_array[start_batch_slice:, :], target_array[0: end_batch_slice - num_examples, :])  , axis=0), other_args)
                    else:
                        prediction = drum_tabber(input_array[ start_batch_slice : end_batch_slice , :, :], training = False)   # the forward pass though the current model, with training = True
                        # losses = compute_loss(prediction, target_array[start_batch_slice : end_batch_slice, :], other_args)

        return losses.numpy()

    best_val_loss = 100000    # start with a high validation loss

    # loop over the number of epochs
    for epoch in range(TRAIN_EPOCHS):

        for spectrogram, target in train_set:   # outputs a full song's spectrogram and target, over the entire dataset
            # do a train step with the current spectrogram and target
            # results = train_song_step(spectrogram, target)
            pass
        for spectrogram, target in val_set:
            # do a validation step with the current spectrogram and target
            # results = validation_song_step(spectrogram, target)
            pass

    return None


if __name__ == '__main__':
    main()

'''
    # start the main function definition
    def main():


    # sets the gpus usage with the following code
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass

   # Does some type of checking of the trian log directory with
    if os.path.exists(TRAIN_LOGDIR): shutil.rmtree(TRAIN_LOGDIR)
    writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

    # Creates the two Dataset objects with
        trainset = Dataset('train')
        testset = Dataset('test')

    # calculates a bunch of useful vars
        steps_per_epoch = len(trainset)    # note that Dataset object is structured as a iterable whose length is equal to the num_batches
        global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
        total_steps = TRAIN_EPOCHS * steps_per_epoch

# Loads a model according to the conditionals provided. Uses custom functions to load weights into the different models constructed
    if TRAIN_FROM_CHECKPOINT:
        blah blah blah
    All use yolo = Create_Yolo3(training = True) option to create the model.
    Note that Create_Yolo3 function returns the object from the call of tf.keras.Model(input_layer, output_tensors)
    So yolo is a Keras.Model object, not any type of custom class extension!
        # Input is a layer imported from tf.keras.layers
        input_layer  = Input([input_size, input_size, channels])
        # output_tensors is a list of TF layers (either Conv2D, BatchNormalization [A CUSTOM VERSION] or a LeakyReLU)
        # Those layers were built up in different functions that built "blocks". Basically this is the Yolo implementation in TF

    # Loads the optimizer
        optimizer = tf.keras.optimizers.Adam()

# defines a function called train_step
    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            # makes the prediction with the image_data given to the train step (most likely the batch of images concatenated)
            # sets your running losses at 0
            pred_result = yolo(image_data, training=True)
            giou_loss=conf_loss=prob_loss=0

            # COMPUTES LOSSES HERE using the compute_loss function and adds the running total together
            grid = 3 if not TRAIN_YOLO_TINY else 2
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            # IMPORTANT CODE HERE!!! APPLIES THE GRADIENTS via the tape.gradient and optimizer.apply_gradients functions
            # Both of these are from TF and are not custom created .
            gradients = tape.gradient(total_loss, yolo.trainable_variables)
            optimizer.apply_gradients(zip(gradients, yolo.trainable_variables))

            # update learning rate
            # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
            global_steps.assign_add(1)    # global_steps.assign_add(1) is how to increment a tf.Variable, which is what global_steps must be
            if global_steps < warmup_steps:# and not TRAIN_TRANSFER:
                lr = global_steps / warmup_steps * TRAIN_LR_INIT
            else:
            # THIS IS THE COSINE LR. Note the tf.cos() function in there
                lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*(
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
            optimizer.lr.assign(lr.numpy())

            # writing summary data, I do't actually know what this does
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()

        return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    # create this validate_writer at this point for some reason???
        validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

# defines a function called validate_step
        with tf.GradientTape() as tape:
            # makes the prediction with the image_data given to it, but this time training=False in the prediction creation
            pred_result = yolo(image_data, training=False)
            giou_loss=conf_loss=prob_loss=0

            # COMPUTE LOSSES HERE using the same code as before in the train_step
            grid = 3 if not TRAIN_YOLO_TINY else 2
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            # Note that we do nothing else to anything. The validation set is reserved only for calculating the losses on a non-training data set.

        return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

# starts the epoch training loop
    best_val_loss = 1000 # should be large at start
    for epoch in range(TRAIN_EPOCHS):     #
        # trainset, made from Dataset class, is an iterable that contains two items at each iter. The X and Y
        for image_data, target in trainset:
            results = train_step(image_data, target)
            cur_step = results[0]%steps_per_epoch
            print("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, giou_loss:{:7.2f}, conf_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}"
                  .format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4], results[5]))

        if len(testset) == 0:
            print("configure TEST options to validate model")
            yolo.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
            continue

        count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
        for image_data, target in testset:
            results = validate_step(image_data, target)
            count += 1
            giou_val += results[0]
            conf_val += results[1]
            prob_val += results[2]
            total_val += results[3]
        # writing validate summary data
        with validate_writer.as_default():
            tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
            tf.summary.scalar("validate_loss/giou_val", giou_val/count, step=epoch)
            tf.summary.scalar("validate_loss/conf_val", conf_val/count, step=epoch)
            tf.summary.scalar("validate_loss/prob_val", prob_val/count, step=epoch)
        validate_writer.flush()

        print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
              format(giou_val/count, conf_val/count, prob_val/count, total_val/count))

# Saving the model based on the configs variable
        if TRAIN_SAVE_CHECKPOINT and not TRAIN_SAVE_BEST_ONLY:
            yolo.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME+"_val_loss_{:7.2f}".format(total_val/count)))
        if TRAIN_SAVE_BEST_ONLY and best_val_loss>total_val/count:
            yolo.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
            best_val_loss = total_val/count
        if not TRAIN_SAVE_BEST_ONLY and not TRAIN_SAVE_CHECKPOINT:
            yolo.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))



'''



'''
: if __name__ == '__main__':   calls the function main if train.py is run?
    main()
'''
