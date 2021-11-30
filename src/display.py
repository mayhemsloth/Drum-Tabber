#================================================================
#
#   File name   : utils.py
#   Author      : Thomas Hymel
#   Created date: 3-2-2020
#   GitHub      : https://github.com/mayhemsloth/Drum-Tabber
#   Description : Tools for inference and display
#
#================================================================

import os
import json
import warnings
import numpy as np
import pandas as pd
import librosa as lb
import tensorflow as tf
from datetime import date
from pydub import AudioSegment   # main class from pydub package used to upload mp3 into Python and then get a NumPy array
import IPython.display as ipd    # ability to play audio in Jupyter Notebooks if needed
import matplotlib.pyplot as plt


from src.configs import *
from src.utils import clean_labels, collapse_class, one_hot_encode, create_FullSet_df

# START OF VISUAL DISPLAY FUNCTIONS


# END OF VISUAL DISPLAY FUNCTIONS


# START OF VISUAL VERIFICATION FUNCTIONS
def prep_visually_verify(configs_dict = None):
    '''
    Helper function that is designed to be run once before using the visually_verify function (to speed that up)
    If a configs_dict is passed, it means you are preparing to run visually_verify with a model (and probable inference)
    As such, this function changes a number of global variables that exist from the 'from configs.py import *' import.
    This is done so that previous code can be reused, and so that the global variable's values correspond with the
    model values and not the current configs file variables.

    Args:
        configs_dict [dict]: Default None. If not none, will use the configurations from the current model to set up the FullSet_df

    Returns:
        Dataframe: FullSet_df that will be used to pluck song information from (only need to create once)

    '''



    if configs_dict is not None:   # change all the global variables necessary, so the Dataset.preprocess function can be reused properly with the current model configs
        c_d = configs_dict['classification_dict']
        keep_dyn, keep_bel, keep_tom, hihat_cla, cymbal_cla = c_d['keep_dynamics'], c_d['keep_bells'], c_d['keep_toms_separate'], c_d['hihat_classes'], c_d['cymbal_classes']
        global HOP_SIZE, N_MELS, MODEL_TYPE, WINDOW_SIZE, FMAX, N_CONTEXT_PRE, N_CONTEXT_POST, NEGATIVE_WINDOW_FRACTION, POSITIVE_WINDOW_FRACTION, INCUDE_FO_DIFFERENTIAL
        global SHIFT_TO_DB, TRAIN_USE_DRUM_STEM

    else:
        keep_dyn, keep_bel, keep_tom, hihat_cla, cymbal_cla = KEEP_DYNAMICS, KEEP_BELLS, KEEP_TOMS_SEPARATE, HIHAT_CLASSES, CYMBAL_CLASSES

    FullSet = clean_labels(create_FullSet_df(SONGS_PATH))  # create and clean FullSet_df

    FullSet_df = one_hot_encode(collapse_class(FullSet_df = FullSet,
                        keep_dynamics = keep_dyn,
                        keep_bells = keep_bel,
                        keep_toms_separate = keep_tom,
                        hihat_classes = hihat_cla,
                        cymbal_classes = cymbal_cla))


    return FullSet_df


def visually_verify(song_title, FullSet_df, samples_per_class = 4, requested_classes = [0],
                    infer=False, model=None, configs_dict=None, verify_drum_stem = False,
                    data_augment=False, scale=3.0, threshold = 0.5):
    '''
    Creates and displays a compilation of spectro images for each class specified, for a number of random samples.
    When given a model and its configs dict, may also run inference to augment images to help with visually verifying performance of a model
    Essentially assumes a "context" based model and not a sequence based model

    Args:
        song_title [str]: string of only the song title, as used in the songs data folder
        FullSet_df [Dataframe]:
        samples_per_class [int]: Default 4. Number of random samples to display, per class requested
        requested_classes [list]: list of ints, Default [0]. Corresponds to the desired classes
        infer [bool]: Default False. If True, use the passed model to run inference on the song to display the predicted probability for each sample
        model [TF Keras]: Tensorflow Keras model that corresponds to a trained drum_tabber model.
        configs_dict [dict]: Configs dict that is created, saved, and loaded with the drum_tabber model.
        verify_drum_stem [bool]: Whether to use the drum stem, instead of the original song, to visually inspect/run inference on
        data_augment [bool]: Default False. If True, activates data augmentation while preprocessing the song
        scale [float]: Default 3.0. Changes the absolute scale of each subimage, and thus the overall image size
        threshold [float]: Default 0.5. Below this threshold, any prediction from inference will not be shown. Above this threshold, class and probability will be printed

    Returns:
        figures? outputs to screen

    '''

    # prepare the Dataset to be pulled from
    verify_ds = Dataset('verify', FullSet_df = FullSet_df)   # prepare the Dataset object with full song selection
    verify_ds.data_aug = data_augment   # change the data augmentation bool to desired bool
    if verify_ds.data_aug:
        verify_ds.aug_comp = Dataset.create_composition()
    if verify_drum_stem:                # change the drum stem bool to desired bool
        verify_ds.stem_dict['use_drum_stem'] = True
        verify_ds.stem_dict['include_drum_stem'] = True
        verify_ds.stem_dict['include_mixed_stem'] = False  # needs to be changed if later include ability to vis verify mixed stems

    # preprocess song, or print message if song_title isn't in the song list
    if song_title in verify_ds.song_list:
        spectrogram, target, label_ref_df = verify_ds.preprocess_song(song_title)   # get the spectrogram and target matrices
    else:
        print(f'{song_title} is not a valid choice. Please choose a song in the dataset to visually verify.')
        return None

    # prepare spectrogram
    spectrogram = (spectrogram+80.0)/80.0  # map spectro values to [0,1]
    spectrogram = spectrogram[:,:,1] if verify_drum_stem else spectrogram[:,:,0]  # spectrogram choose the correct channel, and converts to a 2D array
    target = target[:,:,1] if verify_drum_stem else target[:,:,0] # target.shape = (num_classes, spectro_slices), spectrogram.shape = (n_mels, spectro_slices)
    n_features, _ = spectrogram.shape
    pre_context, post_context = N_CONTEXT_PRE, N_CONTEXT_POST
    input_width = pre_context + 1 + post_context
    zeropad_pre, zeropad_post = np.full(shape = (n_features,pre_context), fill_value = np.min(spectrogram)), np.full(shape = (n_features,post_context), fill_value = np.min(spectrogram))
    spectrogram_zeropadded = np.concatenate([zeropad_pre, spectrogram, zeropad_post], axis=1)

    # pick the random samples
    true_rand_samples = np.stack([np.random.choice(np.where(target[class_num,:]==True)[0], size=samples_per_class) for class_num in requested_classes])
    false_rand_samples = np.stack([np.random.choice(np.where(target[class_num,:]==False)[0], size=samples_per_class) for class_num in requested_classes])
    all_rand_samples = np.stack([true_rand_samples, false_rand_samples], axis=-1) # shape = (requested_classes, samples_per_class, 2 (0=true, 1=false))

    # create the figure and axes
    fig, axs = plt.subplots(nrows = len(requested_classes)*2, ncols = samples_per_class,
                            sharex = True, sharey = True, squeeze = False,
                            figsize=(1.3*scale*samples_per_class, scale*2*len(requested_classes) ))

    for i, class_num in enumerate(requested_classes):    # go through the list of classes requested
        for k in [0,1]:                                  # choose doing True (0) or False (1)
            for j in range(samples_per_class):           # go through each True or False sample
                sample_number = all_rand_samples[i,j,k]  # grab the sample number
                this_ax = axs[(i)*2 + k][j]
                this_sample_labels = target[:,sample_number]    # grab the target labels of all classes from current sample
                this_spectro_input = spectrogram_zeropadded[:, (sample_number+pre_context)-pre_context: (sample_number+pre_context)+post_context+1]
                img = lb.display.specshow(this_spectro_input, sr=44100, x_axis='time', y_axis= 'mel', ax=this_ax, hop_length=441)

                if infer and (model is not None) and (configs_dict is not None):  # inference time
                    pred = model(np.expand_dims(this_spectro_input, axis=(0,3)), training=False).numpy()[0,:]   # make prediction using this_spectro_input (expanded to fit into the model)
                    above_threshold_classes = np.where(pred>threshold)[0]
                    pred_string = ' | '.join([verify_ds.classes[x] + " = {:3.2f}".format(pred[x]) for x in above_threshold_classes if verify_ds.classes[x] not in ['tk_beat', 'tk_downbeat']])
                    this_ax.set_xlabel(xlabel = pred_string)    # puts the class title and prediction probability in the x axis label
                    # disregard the tk_downbeat and tk_beat classes (in the pred_string declaration line)

                # get the true labels for this sample, and make it the title of the subplot
                this_true_label_names = [verify_ds.classes[x] for x in np.where(this_sample_labels ==True)[0]]
                this_ax.set_title(label = ' | '.join(this_true_label_names))

            fig.colorbar(img, ax=axs[i*2+k].ravel().tolist(), format='%+2.2f db')  # add colorbar axis at each end of row

    return None

# START OF VISUAL VERIFICATION FUNCTIONS



# START OF INFERENCE FUNCTIONS
def song_to_tab(drum_tabber, configs_dict, song_file_w_extension, songs_to_tab_folder_path, write_to_txt = True):
    '''
    High-level function that takes a trained model and a new song, makes an inference on that song, and then writes the results to a filepath

    Args:
        drum_tabber [keras.Model]:
        configs_dict [dict]:
        song_file_w_extension [str]:
        songs_to_tab_folder_path [str]:
        write_to_txt [bool]: Default True. If true, will write to a .txt file.

    Returns:
        list: list of strings, a reconstructed machine-friendly tab that represents the song's drum tab from the model inference pass and post-processing.
    '''

    '''---LOAD SONG---'''
    song_name = os.path.splitext(song_file_w_extension)[0]  # grabs the string of the song name only
    song_file_ext = os.path.splitext(song_file_w_extension)[1][1:]  # grabs the string extension of the file
    abs_song_fp = os.path.join(songs_to_tab_folder_path, song_name, song_file_w_extension) # ASSUMES A song_to_tab_folder ---> song_name_folder ---> song_file_w_extension folder storage format
    song, sr_song = load_song(abs_song_fp)    # loads mono here

    '''---CONVERT SONG INTO SPECTROGRAM (using configs_dict parameters)---'''
    spectrogram = song_to_spectrogram(song, sr_song, configs_dict)

    '''---TRANSFORM SPECTROGRAM INTO INPUT ARRAY---'''
    input = spectrogram_to_input(spectrogram, configs_dict)  # input.shape = (n_windows (examples), n_features, width_size)

    '''---MAKE INFERENCE WITH TRAINED MODEL AND INPUT ARRAY---'''
    prediction = drum_tabber(np.expand_dims(input, axis=-1), training = False).numpy()   # expand dimension to make proper dimension, then change output from TF to numpy

    '''---SEND PREDICTIONS THROUGH PEAK PICKING FUNCTION---'''
    detected_peaks = detect_peaks(prediction)

    '''---DECODE THE PEAKS (PREDICTED ONSET EVENTS!) INTO A TAB---'''
    class_names_dict = configs_dict['class_names_dict']     # class_names_dict has structure of {idx_in_prediction : 'class_label_name'}



    return None

def load_song(direct_filepath, mono_channel=True):
    '''
    Helper function used to load a song in to be tabbed (using librosa load)

    Args:
        direct_filepath [str]: absolute filepath to the song's file to be loaded in
        mono_channel [bool]: Default is True. Boolean to choose to load song in mono or stereo

    Returns:
        np.array: librosa song samples, either (n,) or (2,n) shape depending on mono or stereo
        int: sample rate (sr) for the song
    '''

    # uses librosa to output a np.ndarray of shape (n,) or (2,n) depending on the channels
    lb_song, sr_song = lb.core.load(direct_filepath, sr=None, mono=mono_channel)

    return lb_song, song_sr

def song_to_spectrogram(song, sr_song, configs_dict):
    '''
    Helper function to make a song into a spectrogram based on the model configurations that the spectrogram will be processed in

    Args:
        song [np.array]: librosa song samples
        sr_song [int]: sample rate (sr) for the song
        configs_dict [dict]: configurations dictionary created with the Keras model upon training

    Returns:
        np.array: spectrogram, created with correct configs, and also normalized corectly
    '''

    spectro = lb.feature.melspectrogram(np.asfortranarray(song), sr=sr_song, n_fft = configs_dict['window_size'], hop_length = configs_dict['hop_size'], center = False, n_mels = configs_dict['n_mels'])
    if configs_dict['shift_to_db']:
        spectro = lb.power_to_db(spectro, ref = np.max)
    # manually normalize the current spectro channel
    spectro_norm = (spectro - spectro.mean())/spectro.std()
    if configs_dict['include_fo_differential']:
        spectro_ftd = lb.feature.delta(data = spectro, width = 9, order=1, axis = -1)    # calculate the first time derivative of the spectrogram. Uses 9 frames to calculate
            # spectro_f(irst)t(ime)d(erivative).shape = (n_mels, t) SAME AS spectro
        # manually normalize current spectro_ftd
        spectro_ftd_norm = (spectro_ftd - spectro_ftd.mean())/spectro_ftd.std()
        spectro_norm = np.concatenate([spectro_norm, spectro_ftd_norm], axis = 0)    # first time derivative attached at end of normal log mel spectrogram (n_mels of spectro, then n_mels of ftd)
            # spectro.shape = (2* n_mels, t) = (n_mels from spectro THEN n_mels from spectro_ftd, t)
    spectrogram = np.copy(spectro_norm)   # 2D np.array

    return spectrogram

def spectrogram_to_input(spectrogram, configs_dict):
    '''
    Helper function to change a spectrogram into the proper input shape of the current model

    Args:
        spectrogram [np.array]: 2D spectrogram
        configs_dict [dict]: configurations dictionary created with the Keras model upon training

    Returns:
        np.array: proper input shape of the spectrogram
    '''

    n_features, n_windows = spectrogram.shape

    # TODO: Finish the other model type options when they become available
    if configs_dict['model_type'] in  ['Context-CNN', 'TimeFreq-CNN']:
        pre_context, post_context = configs_dict['n_context_pre'], configs_dict['n_context_post']
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
                input_array[idx, :, :input_width-end] = spectrogram[:, idx-pre_context: n_windows]
            else:    # in a "normal" middle window where you slice into the spectrogram normally
                input_array[idx, :,:] = spectrogram[:, idx-pre_context : idx+post_context+1]

    else:
        input_array = None
        print('Other model types are not implemented yet!')

    return input_array

def peaks_to_tab(detected_peaks, configs_dict):
    '''
    Changes the detected peaks array into a tab-like array by morphing the spectrogram-sized array into a time-based array.
    Heavily relies on detected peaks of beats being "correct" (to infer the time).

    Args:
        detected_peaks [np.array]: a 0s/1s array of shape (n_samples, n_classes) that denotes location of peaks. Output of detect_peaks function
        configs_dict [dict]:
    '''

    return None
