#================================================================
#
#   File name   : dataset.py
#   Author      : Thomas Hymel
#   Created date: 9-2-2020
#   GitHub      : https://github.com/mayhemsloth/Drum-Tabber
#   Description : Functions used to prepare dataset for training
#
#================================================================

import os
import librosa as lb          # loads the librosa package
import numpy as np

from src.configs import *


# defining the Dataset class
class Dataset(object):
    def __init__(self, dataset_type):   # dataset_type = 'train' or 'val'
        self.songs_path = SONGS_PATH    # train and validation subfolders will be contained in this folder file path form configs
        self.data_aug = TRAIN_DATA_AUG if dataset_type == 'train' else VAL_DATA_AUG   # boolean from configs


        # THESE ARE IN THE OTHER DATASET CLASS
        self.input_size = None
        self.classes = None
        self.num_classes = None

        # self.annotations = self.load_annotations(dataset_type)


        self.songs= self.load_songs(dataset_type)

    def __iter__(self):
        return self

    # loads the raw song data
    def load_songs(self, dataset_type):

        return None


    @staticmethod
    def somefunction():
        return None

def create_spectrogram(song_fp):
    '''
    Makes a spectrogram based on the song filepath given and the model options in the configs.py file

    Args:
        song_fp [str]: string of the filepath to the song file to be made into a spectrogram

    Returns:
        np.array: numpy array that is the spectrogram: either a n by m by 1 or n by m by 3 depending on the INCLUDE_LR_CHANNELS
    '''

    lb_song, sr = lb.core.load(song_fp, sr=None, mono=False)    # returns numpy array of shape (2,n) for stereo
    lb_mono = lb.core.to_mono(lb_song)          # returns a numpy array of shape (n,) that is the mono of the loaded song
    print(f'create_spectrogram: lb_song.shape = {lb_song.shape}')
    print(f'create_spectrogram: sr = {sr}')

    # create mono spectro
    print(f'create_spectrogram: lb_mono.shape = {lb_mono.shape}')
    mono_S = lb.feature.melspectrogram(lb_mono, sr=sr, n_fft = WINDOW_SIZE, hop_length = HOP_SIZE, center = False, n_mels = N_MELS) # numpy array of shape (n_mels, t)
    print(f'create_spectrogram: mono_S.shape = {mono_S.shape}')

    if INCLUDE_LR_CHANNELS:
        # create left channel spectro
        L_song = lb_song[0,:]
        L_S = lb.feature.melspectrogram(np.asfortranarray(L_song), sr=sr, n_fft = WINDOW_SIZE, hop_length = HOP_SIZE, center = False, n_mels = N_MELS)
        # create right channel spectro
        R_song = lb_song[1,:]
        R_S = lb.feature.melspectrogram(np.asfortranarray(R_song), sr=sr, n_fft = WINDOW_SIZE, hop_length = HOP_SIZE, center = False, n_mels = N_MELS)

    if SHIFT_TO_DB:
        mono_S = lb.power_to_db(mono_S, ref = np.max) # use entire song to convert to log mel spectrogram
        if INCLUDE_LR_CHANNELS:
            L_S = lb.power_to_db(L_S, ref = np.max)
            R_S = lb.power_to_db(R_S, ref = np.max)

    if INCLUDE_FO_DIFFERENTIAL:
        mono_S_ftd = lb.feature.delta(mono_S, order=1) # calculate the first time derivative of the spectrogram
            # mono_S_firsttimederiv.shape = (n_mels, t) SAME AS full_song_spectro
        mono_S = np.concatenate([mono_S, mono_S_ftd], axis = 0)    # first time derivative attached at end of normal log mel spectrogram (n_mels of spectro, then n_mels of ftd)
            # mono_S.shape = (2* n_mels, n_spectro_row)
        if INCLUDE_LR_CHANNELS:
            L_S_ftd = lb.feature.delta(L_S, order=1)
            L_S = np.concatenate([L_S, L_S_ftd], axis = 0)
            R_S_ftd = lb.feature.delta(R_S, order=1)
            R_S = np.concatenate([R_S, R_S_ftd], axis = 0)

    if INCLUDE_LR_CHANNELS:
        spectrogram = np.stack([mono_S, L_S, R_S], axis = -1)     # spectrogram channel dimension order is mono, L, R
        print(f'create_spectrogram: mono_S.shape after ftd = {mono_S.shape}')
        print(f'create_spectrogram: L_S.shape after ftd = {L_S.shape}')
        print(f'create_spectrogram: R_S.shape after ftd = {R_S.shape}')
        print(f'create_spectrogram: spectrogram.shape after ftd = {spectrogram.shape}')
    else:
        spectrogram = np.stack([mono_S], axis = -1)

    spectrogram = np.transpose(spectrogram, (1,0,2)) # transposes the first two dimensions to ensure that it is (t, n_mels_total, n_spectro_channels)
    print(f'create_spectrogram: spectrogram.shape after transpose = {spectrogram.shape}')

    return spectrogram
