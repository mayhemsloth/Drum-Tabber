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
import json
import random
import warnings
import numpy as np
import librosa as lb
import tensorflow as tf
import audiomentations as adm    # audiomentations package used for data augmentations

from src.configs import *

class Dataset(object):
    '''
    Custom Dataset object class used to iterate through the training and validation data during model training
    '''
    def __init__(self, dataset_type, FullSet_df = None):   # dataset_type = 'train' or 'val', FullSet_df = encoded FullSet of songs in memory
        self.song_list = self.get_song_list(dataset_type)# train and validation subfolders will be contained in this folder file path form configs
        self.data_aug = TRAIN_DATA_AUG if dataset_type == 'train' else VAL_DATA_AUG   # boolean from configs
        self.FullSet_memory = True if (FullSet_df is not None) and TRAIN_FULLSET_MEMORY else False
        self.subset_df = FullSet_df.loc[self.song_list].copy() if self.FullSet_memory else None   # makes a copy of the subset of the FullSet, still with the multi-index labels of the songs
        self.aug_comp = Dataset.create_composition() if self.data_aug else None
        self.stem_dict = self.create_spleeter_configs_dict(dataset_type)

        self.set_type = dataset_type
        self.classes = [x for x in list(self.subset_df.columns) if '_' in x]  if self.subset_df is not None else None  # assumes that the labels in the df are appropriately named ('_' only being introduced at encoded phase)
        self.num_classes = len(self.classes)
        self.num_songs = len(self.song_list)
        self.batch_size = TRAIN_BATCH_SIZE if dataset_type == 'train' else VAL_BATCH_SIZE

        self.song_count = 0  # indexer for iterating through the set (song list)

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_songs

    def __next__(self):
        with tf.device('/cpu:0'):   # not sure what this does, but it was in the tutorial code

            if self.song_count < self.num_songs:   # if <, then in this iterable call we still have songs remaining in the Dataset
                # do preprocessing here and return the spectrogram, targets of the song
                song_title = self.song_list[self.song_count]
                #print(f'Dataset class __next__: preprocessing {song_title}')
                with warnings.catch_warnings():    # used to ignore the Pydub warning that always comes up
                    warnings.simplefilter("ignore")
                    spectrogram, target, label_ref_df = self.preprocess_song(song_title)

                self.song_count += 1
                return spectrogram, target, label_ref_df, song_title

            else:   # we went through all the songs in the Dataset, so let's reset the object to its default state and randomize
                self.song_count = 0
                random.shuffle(self.song_list)
                raise StopIteration       # stops the iterator from continuing

    def preprocess_song(self, song_title):
        '''
        High level function that uses the subset_df and the given song title to preprocess the song and encoded labels into the input and output

        Args:
            song_title [str]: the string of the name of the song title to parse_song

        Returns:
            numpy.array: input of the song for the model to train/val on (Spectrogram of varying size)
            numpy.array: targets of the song+tab for the model to train/val on (one-hot encoded numpy array)
        '''
        if self.FullSet_memory:               # if True, we have access to the FullSet subset
            song_df = self.subset_df.loc[song_title].copy()   # won't affect the underlying self.subset_df and gets rid of multi-indexing song title labels
            song_df['sample start'] = song_df['sample start'].apply(lambda valu: valu-song_df.at[0, 'sample start']) # realigning the sample start number to beginning of the reconstructed slice (instead of beginning of the actual song)
            if self.data_aug:
                song_df = self.shift_augmentation(song_df)  # implemented my own shift augmentation function because it was simpler to move the entire df labels and samples together
            song = np.vstack(song_df['song slice'].to_numpy()).T   # stacks the song slices back into a single numpy array of shape (channels, samples)

            mono_song = lb.core.to_mono(song)

            channels = [mono_song]              # channels is a list of audio channels describing the different mixes of the song

            if self.stem_dict['use_drum_stem']:     # in the case where the drum only stem is somehow used
                drums = np.vstack(song_df['drums slice'].to_numpy()).T   #  the stacked-back-to-normal numpy array containing the drum stack (channels, samples)
                if self.stem_dict['include_drum_stem']:
                    channels.append(lb.core.to_mono(drums))
                if self.stem_dict['include_mixed_stem'] or self.stem_dict['replace_with_mixed_stem']:
                    mixed_stem = song*(self.stem_dict['mixed_stem_weights'][0]) + drums*(self.stem_dict['mixed_stem_weights'][1])   # (song weight, drums weight)
                    mixed_stem_mono = lb.core.to_mono(mixed_stem)
                    if self.stem_dict['replace_with_mixed_stem']:
                        channels[0] = mixed_stem_mono  # if train replace is true, replace the mono_song with mixed_stem
                    else:
                        channels.append(mixed_stem_mono) # logic to avoid adding the mixed stem AND replacing the mono_song

            # TODO: Get the correct sample rate (sr) from the song_info dictionary back in the creation of the MAT
                # For now, assume all sr's are = 44100
            if self.data_aug:
                channels = self.augment_audio_cp(channels, self.aug_comp, sr=SAMPLE_RATE)    # augment_audio_cp means class-preserving. The audio isn't significantly changed to change the class labels

            # make spectrogram and augment spectrogram directly if desired
            spectrogram = self.create_spectrogram(channels, sr=SAMPLE_RATE)
            if self.data_aug:
                spectrogram = self.augment_spectrogram(spectrogram)

            # make the targets using information of the spectrogram and the song_df
            target = self.create_target(spectrogram, song_df)

            # make a label reference df to help with error metric calculations
            label_ref_df = self.create_label_ref(song_df)

        # TODO: Implement the case where the FullSet_memory == False and we need to load the songs individually everytime
        else:     # case of not keeping FullSet in memory
            print('FULLSET_MEMORY == FALSE NOT IMPLEMENTED YET. EVERYTHING ELSE WILL NOT FUNCTION PROPERLY')
            spectrogram, target, label_ref_df = None, None, None

        return spectrogram, target, label_ref_df

    # START Helper Functions
    def get_song_list(self, dataset_type):
        '''
        Helper function to get the song list for the current Dataset type

        Args:
            dataset_type [str]: either 'train' or 'val' or 'verify'

        Returns:
            list: a list of strings that are the names of the song subfolders that exist for that dataset type
        '''
        if dataset_type == 'train':
            return [x.name for x in os.scandir(SONGS_PATH) if x.is_dir() and x.name not in VAL_SONG_LIST]
        if dataset_type == 'val':
            return [x.name for x in os.scandir(SONGS_PATH) if x.is_dir() and x.name in VAL_SONG_LIST]
        if dataset_type == 'verify':
            return [x.name for x in os.scandir(SONGS_PATH) if x.is_dir()]

    @staticmethod
    def create_composition():
        '''
        Creates the data augmentation composition class object for transforming (audiomentations.Compose)

        Args:
            None

        Returns:
            audiomentations.Compose: the class used to do the data augmentation (at least the audio augmentation)
        '''
        # Building up a transform list using the transform classes found in audiomentations
        transform = []
        transform.append(adm.PolarityInversion(p=POLARITY_CHANCE))
        transform.append(adm.FrequencyMask(min_frequency_band = 0.2, max_frequency_band = 0.5, p=FREQUENCY_MASK_CHANCE))
        transform.append(adm.AddGaussianSNR(min_SNR = 0.05, max_SNR=0.1, p=GAUSSIAN_SNR_CHANCE))
        transform.append(adm.PitchShift(min_semitones = -4, max_semitones = 4, p=PITCH_SHIFT_CHANCE))
        transform.append(adm.Normalize(p=NORMALIZE_CHANCE))
        transform.append(adm.ClippingDistortion(min_percentile_threshold = 10, max_percentile_threshold = 25, p=CLIPPING_DISTORTION_CHANCE))
        transform.append(adm.AddBackgroundNoise(sounds_path=BACKGROUNDNOISES_PATH, min_snr_in_db=5, max_snr_in_db=15, p=BACKGROUND_NOISE_CHANCE))
        transform.append(adm.AddGaussianNoise(min_amplitude = 0.005, max_amplitude = 0.025, p = GAUSSIAN_NOISE_CHANCE))
        transform.append(adm.Gain(min_gain_in_db = -12, max_gain_in_db=12, p=GAIN_CHANCE))
        transform.append(adm.Mp3Compression(min_bitrate = 24, max_bitrate = 82, backend = "pydub", p = MP3_COMPRESSION_CHANCE))

        # create the composition of the different transforms
        aug_comp = adm.Compose(transforms = transform, shuffle = False, p = 1.0)

        return aug_comp

    def create_spleeter_configs_dict(self, dataset_type):
        '''
        Creates the spleeter configs dictionary that holds all the bools/configs for implementing stem options of this dataset type

        Args:
            dataset_type [str]: 'train' or 'val'

        Returns:
            dict: a dictionary containing the different configuratin options for the current dataset type (gathered/organized from configs.py vars)
        '''

        use_drum_stem = TRAIN_USE_DRUM_STEM if dataset_type == 'train' else VAL_USE_DRUM_STEM
        include_drum_stem = TRAIN_INCLUDE_DRUM_STEM if dataset_type == 'train' else VAL_INCLUDE_DRUM_STEM
        include_mixed_stem = TRAIN_INCLUDE_MIXED_STEM if dataset_type == 'train' else VAL_INCLUDE_MIXED_STEM
        mixed_stem_weights = TRAIN_MIXED_STEM_WEIGHTS if dataset_type == 'train' else VAL_MIXED_STEM_WEIGHTS
        replace_with_mixed_stem = TRAIN_REPLACE_WITH_MIXED_STEM  if dataset_type == 'train' else VAL_REPLACE_WITH_MIXED_STEM

        spleeter_configs = {'use_drum_stem' : use_drum_stem,
                            'include_drum_stem' : include_drum_stem,
                            'include_mixed_stem': include_mixed_stem,
                            'mixed_stem_weights': mixed_stem_weights,
                            'replace_with_mixed_stem': replace_with_mixed_stem
                            }

        return spleeter_configs
    # END Helper Functions

    # START Augmentation Functions
    def shift_augmentation(self, song_df):
        '''
        Randomly shifts the entire song_df a random amount. Self-implemented data augmentation function used to create a new "starting point" for the song

        Args:
            song_df [Dataframe]: a single song's music aligned tab df, in the process of being parsed for training/val

        Returns:
            Dataframe: song_df, either shifted or not shifted, depending on the resolution of SHIFT_CHANCE
        '''

        if random.random() < SHIFT_CHANCE:   # percentage that data is shifted is dictated by SHIFT_CHANCE configs variable
            # do the shifting
            shift_idx = random.randint(1, len(song_df)-1)         # choose a random index to slice df into two and append first part after second
            first, second = song_df[:shift_idx].copy(), song_df[shift_idx:].copy()   # split the df into two separate parts to eventually swap their places
            ss_shift = second.at[shift_idx, 'sample start']
            second['sample start'] = second['sample start'].apply(lambda ss: ss-ss_shift)   # shifting the sample start back because second becomes beginning
            first['sample start'] = first['sample start'].apply(lambda ss: ss+ss_shift)   # shifting the sample start forward because first becomes second part
            song_df = second.append(first, ignore_index = True)          # finalize the shift, and reset index
        return song_df

    def augment_audio_cp(self, channels, aug_comp, sr):
        '''
        Applies data augmentations to the current song audio for class-preserving

        Args:
            channels [list]: list of numpy.arrays that represent the samples (either [mono] or [mono, L, R])
            aug_comp [adm.Compose]: the Compose object that contains all the class-preserving list of audio augmentations
            sr [int]: sample rate of the current song

        Returns:
            list: list of np.arrays that are the samples of the augmented song audio
        '''

        # channel is now a 1D numpy.array to be input into the audiomentations Compose object
        augmented_channels = [aug_comp(samples = np.asfortranarray(channel), sample_rate = sr) for channel in channels]

        return augmented_channels

    def augment_spectrogram(self, spectrogram):
        '''
        Augments the spectrogram with the currently coded spectrogram augmentation functions defined interally.

        Args:
            spectrogram [np.array]: spectrogram of the curent song. Shape is either a n by m by 1 or n by m by x

        Returns:
            np.array: spectrogram, either augmented or the original depending on the random triggering of the augmentations
        '''

        def bin_dropout(spectrogram, p):
            '''
            Spectrogram augmentation that drops out (sets to equivalent 0, or minimum value of the spectrogram due to dB unit)
            random elements of the spectrograms
            '''
            m, n, n_channels = spectrogram.shape
            spectro_channels = []
            for idx in range(n_channels):
                if random.random() < p:
                    min_value = np.min(spectrogram[:,:,idx])
                    # do the bin dropout augment
                    mask = np.random.rand(m,n) < BIN_DROPOUT_RATIO
                    spectrogram[:,:,idx][mask] = min_value   # apply the mask to the current channel, changing them to the minimum value
                spectro = spectrogram[:,:, idx] # transfer the current channel (augmented or not) to the spectro var
                spectro_channels.append(spectro)
            return np.stack(spectro_channels, axis = -1) # stacks the spectro_channels back into a single np.array object

        def S_noise(spectrogram, p):
            '''
            Spectrogram augmentation that multiplies the entire spectrogram by random amounts of small values.
            '''
            m, n, n_channels = spectrogram.shape
            spectro_channels = []
            for idx in range(n_channels):
                if random.random() < p:
                    # do the S_noise augment
                    multiplier_matrix = np.random.uniform(low = 1-S_NOISE_RANGE_WIDTH/2, high = 1+S_NOISE_RANGE_WIDTH/2, size = (m,n))  # creates a matrix of random numbers the same shape as one channel of the spectrogram
                    spectro = np.multiply(spectrogram[:,:,idx], multiplier_matrix) # augments the current spectro channel
                else:
                    spectro = spectrogram[:,:,idx]
                spectro_channels.append(spectro)
            return np.stack(spectro_channels, axis = -1)  # stacks the spectro_channels back into a single np.array object

        spectrogram = bin_dropout(spectrogram.copy(), BIN_DROPOUT_CHANCE)
        spectrogram = S_noise(spectrogram.copy(), S_NOISE_CHANCE)

        return spectrogram
    # END Augmentation Functions

    def create_spectrogram(self, channels, sr):
        '''
        Makes a spectrogram based on the song channels given and the model options from the configs.py file

        Args:
            channels [list]: list of np.arrays that are the samples
            sr [int]: sample rate of the current song

        Returns:
            np.array: numpy array that is the spectrogram: either a n by m by 1 or n by m by x depending on how many channels exist
        '''

        spectro_channels = [] # create either spectrograms based on the number of channels (different versions of the same song)
        for channel in channels:
            spectro = lb.feature.melspectrogram(np.asfortranarray(channel), sr=sr, n_fft = WINDOW_SIZE, hop_length = HOP_SIZE,
                                                center = False, n_mels = N_MELS, fmax=FMAX) # numpy array of shape (n_mels, t)
            # print(f'create_spectrogram: spectro.shape = {spectro.shape}')
            if SHIFT_TO_DB:
                spectro = lb.power_to_db(spectro, ref = np.max)   # range of [-80,0] values
            if INCLUDE_FO_DIFFERENTIAL:
                spectro_ftd = lb.feature.delta(data = spectro, width = 9, order=1, axis = -1)    # calculate the first time derivative of the spectrogram. Uses 9 frames to calculate
                    # spectro_f(irst)t(ime)d(erivative).shape = (n_mels, t) SAME AS spectro
                # manual normalize of current spectro_ftd
                spectro_ftd_norm = (spectro_ftd - spectro_ftd.mean())/spectro_ftd.std()
                spectro = np.concatenate([spectro, spectro_ftd_norm], axis = 0)    # first time derivative attached at end of normal log mel spectrogram (n_mels of spectro, then n_mels of ftd)
                    # spectro.shape = (2* n_mels, t)
            spectro_channels.append(spectro)

        spectrogram = np.stack(spectro_channels, axis = -1)

        return spectrogram # spectrogram has shape of (n_mels (perhaps x2), t (determined by length of song and HOP_SIZE), n_channels (1 or 3) )

    def create_target(self, spectrogram, song_df):
        '''
        Creates the target labels from the tab dataframe that contains the one-hot encoded labels and aligns them to the spectrogram columns
        The model will have a 1:1 labeling format: for every frame ( or window) that goes in, a prediction on labels comes out.
        NOTE that this code assumes each channel contains the same "drum onset data" for labeling purposes. That is, no data augmentation
        function that changes the channels individually should affect the labels.

        Args:
            spectrogram [np.array]: a spectrogram object that represents the X that will go into the model (n_mels, t, n_channels)
            song_df [Dataframe]: tab df that contains all the one-hot-encoded columns along with a 'sample start' column that contains the sample number start of that row w.r.t. the audio in the spectrogram

        Returns:
            np.array: the array of output targets [0,1]. Shape is (n_classes, n_windows, n_channels)
        '''

        _, n_windows, n_channels = spectrogram.shape   # S.shape is (n_mels, t, n_channels), where t = number of frames/windows
        labels_df = song_df.drop(columns = ['sample start', 'song slice', 'drums slice'], errors = 'ignore')   # get a dataframe that only has the labels
        class_names = labels_df.columns
        n_classes = len(class_names)
        targets = np.zeros((n_classes, n_windows, n_channels), dtype=int)  # initializes as all zeros in shape of (n_classes number of rows, n_frames number of columns)
        sample_starts = list(song_df['sample start'])

        # the spectrogram represents the entire "song" or clip, and thus we assume it starts at sample 0 w.r.t. the song's samples
        # each window is WINDOW_SIZE long, and each window starts at sample idx_frame_in_list * HOP_SIZE

        # loop through each window, and check if that window is in "range" to be labeled
        tab_slice_idx = 0
        for idx in range(n_windows):
            window_start = idx * HOP_SIZE     # the sample number of the window's beginning
            positive_window = int(window_start + POSITIVE_WINDOW_FRACTION*WINDOW_SIZE)
            negative_window = int(window_start - NEGATIVE_WINDOW_FRACTION*WINDOW_SIZE)
            # TODO: fix this algorithmic approach so that it is more compatible with a larger WINDOW_SIZE while also being speedy
            if negative_window < sample_starts[tab_slice_idx] <= positive_window: # the drum event's sample_start falls within the acceptable range
                targets[:, idx, :] = np.stack([labels_df.loc[tab_slice_idx].to_numpy() for _ in range(n_channels)], axis=-1)
                ''' ASSUMES all channels contain the same "drum data" for labeling purposes '''
            if sample_starts[tab_slice_idx] < negative_window:   # if the sliding windows have passed the current tab slice
                tab_slice_idx +=1
            if tab_slice_idx >= len(sample_starts):
                break

        return targets

    def create_label_ref(self, song_df):
        '''
        Creates the label reference dataframe for the current song. Used to track true/false positive/negatives

        Args:
            song_df [Dataframe]: tab df that contains all the one-hot-encoded columns along with 'song slice' column and a 'sample start' column that contains the sample number start of that row w.r.t. the audio in the spectrogram

        Returns:
            Dataframe: label_ref_df that contains the new column of 'sample start ms' which is the ms number where that df row starts
        '''

        label_ref_df = song_df.drop(columns = ['song slice', 'drums slice'], errors = 'ignore').copy()   # get a dataframe that only has the labels and sample starts

        return label_ref_df
