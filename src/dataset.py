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
import random
import librosa as lb          # loads the librosa package
import librosa.display
import numpy as np
import audiomentations as adm    # audiomentations package used for data augmentations

from src.configs import *


# defining the Dataset class
class Dataset(object):
    def __init__(self, dataset_type, FullSet_df = None):   # dataset_type = 'train' or 'val', FullSet_df = encoded FullSet of songs in memory
        self.song_list = self.get_song_list(dataset_type)# train and validation subfolders will be contained in this folder file path form configs
        self.data_aug = TRAIN_DATA_AUG if dataset_type == 'train' else VAL_DATA_AUG   # boolean from configs
        self.FullSet_memory = True if (FullSet_df is not None) and TRAIN_FULLSET_MEMORY else False
        self.subset_df = FullSet_df.loc[self.song_list].copy() if self.FullSet_memory else None   # makes a copy of the subset of the FullSet, still with the multi-index labels of the songs
        self.aug_comp = Dataset.create_composition() if self.data_aug else None

        self.num_songs = len(self.song_list)

        # THESE ARE IN THE OTHER DATASET CLASS
        self.input_size = None
        self.classes = None
        self.num_classes = None

        # self.annotations = self.load_annotations(dataset_type)

    def __iter__(self):
        return self

    def __next__(self):

        '''
        num = 0
        if self.batch_count < self.num_batches:
            while num < self.batch_size:
                # do preprocessing here
                num +=1

            self.batch_count +=1
            return batch_spectrogram, batch_targets
        else:
            self.batch_count = 0
            # np.random.shuffle(self.annotations)
            raise StopIteration       # stops the iterator from continuing
        '''

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
            song_df['sample start'] = song_df['sample start'].apply(lambda val: val-song_df.at[0, 'sample start']) # realigning the sample start number to beginning of the reconstructed slice
            if self.data_aug:
                song_df = self.shift_augmentation(song_df)  # implemented my own shift augmentation function because it was simpler to move the entire df labels and samples together
            song = np.vstack(song_df['song slice'].to_numpy()).T   # stacks the song slices back into a single numpy array of shape (channels, samples)

            mono_song = lb.core.to_mono(song)

            channels = [mono_song]              # channels is a list of either [mono_song] or [mono, L_song, R_song]
            if INCLUDE_LR_CHANNELS:             # appending the LR channels to the channels variable
                channels.append(song[0,:])
                channels.append(song[1,:])

            # TODO: Get the correct sample rate (sr) from the song_info dictionary back in the creation of the MAT
                # For now, assume all sr's are = 44100
            if self.data_aug:
                channels = self.augment_audio_cp(channels, self.aug_comp, sr=SAMPLE_RATE)

            # make spectrogram
            spectrogram = self.create_spectrogram(channels, sr=SAMPLE_RATE)

            # make the targets using information of the spectrogram and the song_df
            target, target_dict = self.create_target(spectrogram, song_df)


        # TODO: Implement the case where the FullSet_memory =- False and we need to load the songs individually everytime
        else:     # case of not keeping FullSet in memory
            print('FULLSET_MEMORY == FALSE NOT IMPLEMENTED YET. EVERYTHING ELSE WILL NOT FUNCTION PROPERLY')
            spectrogram, target, target_dict = None, None, None

        return spectrogram, target, target_dict

    # START Helper Functions

    def get_song_list(self, dataset_type):
        '''
        Helper function to get the song list for the current Dataset type

        Args:
            dataset_type [str]: either 'train' or 'val'

        Returns:
            list: a list of strings that are the names of the song subfolders that exist for that dataset type
        '''
        if dataset_type == 'train':
            return [x.name for x in os.scandir(SONGS_PATH) if x.is_dir() and x.name not in VAL_SONG_LIST]
        if dataset_type == 'val':
            return [x.name for x in os.scandir(SONGS_PATH) if x.is_dir() and x.name in VAL_SONG_LIST]

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
        augmented_channels = [aug_comp(samples = channel, sample_rate = sr) for channel in channels]

        return augmented_channels

    # END Helper Functions

    def create_spectrogram(self, channels, sr):
        '''
        Makes a spectrogram based on the song channels given and the model options from the configs.py file

        Args:
            channels [list]: list of np.arrays that are the samples
            sr [int]: sample rate of the current song

        Returns:
            np.array: numpy array that is the spectrogram: either a n by m by 1 or n by m by 3 depending on the INCLUDE_LR_CHANNELS
        '''


        spectro_channels = [] # create either 1 spectrogram or 3 depending on how many channels are being used
        for channel in channels:
            spectro = lb.feature.melspectrogram(channel, sr=sr, n_fft = WINDOW_SIZE, hop_length = HOP_SIZE, center = False, n_mels = N_MELS) # numpy array of shape (n_mels, t)
            print(f'create_spectrogram: spectro.shape = {spectro.shape}')
            # potentially use np.asfortranarray(channel) if there is a weird error at this point
            if SHIFT_TO_DB:
                spectro = lb.power_to_db(spectro, ref = np.max)
            if INCLUDE_FO_DIFFERENTIAL:
                spectro_ftd = lb.feature.delta(spectro, order=1)    # calculate the first time derivative of the spectrogram
                    # spectro_f(irst)t(ime)d(erivative).shape = (n_mels, t) SAME AS spectro
                spectro = np.concatenate([spectro, spectro_ftd], axis = 0)    # first time derivative attached at end of normal log mel spectrogram (n_mels of spectro, then n_mels of ftd)
                    # spectro.shape = (2* n_mels, t)
            spectro_channels.append(spectro)

        spectrogram = np.stack(spectro_channels, axis = -1) # spectrogram channel dimension order is mono, L, R
        print(f'create_spectrogram: spectrogram.shape after ftd = {spectrogram.shape}')

        return spectrogram # spectrogram has shape of (n_mels (perhaps x2), t (determined by length of song and HOP_SIZE), n_channels (1 or 3) )

    def create_target(self, spectrogram, song_df):
        '''
        Creates the target labels from the tab dataframe that contains the one-hot encoded labels and aligns them to the spectrogram columns
        The model will have a 1:1 labeling format: for every frame ( or window) that goes in, a prediction on labels comes out.

        Args:
            spectrogram [np.array]: a spectrogram object that represents the X that will go into the model (n_mels, t, n_channels)
            song_df [Dataframe]: tab df that contains all the one-hot-encoded columns along with a 'sample start' column that contains the sample number start of that row w.r.t. the audio in the spectrogram

        Returns:
            np.array: the array of output targets [0,1]. Shape is
            dict: the target dictionary to map the output column to the correct label
        '''

        _, n_windows, n_channels = spectrogram.shape   # S.shape is (n_mels, t, n_channels), where t = number of frames/windows
        labels_df = song_df.drop(columns = ['sample start', 'song slice'], errors = 'ignore')   # get a dataframe that only has the labels
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
            if sample_starts[tab_slice_idx] < negative_window:   # if the sliding windows have passed the current tab slice
                tab_slice_idx +=1
            if tab_slice_idx >= len(sample_starts):
                break

        # create target_dictionary that maps the indices to the column name
        target_dictionary = {idx : val for idx, val in enumerate(class_names)}

        return targets, target_dictionary

def one_hot_encode(df):
    """
    Encoder for encoding the class labels of a tab dataframe. Encoded as a multi-label one hot vector: that is, a one hot vector that can be 1 in multiple classes for each example
    Note that the labels from the time-keeping line are classifed as "beats" and "downbeats", not "offbeats" and "downbeats"
    Note that the labels names are "encoded" into the column names of the dataframe, regardless of how many labels you might have in each column originally

    Args:
        df [Dataframe]:a dataframe that has been passed through the cleaning and collapsing classes functions already

    Returns:
        Dataframe: an encoded version of the dataframe
    """
    tk_label, measure_char, blank_char = TK_LABEL, MEASURE_CHAR, BLANK_CHAR # get blank_char mainly

    col_list = list(df.drop(columns = ['song slice', 'sample start'], errors = 'ignore').columns) # if 'ignore', suppress error and only existing labels are dropped
    print(f'one_hot_encode: col_list before encoding = {col_list}')
    for col in col_list:                     # goes through all the column names except 'song slice' and 'sample start'
        uniques = [uniq for uniq in df[col].unique() if uniq is not blank_char] # list of unique values found in current column
        if col == 'tk':                   # we are in the time-keeping column case, so we'll hard code handle this
            df['tk_beat'] = df['tk'].apply(lambda row1: 1 if row1 != blank_char else 0)   # create a new tk_beat column if there is any non blank char
            df['tk_downbeat'] = df['tk'].apply(lambda row2: 1 if row2 == DOWNBEAT_CHAR else 0)    # create a new tk_downbeat column where there is capital C, the previously HARD CODED downbeat label
        else:     # we are in all the other column cases, so we have to generically code this
            for label in uniques:   # we treat all the different labels as separate, regardless of how many we have on each line
                df[col + '_' + label] = df[col].apply(lambda row3: 1 if row3 == label else 0)  # create a column named like "BD_o" where it is 1 any time you have that label and 0 elsewhere
    df_encoded = df.drop(columns = col_list)    # drop the columns because we no longer need it as it has been encoded properly
    print(f'one_hot_encode: col_list after encoding = {list(df_encoded.columns)}')

    return df_encoded

def clean_labels(df):
    '''
    Cleans the labels by replacing errors, common mistakes or different notations for labels that already exist

    Args:
        df [Dataframe]: Dataframe object that is a music aligned tab, or a FullSet tab, or some slice of a music aligned tab

    Returns:
        Dataframe: the dataframe but with the labels cleaned up according to the code in here
    '''

    master_format_dict = MASTER_FORMAT_DICT# grabs the dict of the master format to be used here
    tk_label, measure_char, blank_char = TK_LABEL, MEASURE_CHAR, BLANK_CHAR  # get blank_char mainly

    replace_dict = {}   # build up this dict to use as the replace in a later df.replace function
    for drum_chars in master_format_dict.values():
        replace_dict[drum_chars] = {}    # create an empty dict object for each drum char in master format dict so I can later use .update method always

    # get useful, specific subsets of the column names (2 chars) used in the FullSet dataframe, as dictated by the master_format_dict, ensuring that they are in the FullSet_df column names
    cymbals = [master_format_dict[x] for x in master_format_dict.keys() if 'cymbal' in x and master_format_dict[x] in df.columns]  # does NOT include hihat
    hihat = master_format_dict['hi-hat']
    snare = master_format_dict['snare drum']
    ride = master_format_dict['ride cymbal']
    drums = [master_format_dict[x] for x in master_format_dict.keys() if ('drum' in x or 'tom' in x) and master_format_dict[x] in df.columns]   # includes both drums and toms

    """CLEAN UP 1: get rid of the "grab cymbal to stop sustain" notation in cymbal line tabs for all cymbals"""
    for cymbal in cymbals:
        replace_dict[cymbal].update({'#':blank_char})  # Constructing a dict where {column_name : {thing_to_be_replaced: value_replacing} }

    """CLEAN UP 2: get rid of the 'f', 's', and 'S' on the 'HH' column (usually denotes foot stomp on hihat pedal)"""
    replace_dict[hihat].update({'f':blank_char, 's':blank_char, 'S' : blank_char})

    """CLEAN UP 3: replace the washy 'w' and 'W' with the normal washy hi-hat notation 'X'  (overall inconsistent notation but consistent enough to map properly)"""
    replace_dict[hihat].update({'w': 'X', 'W':'X'})

    """CLEAN UP 4: get rid of 'r' on the 'SD' column (rimshots on the snare drum) and change 'x' to 'o' (sometimes used in drum solos for easier reading)"""
    replace_dict[snare].update({'r' : blank_char, 'x' : 'o', '0' : 'O'})

    """CLEAN UP 5: get rid of doubles notation ('d') and flams ('f'), and replace them with equivalent single hits"""
    for drum in drums:
        replace_dict[drum].update({'d' : 'o', 'D' : 'O', 'f' : 'o'})
    replace_dict[hihat].update({'d' : 'x', 'f' : 'x'})
    replace_dict[ride].update({'d' : 'x', 'f' : 'x'})

    """CLEAN UP 6: On hihat line, O and o are going to sound ~the same regardless of actual dynamic strength of hit."""
    replace_dict[hihat].update({'O': 'o'})

    """CLEAN UP 7: Replace m-dashes used in place of the blank char (here, n dash) in every column"""
    for col in master_format_dict.values():
        replace_dict[col].update({'—' : blank_char})

    df = df.replace(to_replace = replace_dict, value = None) # do the replacing using the replace_dict

    return df

def collapse_class(FullSet_df, keep_dynamics = False, keep_bells = False, keep_toms_separate = False, hihat_classes = 1, cymbal_classes = 1):
    """
    Collapses the class labels in the FullSet dataframe to the desired amount of classes for output labels in Y.
    Note that all of the collapsing choices will exist inside this function. There won't be a different place or prompt that
    allows the classes to be customized further. This is where the class decisions making is occurring, HARD CODED into the function
    Note that derived classes will be entirely lower case in the column names, where as normal classes will be entirely upper case

    Args:
        FullSet_df [Dataframe]: the entire set of music aligned tabs in one dataframe, cleaned up at this point
        keep_dynamics [bool]: Default False. If False, collapses the dynamics labels into one single label (normally, capital vs. lower case). If True, don't collapse, effectively keeping dynamics as classes
        keep_bells [bool]: Default False. If False, changes the bells into blank_char, effectively getting rid of them and ignoring their characteristic spectral features.
                           If True, still changes them into blank_char, but create a new column in the dataframe called 'be' that places them in there
        keep_toms_separate [bool]: Default False. If False, collapses the toms into one single tom class. If True, keep the toms labels separate and have multiple tom class
        hihat_classes [int]: Default 1. Hihats have two, or arguably three, distinct classes. One class is the completely closed hihat hit that is a "tink" sound.
                            A second very common way to play hihat is called "washy" where the two hihats are slightly open and can interact with each other after being hit
                            A third class is the completely open hihat, where the top hihat doesn't interact with the bottom at all. This is similar to a cymbal hit
                            Default 2 classes splits
        cymbal_classes [int]: Default 1. Cymbals come in many sizes, tones, and flavors. The most reasonable thing to do is to collapse all cymbals into one class
                              But what about the Ride cymbal? which normally is not "crashed" but hit like the hihat
                              If == 2, Ride will be split out of the rest of the crash cymbals
                              If == -1, keep all cymbal classes intact (generally for debugging)
    Returns:
        Dataframe: the FullSet dataframe but with classes collapsed, which most likely means that certain columns will be gone and new columns will be present
    """

    master_format_dict = MASTER_FORMAT_DICT # grabs the dict of the master format to be used here
    tk_label, measure_char, blank_char = TK_LABEL, MEASURE_CHAR, BLANK_CHAR  # get blank_char mainly

    # get useful, specific subsets of the column names (2 chars) used in the FullSet dataframe, as dictated by the master_format_dict, ensuring that they are in the FullSet_df column names
    drums = [master_format_dict[x] for x in master_format_dict.keys() if ('drum' in x or 'tom' in x)  and master_format_dict[x] in FullSet_df.columns]  # drums AND toms in this list
    cymbals = [master_format_dict[x] for x in master_format_dict.keys() if 'cymbal' in x and master_format_dict[x] in FullSet_df.columns]         # Notably EXCLUDING hi-hat
    toms = [master_format_dict[x] for x in master_format_dict.keys() if 'tom' in x and master_format_dict[x] in FullSet_df.columns]                 # toms ONLY
    hihat = master_format_dict['hi-hat']            # get the label for the hi-hat column from master_format_dict

    """HIHAT - determine the number of classes desired in the hi-hat line. CRITICAL that this occurs before CYMBALS"""
    FullSet_df = FullSet_df.replace(to_replace = {hihat: {'g': blank_char}}, value = None)  # gets rid of ghost notes on the hihat no matter how many classes are chosen
    if hihat_classes == 2 or hihat_classes == 1:    # with only 1 or 2 classes, the washy ('X') and open ('o') hits are combined into one ('X') on the same line
        FullSet_df = FullSet_df.replace(to_replace = {hihat: {'o': 'X'}}, value = None) # replaces all 'o' with 'X' in the hihat column
        if hihat_classes == 1:                      # with only one class, need to keep the closed hi-hat ('x') on its own column, and then move the open 'o' and washy 'X' into the Crash Cymbal ('CC') column
            FullSet_df.loc[FullSet_df[hihat] == 'X', master_format_dict['crash cymbal']] = FullSet_df.loc[FullSet_df[hihat] == 'X', hihat]  # sets the values in the CC column, in the rows where the hihat == 'X', to the values that are in the hihat column of those rows
            FullSet_df = FullSet_df.replace(to_replace = {hihat: {'X': blank_char}}, value = None) # rids the hihat column of the 'X's that have been moved to the CC column
    if hihat_classes == 3:
        None            # keep the expected notations of 'x' for closed, 'X' for washy, and 'o' for completely open

    """DYNAMICS - Making everything lower case that needs to be, and get rid of ghost notes; doesn't touch the hihat"""
    if not keep_dynamics:   # in the case where the dynamics are NOT kept. That is, this code should collapse the dynamics
        replace_dict = {}   # build up this dict to use as the replace in a later df.replace function
        for element in drums + cymbals:
            replace_dict[element] = {'X':'x', 'O':'o', 'g':blank_char} # prepare to search for X to replace with x, and O to replace with o whenever applicable
        FullSet_df = FullSet_df.replace(to_replace = replace_dict, value = None) # do the replacing using the replace_dict

    """BELLS - get rid of bell hits entirely or move them into a new column"""
    if not keep_bells:          # in the case where bell hits are thrown away
        FullSet_df = FullSet_df.replace(to_replace = 'b', value = blank_char) # NOTE: replaces 'b' ANYWHERE in the dataframe labels with the blank_char
    else:                       # in the case where bell hits are moved to a new column and replaced with blank_char after that
        FullSet_df['be'] = blank_char  # new bell column is titled 'be' for 'bell' and is initially all blank_char
        replace_dict = {}
        for cymbal in cymbals:
            FullSet_df.loc[FullSet_df[cymbal] == 'b','be'] = FullSet_df.loc[FullSet_df[cymbal] == 'b', cymbal]
            replace_dict[cymbal] = {'b':blank_char}
        FullSet_df = FullSet_df.replace(to_replace = replace_dict, value = None) # do the replacing using the replace_dict

    """TOMS - keep toms as their own classes or collapse into one"""
    if not keep_toms_separate:         # in the case where toms are collapsed into one class
        FullSet_df['at'] = blank_char    # The new column is titled 'at' for 'all toms' as it represents the labels of all the toms at once, initially set to the blank_char for all rows
        for tom in toms:
            FullSet_df.loc[FullSet_df[tom] != blank_char,'at'] = FullSet_df.loc[FullSet_df[tom] != blank_char, tom]  #  finds all the rows where a tom event occurs, and the value of those rows in the tom event
        FullSet_df = FullSet_df.drop(columns = toms)  # drop the original toms columns

    """CYMBALS - determine the number of cymbal classes"""
    if cymbal_classes == 1:   # the case where we collapse all the cymbal classes down to one class
        FullSet_df['ac'] = blank_char    # new column is titled 'ac' for 'all cymbals' as it represents the labels of all the cymbals at once
        for cymbal in cymbals:
            FullSet_df.loc[FullSet_df[cymbal] != blank_char, 'ac'] = FullSet_df.loc[FullSet_df[cymbal] != blank_char, cymbal]
        FullSet_df = FullSet_df.drop(columns = cymbals)
    if cymbal_classes == 2: # the case where we collapse all the cymbal classes except the ride cymbal down to one class
        FullSet_df['mc'] = blank_char   # new column is titled 'mc' for 'most cymbals' as it represents most cymbals
        most_cymbals = [x for x in cymbals if x != master_format_dict['ride cymbal']]   # grabbing all the cymbals not the ride cymbal
        for cymbal in most_cymbals:
            FullSet_df.loc[FullSet_df[cymbal] != blank_char, 'mc'] = FullSet_df.loc[FullSet_df[cymbal] != blank_char, cymbal]
        FullSet_df = FullSet_df.drop(columns = most_cymbals)    # drop the cymbal columns no longer needed
    if cymbal_classes == -1:
        pass    # used for debugging, keeps the full cymbals set for further inspection

    """BEATS AND DOWNBEATS - change the time-keeping line notation to denote downbeats and other beats"""
    non_digits = [x for x in FullSet_df['tk'].unique() if not x.isdigit()]         # finds all non-digit values used in the tk column
    non_ones_digits = [x for x in FullSet_df['tk'].unique() if x.isdigit() and x != '1'] # finds all digit values that are not equal to 1
    replace_dict = {'tk': {}}     # create empty dict for the 'tk' column
    for el in non_digits:
        replace_dict['tk'].update({el: blank_char})    # replacing non-digits elements in tk column for blank_chars
    for el in non_ones_digits:
        replace_dict['tk'].update({el: BEAT_CHAR})           # replacing non-ones digits elements in tk column for BEAT_CHAR = 'c', which stands for 'click', as if you were listening to a metronome hearing clicks on the beats
    replace_dict['tk'].update({'1': DOWNBEAT_CHAR})              # DOWNBEAT_CHAR = 'C' stands for 'Click', a louder click from a metronome, used to denote the downbeat
    FullSet_df = FullSet_df.replace(to_replace = replace_dict, value = None)

    return FullSet_df
