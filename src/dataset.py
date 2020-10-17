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
import librosa.display
import numpy as np
import audiomentations as adm

from src.configs import *


# defining the Dataset class
class Dataset(object):
    def __init__(self, dataset_type):   # dataset_type = 'train' or 'val'
        self.song_list = self.get_song_list(dataset_type)# train and validation subfolders will be contained in this folder file path form configs
        self.data_aug = TRAIN_DATA_AUG if dataset_type == 'train' else VAL_DATA_AUG   # boolean from configs


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
                # do parsing and preprocessing here
                num +=1

            self.batch_count +=1
            return batch_spectrogram, batch_targets
        else:
            self.batch_count = 0
            # np.random.shuffle(self.annotations)
            raise StopIteration       # stops the iterator from continuing
        '''

    # Helper Functions
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
    print(f'create_spectrogram: lb_song.shape = {lb_song.shape}')
    print(f'create_spectrogram: sr = {sr}')

    # DATA AUGMENTATIONS APPLIED HERE
    # lb_song = apply_augmentations(lb_song)

    # create mono spectro
    lb_mono = lb.core.to_mono(lb_song)          # returns a numpy array of shape (n,) that is the mono of the loaded song
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

    return spectrogram # spectrogram has shape of (n_mels (perhaps x2), t (determined by length of song and HOP_SIZE), n_channels (1 or 3) )

def create_targets(spectrogram, df):
    '''
    Creates the target labels from the tab dataframe that contains the one-hot encoded labels and aligns them to the spectrogram columns
    The model will have a 1:1 labeling format: for every frame ( or window) that goes in, a prediction on labels comes out.

    Args:
        spectrogram [np.array]: a spectrogram object that represents the X that will go into the model (n_mels, t, n_channels)
        df [Dataframe]: tab df that contains all the one-hot-encoded columns along with a 'sample start' column that contains the sample number start of that row w.r.t. the entire song
    '''

    _, n_windows, n_channels = spectrogram.shape   # S.shape is (n_mels, t, n_channels), where t = number of frames/windows
    df_labels = df.drop(columns = ['sample start', 'song slice'], errors = 'ignore')   # get a dataframe that only has the labels
    class_names = df_labels.columns
    n_classes = len(class_names)
    targets = np.zeros((n_classes, n_windows, n_channels), dtype=int)  # initializes as all zeros in shape of (n_classes number of rows, n_frames number of columns)
    sample_starts = list(df['sample start'])


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
            targets[:, idx, :] = np.stack([df_labels.loc[tab_slice_idx].to_numpy() for _ in range(n_channels)], axis=-1)
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
    print(f'one_hot_encode: col_list = {col_list}')
    for col in col_list:                     # goes through all the column names except 'song slice' and 'sample start'
        uniques = [uniq for uniq in df[col].unique() if uniq is not blank_char] # list of unique values found in current column
        if col == 'tk':                   # we are in the time-keeping column case, so we'll hard code handle this
            df['tk_beat'] = df['tk'].apply(lambda row1: 1 if row1 != blank_char else 0)   # create a new tk_beat column if there is any non blank char
            df['tk_downbeat'] = df['tk'].apply(lambda row2: 1 if row2 == DOWNBEAT_CHAR else 0)    # create a new tk_downbeat column where there is capital C, the previously HARD CODED downbeat label
        else:     # we are in all the other column cases, so we have to generically code this
            for label in uniques:   # we treat all the different labels as separate, regardless of how many we have on each line
                df[col + '_' + label] = df[col].apply(lambda row3: 1 if row3 == label else 0)  # create a column named like "BD_o" where it is 1 any time you have that label and 0 elsewhere
    df_encoded = df.drop(columns = col_list)    # drop the columns because we no longer need it as it has been encoded properly

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
        None    # used for debugging, keeps the full cymbals set for further inspection

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
