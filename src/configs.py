#================================================================
#
#   File name   : configs.py
#   Author      : Thomas Hymel
#   Created date: 9-2-2020
#   GitHub      : https://github.com/mayhemsloth/Drum-Tabber
#   Description : Configuration File to declare "global" variables
#
#================================================================

# model options
N_MELS      = 150        # number of mel bins to be created in the spectrograms
WINDOW_SIZE = 2048  # number of samples large that each spectro slice is. At 2048 and 44100 Hz sample rate, each window is 46 ms
HOP_SIZE    = 441      # number of samples to hop each time when creating the spectrogram. 441 gives a 10ms hop size. That is, you produce a window every 10 ms
INCLUDE_FO_DIFFERENTIAL = True  # keeps the first order differential over time of the spectrograms

# classification options. See clean_labels and collapse_class functions for full functionality
CLEAN_DATA         = True
KEEP_DYNAMICS      = False
KEEP_BELLS         = False    # If False, gets rid of bell hits by setting to blank char. If True, splits bell hits into its own class 'be'
KEEP_TOMS_SEPARATE = False    # If False, collapses all toms into a single class 'at' for 'all toms'
HIHAT_CLASSES      = 1        # 1 is only closed HH ('x'), others moved to crash cymbal. 2 is closed ('x') and a combined open and washy into one ('X'). 3 is keep all closed ('x'), washy ('X'), and opened ('o')
CYMBAL_CLASSES     = 1        # 1 is all cymbals, including ride, to one class 'ac'. 2 is all crash cymbals to one class 'mc', and ride gets split out. -1 is no cymbals get affected, for debugging

# train options
SONGS_PATH = "C:/Users/Thomas/Python Projects/Drum-Tabber-Support-Data/Songs"    # the relative filepath to the folder containing all the songs data
INCLUDE_LR_CHANNELS = True              # if true, uses the Left and Right channels as their own mono channel to include in the data set (whitchever data set that is)
TRAIN_SAVE_CHECKPOINT_MAX_BEST = True   # if true, saves only the absolute best model according to the validation loss (will overwrite the previous max best model)
TRAIN_SAVE_CHECKPOINT_ALL_BEST = False  # if true, saves all best validation checkpoints in the training process


# augmentation options
TRAIN_DATA_AUG = True



# validation options
VAL_DATA_AUG = False



# pre-processing and formatting options
MASTER_FORMAT_DICT = {'bass drum'      : 'BD',     # the two-letter char codes that all drum pieces in tabs will convert to
                      'snare drum'     : 'SD',
                       'high tom'      : 'HT',
                       'mid tom'       : 'MT',
                       'low tom'       : 'LT',
                       'hi-hat'        : 'HH',
                       'ride cymbal'   : 'RD',
                       'crash cymbal'  : 'CC',
                       'crash cymbal 2': 'C2',     # extra cymbals if needed, classification will ultimately collapse later
                       'crash cymbal 3': 'C3',     # extra cymbals if needed, classification will ultimately collapse later
                       'crash cymbal 4': 'C4',     # extra cymbals if needed, classification will ultimately collapse later
                       'splash cymbal' : 'SC',
                       'china cymbal'  : 'CH'
                       }

TK_LABEL          = '  '    # time-keeping label, or start of a time-keeping line, denoted by two spaces
MEASURE_CHAR      = '|'     # the character used to denote the measures in a text tabber
BLANK_CHAR        = '-'     # the character used to denote a blank in a drum line
QUARTER_TRIPLET   = 'tq'    # 2-char string that denotes when quarter note triplet occurs in the tk line
EIGHTH_TRIPLET    = 'te'    # 2-char string that denotes when eighth note triplet occurs in the tk line
SIXTEENTH_TRIPLET = 'ts'    # 2-char string that denotes when sixteenth note triplet occurs in the tk line

DESIRED_ORDER_DICT = {'BD': 1,     # the order desired by the user to display the drum piece lines in, from bottom to top
                      'SD': 2,
                      'HT': 9,
                      'MT': 8,
                      'LT': 7,
                      'HH': 3,
                      'RD': 4,
                      'CC': 5,
                      'C2': 6,     # extra cymbals if needed, classification will ultimately collapse later
                      'C3': 12,     # extra cymbals if needed, classification will ultimately collapse later
                      'C4': 13,     # extra cymbals if needed, classification will ultimately collapse later
                      'SC':11,
                      'CH':10
                      }
