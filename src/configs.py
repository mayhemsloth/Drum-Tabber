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
N_MELS      = 100        # number of mel bins to be created in the spectrograms
WINDOW_SIZE = 4096      # number of samples large that each spectro slice is. At 2048 and 44100 Hz sample rate, each window is 46 ms
HOP_SIZE    = 441        # number of samples to hop each time when creating the spectrogram. 441 gives a 10ms hop size. That is, you produce a window every 10 ms
FMAX        = 10000      # in Hz, the maximum frequency that the mel spectrogram outputs; if None, function uses sr / 2.0
SHIFT_TO_DB = True       # changes the power spectrum to db instead of... whatever it is in when you get the output from lb.melspectrogram
INCLUDE_FO_DIFFERENTIAL = False  # keeps the first order differential over time of the spectrograms
POSITIVE_WINDOW_FRACTION = 0.2   # this number denotes the fraction (of the WINDOW_SIZE) of the first part of any frame to determine if a frame is labeled with that drum note onset sample
NEGATIVE_WINDOW_FRACTION = 0.1   # this number denotes the fraction (of the WINDOW_SIZE) of the "negative" part of any frame to determine if the frame is labeled with that drum note onset sample
MODEL_TYPE = 'Context-CNN' # the model type desired to build. Possible choices are 'Context-CNN'
N_CONTEXT_PRE  = 13    # the number of context windows included before the target window in any context model type
N_CONTEXT_POST = 13    # the number of context windows included after the target window in any context model type
TOLERANCE_WINDOW = 30  # in ms, the amount of time that is allowable left and right of sample labelled as correct. Note that a 200 BPM 16th note grid corresponds to 75 ms duration. 150 BPM is 100 ms duration


SAMPLE_RATE = 44100  # need to delete this whenever I finally implement the sr carryover from the song loading in

# classification options. See clean_labels and collapse_class functions for full functionality
CLEAN_DATA         = True     # This should always be true, there's really no reason to not clean the labels
KEEP_DYNAMICS      = False    # If False, gets rid of dynamics of certain drum tab notation ('O' goes to 'o', 'X' goes to 'x', etc.). If True, keeps all dynamics
KEEP_BELLS         = False    # If False, gets rid of bell hits by setting to blank char. If True, splits bell hits into its own class 'be'
KEEP_TOMS_SEPARATE = False    # If False, collapses all toms into a single class 'at' for 'all toms'. If True, keeps toms the same way and in separate classes
HIHAT_CLASSES      = 1        # 1 is only closed HH ('x'), others moved to crash cymbal. 2 is closed ('x') and a combined open and washy into one ('X'). 3 is keep all closed ('x'), washy ('X'), and opened ('o')
CYMBAL_CLASSES     = 1        # 1 is all cymbals, including ride, to one class 'ac'. 2 is all crash cymbals to one class 'mc', and ride gets split out. -1 is no cymbals get affected, for debugging

# train options
SONGS_PATH = 'C:/Users/Thomas/Python Projects/Drum-Tabber-Support-Data/Songs'    # the absolute filepath to the folder containing all the songs data structured in the correct way with song subfolders
SONGS_TO_TAB_PATH = 'C:/Users/Thomas/Python Projects/Drum-Tabber-Support-Data/Songs-to-Tabs'  # absolute filepath to the folder containing songs that can be converted into tabs after having a trained model
SAVED_MODELS_PATH = 'C:/Users/Thomas/Python Projects/Drum-Tabber/models/saved_models'   # absolute filepath to the folder containing the saved models
INCLUDE_LR_CHANNELS            = False           # if true, uses the Left and Right channels as their own mono channel to include in the data set (whichever data set that is)
TRAIN_SAVE_CHECKPOINT_MAX_BEST = True   # if true, saves only the absolute best model according to the validation loss (will overwrite the previous max best model)
TRAIN_SAVE_CHECKPOINT_ALL_BEST = False  # if true, saves all best validation checkpoints in the training process
TRAIN_FULLSET_MEMORY           = True            # if true, utilizes the FullSet dataframe in memory to continuously pull from during training/val. ASSUMES FullSet (all songs) can be held in memory
TRAIN_LOGDIR                   = 'logs'
TRAIN_CHECKPOINTS_FOLDER       = 'models/checkpoints'
TRAIN_FROM_CHECKPOINT          = False
TRAIN_CHECKPOINT_MODEL_NAME    = ''


TRAIN_BATCH_SIZE      = 256       # the number of individual images (slices of the spectrogram: windows and their contexts) before the model is updated
TRAIN_LR_INIT         = 1e-4
TRAIN_LR_END          = 1e-6
TRAIN_WARMUP_EPOCHS   = 1
TRAIN_EPOCHS          = 10


# augmentation options
TRAIN_DATA_AUG             = False
BACKGROUNDNOISES_PATH      = "C:/Users/Thomas/Python Projects/Drum-Tabber-Support-Data/BackgroundNoises/normalized"    # the absolute filepath to the folder containing the BackgroundNoises used to add noises
SHIFT_CHANCE               = 0.5   # half of songs will start at a random point instead of the beginning, but still include the full song (wrapped around to end)
POLARITY_CHANCE            = 0.5   # half of the songs will be flipped upside. Should produce more varied samples when adding things
FREQUENCY_MASK_CHANCE      = 0.25
GAUSSIAN_SNR_CHANCE        = 0.25
GAUSSIAN_NOISE_CHANCE      = 0.25
PITCH_SHIFT_CHANCE         = 0.25
NORMALIZE_CHANCE           = 0.25
CLIPPING_DISTORTION_CHANCE = 0.25
BACKGROUND_NOISE_CHANCE    = 0.25
GAIN_CHANCE                = 0.25
MP3_COMPRESSION_CHANCE     = 0.25
BIN_DROPOUT_CHANCE         = 0.1
BIN_DROPOUT_RATIO          = 0.05  # percentage of bins that will be set to 0 if bin dropout chance is successful
S_NOISE_CHANCE             = 0.1
S_NOISE_RANGE_WIDTH        = 0.1   # width of range of numbers around 1 that S_noise will choose from to multiply by to create noise (0.1 ==> pulls from 0.95 to 1.05)


# validation options
VAL_DATA_AUG         = False
VAL_SONG_LIST        = ['misery_business', 'four_years']     # the songs that will be not used in the training set but instead in the validation set
VAL_BATCH_SIZE       = 256


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
BEAT_CHAR         = 'c'     # the character used to denote a beat in the 'tk' line when cleaning labels
DOWNBEAT_CHAR     = 'C'     # the character used to denote a downbeat in the 'tk' line when cleaning labels

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
                      'SC': 11,
                      'CH': 10
                      }
