#================================================================
#
#   File name   : configs.py
#   Author      : Thomas Hymel
#   Created date: 9-2-2020
#   GitHub      : https://github.com/mayhemsloth/Drum-Tabber
#   Description : Configuration File to declare "global" variables
#
#================================================================

''' core feature/labeling preparation options '''
N_MELS      = 250     # number of mel bins to be created in the spectrograms
WINDOW_SIZE = 2048    # number of samples large that each spectro slice is. At 2048 and 44100 Hz sample rate, each window is 46 ms
HOP_SIZE    = 441     # number of samples to hop each time when creating the spectrogram. 441 gives a 10ms hop size. That is, you produce a window every 10 ms
FMAX        = None   # in Hz, the maximum frequency that the mel filter-bank (spectrogram) outputs; if None, function uses sr / 2.0
INCLUDE_FO_DIFFERENTIAL = False  # keeps the first order differential over time of the spectrograms
NEGATIVE_WINDOW_FRACTION = 0.15   # this number denotes the fraction (of the WINDOW_SIZE) of the "negative" part of any frame to determine if the frame is labeled with that drum note onset sample
POSITIVE_WINDOW_FRACTION = 0.15   # this number denotes the fraction (of the WINDOW_SIZE) of the first part of any frame to determine if a frame is labeled with that drum note onset sample

SAMPLE_RATE = 44100  # TODO: need to delete this whenever I finally implement the sr carryover from the song loading in


MODEL_TYPE = 'TimeFreq-CNN'    # the model type desired to build. Possible choices are 'Context-CNN', 'TimeFreq-CNN', 'TL-DenseNet121/169/201'
''' CNN model options '''
N_CONTEXT_PRE  = 15    # the number of context windows included before the target window in any context model type
N_CONTEXT_POST = 15    # the number of context windows included after the target window in any context model type
TOLERANCE_WINDOW = 20  # in ms, the amount of time that is allowable left and right of sample labelled as correct. Note that a 200 BPM 16th note grid corresponds to 75 ms duration. 150 BPM is 100 ms duration
SHIFT_TO_DB = True       # changes the power spectrum to db instead of... whatever it is in when you get the output from lb.melspectrogram

''' time series transformer options '''
D_FEATURES_IN = 250     # dimensionality of the input sequence
LEN_SEQ = 32            # number of time steps in each input sequence - at the moment, expected that information is fully known and no padding needed
D_MODEL = 128           # internal dimension of learned features
N_HEADS = 2             # number of heads in the multihead attention. Note D_MODEL must be divisible by N_HEADS
N_ENCODER_LAYERS = 3    # number of sequential encoder layer blocks
D_FFN = 256             # number of nodes in each feed forward network layer inside transformer encoder layer
ACTIV = 'gelu'          # the activation function used for the transformer network. Should be a tensorflow compatible string activation function
MHA_BIAS = False        # if False, does not use a bias vector for multihead attention layer. If True, does use a bias vector
ATTENTION_DROPOUT_P = 0.1  # probability of dropout in the attention network dropout layer
FFN_DROPOUT_P = 0.1     # probability of dropout in the feed forward dropout layer

''' training and output options '''
D_OUT = 10                   # The expected number of classes/variables for output
OUTPUT_TYPE = 'multilabel'   # options are 'regr', 'softmax', or 'multilabel'
CUSTOM_HEAD = None           # list of tensorflow layers to be used as a custom prediction head
SELF_SUPERVISED_TRAINING = True  # if True, self-supervised training occurs. If False, fine-tune training (full network training) occurs instead
SELF_SUPERVISED_TRAIN_EPOCHS = 10  # number of epochs during self-supervise training

''' self-supervised mask options '''
MASK_TYPE = 'seq'           # valid options are: 'seq' (random time sequences), 'feature' (entire rows), 'time' (entire columns), 'forecast' (last columns), 'noise' (random throughout)
MASK_R = 0.15               # roughly the proportion of values that are masked for each sample during self-supervised training
MASK_LM = 4                 # for 'seq' option, the average length of masked sequences
MASK_RANDOM_TYPE_LIST = None # a list of strings corresponding to the set of desired mask types, one of which is randomly chosen per batch
MASK_ALL_BATCH_SAME = False  # if True, all samples in a batch will have the same mask (the first of the batch). If False, random mask is produced for each sample


''' classification options. See clean_labels and collapse_class functions for full functionality explanation '''
CLEAN_DATA         = True     # This should always be true, there's really no reason to not clean the labels
KEEP_DYNAMICS      = False    # If False, gets rid of dynamics of certain drum tab notation ('O' goes to 'o', 'X' goes to 'x', etc.). If True, keeps all dynamics
KEEP_BELLS         = False    # If False, gets rid of bell hits by setting to blank char. If True, splits bell hits into its own class 'be'
KEEP_TOMS_SEPARATE = False    # If False, collapses all toms into a single class 'at' for 'all toms'. If True, keeps toms the same way and in separate classes
HIHAT_CLASSES      = 1        # 1 is only closed HH ('x'), others moved to crash cymbal. 2 is closed ('x') and a combined open and washy into one ('X'). 3 is keep all closed ('x'), washy ('X'), and opened ('o')
CYMBAL_CLASSES     = 1        # 1 is all cymbals, including ride, to one class 'ac'. 2 is all crash cymbals to one class 'mc', and ride gets split out. -1 is no cymbals get affected, for debugging
SIMPLE_CLASS_ONLY  = False    # If True, drops all classes except SD, BD, and Cymbals

''' train options '''
SONGS_PATH = '/content/gdrive/My Drive/Drum-Tabber-Support-Data/Songs'    # the absolute filepath to the folder containing all the songs data structured in the correct way with song subfolders
SONGS_TO_TAB_PATH = 'C:/Users/Thomas/Python Projects/Drum-Tabber-Support-Data/Songs-to-Tabs'  # absolute filepath to the folder containing songs that can be converted into tabs after having a trained model
SAVED_MODELS_PATH = '/content/gdrive/My Drive/Drum-Tabber/models/saved_models'   # absolute filepath to the folder containing the saved models
SPECTROS_PATH = '/content/gdrive/My Drive/Drum-Tabber-Support-Data/Music-Spectro-Library/Spectrograms'
TRAIN_SAVE_CHECKPOINT_MAX_BEST = True       # if true, saves only the absolute best model according to the validation loss (will overwrite the previous max best model)
TRAIN_SAVE_CHECKPOINT_ALL      = False      # if true, saves all validation model checkpoints in the training process
TRAIN_FULLSET_MEMORY           = True       # if true, utilizes the FullSet dataframe in memory to continuously pull from during training/val. ASSUMES FullSet (all songs) can be held in memory
TRAIN_LOGDIR                   = 'logs'
TRAIN_CHECKPOINTS_FOLDER       = 'models/checkpoints'
TRAIN_FROM_CHECKPOINT          = False
TRAIN_CHECKPOINT_MODEL_NAME    = ''
TRAIN_FINE_TUNE                = False
# Spleeter train options
TRAIN_USE_DRUM_STEM           = True     # if true, use the drum stem slices from the MAT_df to help with training the model
TRAIN_INCLUDE_DRUM_STEM       = True     # if true, uses the separated out drum stem and then append it as an additional channel
TRAIN_INCLUDE_MIXED_STEM      = False     # if true, uses the seperated out drum stem and then mix it with original mix to accentuate drums, then append as additional channel
TRAIN_MIXED_STEM_WEIGHTS      = (0.5,0.5) # the weights multiplied by the full mix and drum mix respectively when added together
TRAIN_REPLACE_WITH_MIXED_STEM = False      # if true, replaces the normal full song channel with the mixed stem

    # Depracated train options INCLUDE_LR_CHANNELS = False # if true, uses the Left and Right channels as their own mono channel to include in the data set (whichever data set that is)

TRAIN_BATCH_SIZE      = 256      # the number of individual images (slices of the spectrogram: windows and their contexts) before the model is updated
TRAIN_LR_INIT         = 1e-4
TRAIN_LR_END          = 5e-6
TRAIN_WARMUP_EPOCHS   = 1
TRAIN_EPOCHS          = 50


''' augmentation options '''
TRAIN_DATA_AUG             = True
BACKGROUNDNOISES_PATH      = "/content/gdrive/My Drive/Drum-Tabber-Support-Data/BackgroundNoises/normalized"    # the absolute filepath to the folder containing the BackgroundNoises used to add noises
SHIFT_CHANCE               = 0.5   # half of songs will start at a random point instead of the beginning, but still include the full song (wrapped around to end)
POLARITY_CHANCE            = 0.5   # half of the songs audiowave will be flipped upside. Should produce more varied samples when adding things
FREQUENCY_MASK_CHANCE      = 0.25
GAUSSIAN_SNR_CHANCE        = 0.25
GAUSSIAN_NOISE_CHANCE      = 0.25
PITCH_SHIFT_CHANCE         = 0.25
NORMALIZE_CHANCE           = 0.25
CLIPPING_DISTORTION_CHANCE = 0.25
BACKGROUND_NOISE_CHANCE    = 0.25
GAIN_CHANCE                = 0.25
MP3_COMPRESSION_CHANCE     = 0.25
BIN_DROPOUT_CHANCE         = 0.25
BIN_DROPOUT_RATIO          = 0.05  # percentage of bins that will be set to 0 if bin dropout chance is successful
S_NOISE_CHANCE             = 0.05
S_NOISE_RANGE_WIDTH        = 0.1   # width of range of numbers around 1 that S_noise will choose from to multiply by to create noise (0.1 ==> pulls from 0.95 to 1.05)


''' validation options '''
VAL_DATA_AUG         = True
VAL_SONG_LIST        = ['track_8', 'sow', 'let_it_enfold_you', 'the_kill', 'misery_business', 'four_years', 'hair_of_the_dog', 'best_of_me', 'mookies_last_christmas', 'coffeeshop_soundtrack' ]     # the songs that will be not used in the training set but instead in the validation set
VAL_BATCH_SIZE       = 256
# Spleeter validation options
VAL_USE_DRUM_STEM           = TRAIN_USE_DRUM_STEM # if true, use the drum stem slices from the MAT_df to help with validating the model
VAL_INCLUDE_DRUM_STEM       = True      # if true, uses spleeter to separate out the drum stem and then append it as an additional channel
VAL_INCLUDE_MIXED_STEM      = False     # if true, uses spleeter to separate out the drum stem and then mix it with original mix to accentuate drums, then append as additional channel
VAL_MIXED_STEM_WEIGHTS      = TRAIN_MIXED_STEM_WEIGHTS # the weights multiplied by the full mix and drum mix respectively when added together
VAL_REPLACE_WITH_MIXED_STEM = False      # if true, replaces the normal full song channel with the mixed stem



''' pre-processing and formatting options '''
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
