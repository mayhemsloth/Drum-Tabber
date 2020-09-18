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
from src.configs import *


# defining the Dataset class
class Dataset(object):
    def __init__(self, dataset_type):   # dataset_type = 'train' or 'val'
        self.songs_path = SONGS_PATH    # train and validation subfolders will be contained in this folder file path form configs
        self.data_aug = TRAIN_DATA_AUG if dataset_type == 'train' else VAL_DATA_AUG   # boolean from configs



        self.songs= self.load_songs(dataset_type)

    # loads the raw song data
    def load_songs(self, dataset_type):

        return asdf
