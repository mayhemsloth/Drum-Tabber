#================================================================
#
#   File name   : utils.py
#   Author      : Thomas Hymel
#   Created date: 9-2-2020
#   GitHub      : https://github.com/mayhemsloth/Drum-Tabber
#   Description : Tools for pre-processing and post-processing of data
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


from src.configs import *


class MusicAlignedTab(object):
    """
    MusicAlignedTab is the class used to store the pre-processed object that will eventually become a part of the training set
    This class is mainly used to give the user a view into the alignments, augmentation effects, and other attributes that
    might need user verification before further processing is done.
    """
    def __init__(self, song_name):   # song_name is the string in the form of the folder names ('forever_at_last')
        self.filepaths = {'folder' : os.path.join(SONGS_PATH, song_name), 'json' : os.path.join(SONGS_PATH, song_name, song_name+'.json') } # filepaths dictionary
        # TODO: maybe add try except block here on importing of json file?
        self.json_dict = self.parse_json_file(self.filepaths['json'])              # json_dict has 'song', 'tab_file', 'tab_char_labels', and 'alignment_info' as keys
        self.filepaths['song'] = os.path.join(self.filepaths['folder'], self.json_dict['song'])
        self.filepaths['tab'] = os.path.join(self.filepaths['folder'], self.json_dict['tab_file'])
        self.hf_tab = self.import_tab()
        self.mf_tab = self.hf_to_mf(self.hf_tab)
        self.MAT = self.align_tab_with_music(self.mf_tab, self.json_dict['alignment_info'])

    def __str__(self):
        song_file = str(self.json_dict['song'])
        return 'This MAT is for ' + song_file

    # START OF MAIN HIGH LEVEL INIT FUNCTIONS
    def parse_json_file(self,json_fp):
        """
        Parses the JSON file information in the song folder to be utilized in the MusicAlignedTab

        Args:
            self [MusicAlignedTab]: can access the attributes of the class
            json_fp [str]: string of the filepath to the json file that describes this song tab's details

        Returns:
            dict: dictionary derived from the JSON file
        """
        with open(json_fp, 'r') as json_file:
            json_dct = json.load(json_file)
        return json_dct

    def import_tab(self):
        """
        Imports the .txt tab file in the song folder and returns the human-friendly form (hf),
        with the converted, master labels accoring to those stored in MASTER_FORMAT_DICT

        Args:
            self [MusicAlignedTab]: can access the attributes of the class

        Returns:
            list: list of lines (still with \n at the end) with correst master format labels, in the correct order
        """

        # create tab label conversion dictionary here
        master_format_dict = MASTER_FORMAT_DICT
        tab_char_labels = self.json_dict['tab_char_labels']   # getting the tab_char_labels dictionary for the object's json_dict
        tab_conversion_dict = {}      # dictionary whose keys are the CURRENT TAB's char labels and whose values are the MASTER FORMAT's char labels
        for key in tab_char_labels:
            if key in master_format_dict:   # ensures the keys are in both sets, which they should be
                tab_conversion_dict[tab_char_labels[key]] = master_format_dict[key]

        # read in the tab .txt file
        with open(self.filepaths['tab'], 'r', encoding='utf-8') as tab_file:
            tab_to_convert = tab_file.readlines()  # returns a python array of the text file with each line as a string, including the \n new line character at the end of each string

        # build this array tab up and return it
        tab = []
        for line in tab_to_convert:                 # loop over each element in tab (code line in tab)
            temp = line                             # store current line in temporary
            for key in tab_conversion_dict:          # At each line, loop over each key in the tab_conversion_dict
                if line.startswith(key):             # If a line starts with the key, reassign it to be the new label from dict...
                    temp = tab_conversion_dict[key] + line[len(key):]   # ...and concatenate with the rest of that line, assigning it to temp
            tab.append(temp)                         # temp is either the same as line, or it got changed because it startswith(key). Either way it needs to be appended to tab
        return tab

    def hf_to_mf(self, tab):
        """
        High level function that changes a tab in human-friendly format to machine-friendly formatting

        Args:
            self [MusicAlignedTab]: can access the attributes of the class
            tab [list]: the tab as read from the .txt file of the tab

        Returns:
            list: list of strings, each of which is garbage, note, Drum tab label, or time-keeping line. Represents the entire machine-friendly tab
        """

        # change the tab to include "white space" drum piece lines whenever there are tab lines WITHOUT a drum piece label
        expanded_tab = self.expand_tab(tab)

        # align all the drum piece lines in the entire tab with each other into one long line for every drum piece line
        mf_tab = self.combine_lines(expanded_tab)

        return mf_tab

    def align_tab_with_music(self, mf_output, alignment_info):
        """
        High level function that combines a machine friendly tab with a song into a pandas dataframe

        Args:
            mf_output [list]: list of strings, the machine friendly tab. Output of hf_to_mf function
            alignment_info [dict]: dictionary that contains useful alignment info for the current song. The format is seen below
                alignment_info = {'triplets' : bool,       # True = triplets exist somewhere in the tab. False = no triplets
                                  'tempo change' : bool,  # True = there is a tempo change. False = no tempo change
                                  'BPM' : int,            # Base tempo in beats per minute
                                  'first note onset' : float (seconds)
                                  # MAIN INFORMATION TO ALIGN TAB TO MUISC
                                  # This number in seconds to a specific decimal, is the onset of the first drum note listed in the
                                  # tab. This number, plus the BPM, will be used to mathematically calculate all the other
                                  # onsets of the other drum notes
                          }

        Returns:
            Dataframe: a music-aligned tab inside a pandas dataframe that contains the input data from a song sliced into the 16th note grid in one column,
                      and then all the other columns that represent the classification of the onsets of the drum pieces for that input data row
        """
        # Use the machine-friendly text output to get the array of strings into a dataframe, naming the table columns with the key or values of master format dict
        tab_df = self.tab_to_dataframe(mf_output)

        # Use the song object data to import the song into a Python object so it can be manipulated,
        # along with other information about the song gathered from the song file
        song, song_info = self.songify(self.filepaths['song'])

        # Use the tab dataframe and the song, along with other information, to align the tab with the music!
        df_MAT = self.combine_tab_and_song(tab_df, song, song_info, alignment_info)     # df_MAT = dataframe object of the Music Aligned Tab (with the music attached as well)

        return df_MAT
    # END OF MAIN HIGH LEVEL INIT FUNCTIONS

    # START OF HUMAN-FACING CLASS UTILITY FUNCTIONS
    def random_alignment_checker(self, drums_to_be_checked, num_buffer_slices):
        """
        Outputs a user-friendly way to check a random section of a song, to ensure that a music aligned tab is properly aligned

        Args:
            drums_to_be_checked [list]: a list of strings that correspond to the master format chars that are being requested to be checked. One random, non-blank result will be checked
            num_buffer_slices [int]: number of slices that is shown *after* the one that is randomly chosen for inspection

        Returns:
            None (outputs to display)
        """
        tk_label, measure_char, blank_char = TK_LABEL, MEASURE_CHAR, BLANK_CHAR # getter function
        MAT_df = self.MAT     # transferring the MAT_df info to MAT_df variable name so I don't have to change the function entirely

        drums_possible = list(set(drums_to_be_checked) & set(MAT_df.columns))  # getting the intersection so that no errors are thrown later

        drop_MAT = MAT_df.drop(columns = ['song slice', 'sample start'], errors = 'ignore')          # drop the slices column so we are left with only tab

        for drum in drums_possible:
            print("Sampling a " + str(drum) + " event for alignment check... Loading tab and audio slice")

            if (len(MAT_df[MAT_df[drum] != blank_char]) != 0):       # ensures there is a non-blank char in an existing column in the dataframe
                selection = MAT_df[MAT_df[drum] != blank_char].sample()   # first we create a mask to filter only for a drum event, then sample
                sel_index = selection.index[0]                            # grab index, which refers to the MAT_df index
                slices = []                                               # build up this array
                for idx in range(num_buffer_slices):                      # append the buffer slices after
                    if (sel_index + idx) < len(MAT_df.index):               # checks to make sure there are slices after
                        slices.append(list(MAT_df.at[sel_index+idx, 'song slice']))  # appends the next slices of the audio after the random selection

                # displaying of tab
                print(drop_MAT.iloc[int(sel_index):int(sel_index+num_buffer_slices), :].transpose()[::-1])  # print the tab corresponding to the audio in the correct orientation that we are used to seeing

                # builder and displaying of audio
                audio = []
                for item in slices:
                    for item2 in item:
                        audio.append(item2)
                audio_np = np.array(audio)
                ipd.display(ipd.Audio(audio_np.T, rate = 44100))
            else:
                print("No valid samples in " + str(drum) + " of that dataframe")

        return None

    @staticmethod
    def labels_summary(df):
        '''
        Outputs a summary of the labels currently

        Args:
            df [Dataframe]: Dataframe object that is a music aligned tab, or a FullSet tab, or some slice of a music aligned tab

        Returns:
            None (writes to console)
        '''
        blank_char = BLANK_CHAR

        # drop all columns whose column names if the column name contains a space
        to_drop = [col for col in list(df.columns) if ' ' in col]
        df_ = df.drop(columns = to_drop)

        # print statistics for each drum line
        if 'int64' not in list(df_.dtypes):
            print("---dataframe.describe() without blank_chars---")
            print(df_[df_ != blank_char].describe())
            print()

        print("---Unique values and frequencies by column name---")
        for col in df_:
            naf_series = df_[col].value_counts()
            print(str(naf_series.to_frame().T))
            print()

        return None
    # END OF HUMAN-FACING CLASS UTILITY FUNCTIONS

    # START OF DATAFRAME PROCESSING FUNCTIONS
    def tab_to_dataframe(self, mf_output):
        """
        Converts a machine-friendly tab to a dataframe object, using drum piece labels as the column names

        Args:
            mf_output [list]: list of strings defining the machine friendly format tab; output of the hf_to_mf function

        Returns:
            Dataframe: columns named from the master format dictionary, with time-keeping line in the first column, then note and garbage lines after
        """

        master_format_dict = MASTER_FORMAT_DICT # grabbing the master format dictionary
        tk_label, measure_char, blank_char = TK_LABEL, MEASURE_CHAR, BLANK_CHAR # grab the special chars
        tk_chars = tk_label + measure_char

        mf_reversed = list(reversed(mf_output))  # the reverse of this so that the tk line is on top, or in column 0 of the dataframe
        max_len = len(mf_reversed[0])            # max length of a line

        tab_dict = {}   # building up a tab dictionary to then put into a dataframe

        for idx, line in enumerate(mf_reversed):    # going through the machine-friendly reversed Python array for the last time
            line_chars = line[:len(tk_label)]       # grab the first chars used to identify the line
            line_as_chars = list(line[len(tk_label):])  # grabs the rest of the line past the two as a list of chars
            if idx == 0:               # time-keeping line
                tab_dict['tk'] = line_as_chars        # HARD CODED name here, and below for note and garbage lines
            elif line_chars in master_format_dict.values():  #  drum piece line case
                tab_dict[line_chars] = line_as_chars         # add it to the tab dictionary, to be made into a dataframe
            elif len(line_as_chars) == max_len - len(tk_label):   # check if we are in a note line
                tab_dict['note'] = line_as_chars
            else:            # in garbage line
                line_as_chars.extend(['']*(max_len-len(tk_label)-len(line_as_chars)))  # extending the garbage line with nulls to fit nicely into the dataframe
                tab_dict['garbage'] = line_as_chars

        tab_df = pd.DataFrame(tab_dict)  # making the dataframe from the constructed Python dictionary

        return tab_df

    def songify(self, song_title):
        """
        Utilizes the pydub package and AudioSegment class to extract Python song object given a song. Assumes a 3-char long audio extension

        Args:
            song_title [str]: string that refers to the filepath of the song file (mp3, wav, etc.)

        Returns:
            np.array: numpy array that is either (2 x n) where n is the number of samples and stereo song, or (n, 1) for mono song
            dict: a dictionary containing random bits of song information grabbed from the pydub AudioSegment class methods
        """
        song_format = os.path.splitext(song_title)[1][1:]  # grabs the string extension of the file
        song = AudioSegment.from_file(song_title, format = song_format)    # pydub AudioSegment class object of the song, utilizing the song_format

        # use get_song_info subfunction to get the dictionary of information about the song
        song_info = self.get_song_info(song, song_title)

        # Load the raw audio using librosa, with information gathered from pydub's AudioSegment class
        channel_mono = False                        # channel_mono = False when song is not mono. Default to stereo here
        if song_info["channels"] == "mono":         # if the song is mono, change the channel_mono boolean
            channel_mono = True
        elif song_info["channels"] == "stereo":     # if the song is stereo, do nothing
            channel_mono = False

        with warnings.catch_warnings():    # used to ignore the Pydub warning that always comes up
            warnings.simplefilter("ignore")
            lb_song, sr_song = lb.core.load(song_title, sr=None, mono=channel_mono) # uses librosa to output a np.ndarray of shape (n,) or (2,n) depending on the channels

        song_info['sr'] = sr_song  # add the sample rate, as loaded from librosa, into the song_info dict

        return lb_song, song_info

    def get_song_info(self, song, song_title):
        """
        Getter function that does menial grabbing from pydub AudioSegment class

        Args:
            song [AudioSegment]: a song as a pydub AudioSegment class
            song_title [str]:  string that refers to the song file (mp3, wav, etc.)

        Returns:
            dict: dictionary containing a bunch of random, perhaps useful information for the future song manipulation
        """
        song_info = {}   # dictionary to create
        base = os.path.basename(song_title)
        filename_only, extension_only = os.path.splitext(base)[0], os.path.splitext(base)[1][1:]
        song_info["title"] = filename_only
        song_info["format"] = extension_only

        # getting song's sample bit depth and writing to dictionary
        bytes_per_sample = song.sample_width
        if bytes_per_sample == 1:
            song_info["width"]= "8bit"
        elif bytes_per_sample == 2:
            song_info["width"]= "16bit"

        # getting song's number of channels and writing to dictionary
        num_channels = song.channels
        if num_channels == 1:
            song_info["channels"] = "mono"
        if num_channels == 2:
            song_info["channels"] = "stereo"

        song_info["frame rate"] = song.frame_rate             # almost certainly 44100 (in Hz), because all songs are that
        song_info["duration_seconds"] = song.duration_seconds # the song's duration in seconds
        song_info["duration_minutes"] = song.duration_seconds / 60 # the song's duration in minutes

        return song_info

    def combine_tab_and_song(self, tab_df, song, song_info, alignment_info):
        """
        Combines the tab dataframe and the song into one dataframe object, aligned properly by utilizing the provided alignment information

        Args:
            tab_df [Dataframe]: still including the note and garbage line and the measure char rows of data
            song [np.array]: raw data of the song from librosa, either in 1D for mono or 2D [L,R] for stereo
            song_info [dict]: dictionary that may come in handy for passing song information around
            alignment_info [dict]: dictionary supplied by the coder/user that contains the following keys/values
                        alignment_info = {'triplets' : bool,       # True = triplets exist somewhere in the tab. False = no triplets
                                           'tempo change' : bool,  # True = there is a tempo change. False = no tempo change
                                            'BPM' : int,            # Base tempo in beats per minute, constant throughout song for 'tempo change' = False
                                           'first drum note onset' : float (seconds)
                                  # MAIN INFORMATION TO ALIGN TAB TO MUISC
                                  # This number in seconds to a specific decimal, is the starting of the 16th note grid wherein the first drum note listed in the
                                  # tab occurs. This number, plus the BPM, will be used to mathematically calculate all the other
                                  # beginnings of the 16th note grids
                                          }

        Returns:
            Dataframe: contains 'song slice', 'tk', 'sample start' and drum piece label columns, aligned properly. Basically the sum of 'song slice' should be most of the song if converted back to audio and played
        """
        tk_label, measure_char, blank_char = TK_LABEL, MEASURE_CHAR, BLANK_CHAR # grab the special chars, we'll need the blank_char and measure_char

        # TODO: Allow Tempo Changes into the set of tabs/music that can be processed
        if alignment_info['tempo change'] == True: # checks if the tab has an tempo changes
            print("This song has a tempo change, rejecting the tab for now.")
            return tab_df    # If so, return the tab dataframe unchanged because we don't want to deal with that right now

        triplets_bool = alignment_info['triplets']  # grabs the boolean of if this tab has ANY triplets in it

        sample_rate = song_info['frame rate']         # almost certainly 44100, but generalized if using some stupidly weird song/recording
        song_length = song_info['duration_seconds']   # length of song in seconds
        BPM = alignment_info['BPM']                   # extract the BPM of the song (int)
        fdn = alignment_info['first drum note onset'] # extract the seconds location of the beginning of the measure that contains the first drum note in the tab

        song = song.T                             # to utilize the code that I had written already to work with pydub, I need to transpose the librosa output here to be consistent
        sample_num = len(song)                    # the total number of samples in the song array, whether it's a 2D array (stereo) or not doesn't matter
        fdn_sample_loc = int(fdn*sample_rate)      # the starting sample number of the first drum note
        sample_delta_out, remainder_delta = divmod(15*sample_rate,BPM)  # the amount of sample change for each 16th note grid, and the remainder, used to calculate
        sample_delta = int(sample_delta_out)        # ensures that the quotient is an integer, regardless of the BPM given: needed for later index slicing
        decimal_delta = float(remainder_delta/BPM)  # the decimal part of the sample_delta that was dropped. Useful for rectifying proper slice length later

        # the main goal is to slice up song array with the correct length slices and attach them to the correct places in tab_df,
        # where we are given an anchor point with first drum note (fdn) in seconds

        # clean up the tab_df object
        tab_df = tab_df.drop(['note','garbage'], axis=1)    # remove the note and garbage columns of the tab dataframe
        tab_df = tab_df[~tab_df['tk'].isin([measure_char])] # removes all the rows of data that contain measure_char by first creating a boolean mask with .isin based on the time-keeping column, and then inverting it with ~
        tab_df.reset_index(drop=True, inplace=True)         # reset the index after we removed all the rows that contain measure_char and drops the old index column

        # find df_index of row with first drum note
        fdn_row_index = tab_df[tab_df.drop(['tk'], axis=1) != blank_char].first_valid_index()  # creates a mask that changes all the blank_char entries to NaN, then grabs the first row which has a non NaN value (after temporarily dropping the tk line, which could affect this mask)
        # print("All the following prints are from combine_tab_and_song function:")
        # print(f'first drum note row = {fdn_row_index}')
        tab_len = len(tab_df.index)     # total length of the drum tab dataframe

        # slice up raw audio AFTER the first drum note, correcting for potential misalignment due to lopping off remainder of sample delta, and handling the triplet case
        song_slices_post_fdn = self.get_post_fdn_slice(song, fdn_sample_loc, sample_num, sample_delta, decimal_delta, triplets_bool, tab_df, fdn_row_index)

        # slice up raw audio BEFORE the first drum note, correcting for potential misalignment due to lopping off remainder of sample delta
        song_slices_pre_fdn = self.get_pre_fdn_slice(song, fdn_sample_loc, sample_num, sample_delta, decimal_delta)

        """PRINTING USEFUL OUTPUT FOR SANITY CHECK"""
        # print('# of song slices post fdn = ' + str(len(song_slices_post_fdn)))
        # print('# of song slices pre fdn = ' + str(len(song_slices_pre_fdn)))
        # print("Produced number of song slices = " + str(len(song_slices_pre_fdn) + len(song_slices_post_fdn)))
        # print("Expected number of song slices (should be same for non-triplet songs) = " + str(sample_num/(sample_delta+decimal_delta)))

        # take the two song slices, and the location of the first drum note index, and produce a dataframe of the same length as the tab frame,
        # with the song slices in the correct index position
        song_slices_df = self.slices_into_df(song_slices_pre_fdn, song_slices_post_fdn, fdn_row_index, tab_len, fdn_sample_loc)

        df_MAT = tab_df.merge(song_slices_df, how = 'left', left_index=True, right_index = True)     # merge the tab frame with the song slice frame!

        return df_MAT

    def get_post_fdn_slice(self, song, fdn_sample_loc, sample_num, sample_delta, decimal_delta, triplets_bool, tab_df, fdn_row_index):
        """
        Helper subfunction in combine_tab_and_song that outputs song slices post the first drum note

        Args:
            song [np.array]: raw audio data of the song
            fdn_sample_loc [int]: an int describing the sample of the first
            sample_num [int]: total number of samples in the song array
            sample_delta [int]: the amount of sample change for each 16th note grid in tab for this song
            decimal_delta [float]: the remainder of the sample_delta. Useful for properly counting the samples into 'song slice'
            triplets_bool [bool]: boolean that denotes the presence of triplets in the tab or not
            tab_df [Dataframe]: dataframe object of the tab in its current form
            fdn_row_index [int]: the index of the row of the first drum note in the tab_df

        Returns:
            list: a list of song slices, where each are an np.array of samples (of roughly the same length, depending on the BPM and presence of triplets)
        """

        quarter_chars, eighth_chars, sixteenth_chars = QUARTER_TRIPLET, EIGHTH_TRIPLET, SIXTEENTH_TRIPLET  # grabs the triplet characters

        song_slices_post = []     # appending this array to build up the song slices
        decimal_counter = 0           # counter needed for rectifying slice length
        postfdn_idx = fdn_sample_loc  # sets the index counter to start at the first drum note sample location
        row_index_counter = fdn_row_index  # sets the row counter to start at the first drum note row location

        if triplets_bool == True:      # Completely split the cases of triplets or not
            while postfdn_idx < (sample_num - sample_delta):    # ensures no indexing bounds error
                # if the next tk column value is char t (the triplet signal), we are at the beginning of a triplet of some kind (Also do a check to ensure indexing error doesn't occur)
                if ( row_index_counter < len(tab_df.index)-1) and (tab_df.at[row_index_counter+1, 'tk'] == quarter_chars[0]):
                    if tab_df.at[row_index_counter+1, 'tk'] + tab_df.at[row_index_counter+2, 'tk'] == quarter_chars: # quarter note trips case
                        new_total_delta = (sample_delta+decimal_delta) * (8/3)    # changing the sample and decimal deltas
                    elif tab_df.at[row_index_counter+1, 'tk'] + tab_df.at[row_index_counter+2, 'tk'] == eighth_chars: # eighth note trips case
                        new_total_delta = (sample_delta+decimal_delta) * (4/3)    # changing the sample and decimal deltas
                    elif tab_df.at[row_index_counter+1, 'tk'] + tab_df.at[row_index_counter+2, 'tk'] == sixteenth_chars: # sixteenth note trips case
                        new_total_delta = (sample_delta+decimal_delta) * (2/3)     # changing the sample and decimal deltas
                    sample_delta_trip, decimal_delta_trip = (int(new_total_delta // 1), new_total_delta % 1) # change the sample_delta and decimal_delta to new values
                    for index in range(3):  # do the following code three times. Copy of the other code, but using the new properly scaled triplet sample and decimal deltas
                        if decimal_counter <= 1:
                            decimal_counter += decimal_delta_trip
                            song_slices_post.append(song[postfdn_idx:postfdn_idx+sample_delta_trip])
                            postfdn_idx += sample_delta_trip
                        else:
                            decimal_counter = decimal_counter-1
                            song_slices_post.append(song[postfdn_idx:postfdn_idx+sample_delta_trip+1])
                            postfdn_idx += sample_delta_trip + 1
                        row_index_counter += 1

                # non-triplet tk char case, but in a triplet-containing tab
                else:     # we are in a non-triplet case, so we use the previously working code, but remember to increment our row_index_counter
                    if decimal_counter <= 1:                        # we are in a case where we don't have to account for the decimal
                        decimal_counter += decimal_delta            # add the decimal delta to the decimal counter
                        song_slices_post.append(song[postfdn_idx : postfdn_idx+sample_delta])   # grab the correct slice
                        postfdn_idx += sample_delta                # increment the index counter
                    else:    # in the case of decimal_counter being over 1
                        decimal_counter = decimal_counter-1        # removing the 1
                        song_slices_post.append(song[postfdn_idx : postfdn_idx+sample_delta+1])   # grab the correct slice, with the 1
                        postfdn_idx += sample_delta+1               # increment the index counter
                    row_index_counter += 1                     # increment the row index counter (used for triplets cases)


        else:    # We are not in the triplet cases, so this code works for non-triplet case
            while postfdn_idx < (sample_num - sample_delta):    # ensures no indexing bounds error
                if decimal_counter <= 1:                        # we are in a case where we don't have to account for the decimal
                    decimal_counter += decimal_delta            # add the decimal delta to the decimal counter
                    song_slices_post.append(song[postfdn_idx : postfdn_idx+sample_delta])   # grab the correct slice
                    postfdn_idx += sample_delta                # increment the index counter
                else:    # in the case of decimal_counter being over 1
                    decimal_counter = decimal_counter-1        # removing the 1
                    song_slices_post.append(song[postfdn_idx : postfdn_idx+sample_delta+1])   # grab the correct slice, with the 1
                    postfdn_idx += sample_delta+1                # increment the index counter

        return song_slices_post

    def get_pre_fdn_slice(self, song, fdn_sample_loc, sample_num, sample_delta, decimal_delta):
        """
        Helper subfunction in combine_tab_and_song that outputs song slices PRE the first drum note. Takes fewer args than post because we don't actually really care about the music/tab before the first drum note

        Args:
            song [np.array]: raw audio data of the song
            fdn_sample_loc [int]: an int describing the sample of the first
            sample_num [int]: total number of samples in the song array
            sample_delta [int]: the amount of sample change for each 16th note grid in tab for this song
            decimal_delta [float]: the remainder of the sample_delta. Useful for properly counting the samples into 'song slice'

        Returns:
            list: a list of song slices, where each are an np.array of samples (of roughly the same length, depending on the BPM and presence of triplets)
        """
        song_slices_pre = []     # appending this array to build up the song slices
        decimal_counter = 0           # counter needed for rectifying slice length
        prefdn_idx = fdn_sample_loc  # sets the index counter to start at the first drum note sample location
        while prefdn_idx > (sample_delta+1):    # ensures no indexing bounds error
            if decimal_counter <= 1:                        # we are in a case where we don't have to account for the decimal
                decimal_counter += decimal_delta            # add the decimal delta to the decimal counter
                song_slices_pre.append(song[prefdn_idx-sample_delta : prefdn_idx])   # grab the correct slice
                prefdn_idx -= sample_delta                # decrement the index counter
            else:                                    # in the case of decimal_counter being over 1
                decimal_counter = decimal_counter-1        # removing the 1
                song_slices_pre.append(song[prefdn_idx-(sample_delta+1) : prefdn_idx])   # grab the correct slice, with the 1
                prefdn_idx -= (sample_delta+1)            # decrement the index counter

        song_slices_pre.reverse()      # since we appended the list with an index counting backwards, reverse the list to get proper order

        return song_slices_pre

    def slices_into_df(self, slices_pre_fdn, slices_post_fdn, fdn_row_index, tab_len, fdn_sample_loc):
        """
        Helper subfunction in combine_tab_and_song that pushes the song slices into its own df.

        Args:
            slices_pre_fdn [list]: list of np.arrays that are the raw audio of the song slices pre first drum note
            slices_post_fdn [list]: list of np.arrays that are the raw audio of the song slices post first drum note
            fdn_row_index [int]: index of the row of the first drum note in the tab dataframe
            tab_len [int]: length of the tab dataframe
            fdn_sample_loc [int]:

        Returns:
            Dataframe: dataframe of two columns: one named 'song slice' that contains all the rows of the song slices, in the correct index position, to afterwards be immediately adjoined with the tab
                        The other column named 'sample start' which is the beginning sample index number of each song slice, as measured by the initial entire song file
        """

        partial_slices_post_fdn = slices_post_fdn[0: tab_len - fdn_row_index] # grabs the first tab_len - fdn_row_index of slices post fdn
        partial_slices_pre_fdn = slices_pre_fdn[len(slices_pre_fdn) - fdn_row_index :]  # grabs the last fdn_row_index number of slices from slices_pre_fdn
        song_slices_tab_indexed = partial_slices_pre_fdn + partial_slices_post_fdn   # concatenates the two arrays

        # Getting the sample index corresponding to the first sample of each song slice, with the same
        sample_start_post = []
        counter = 0
        for slice1 in partial_slices_post_fdn:
            sample_start_post.append(counter + fdn_sample_loc)    # the current slice's sample start
            counter += len(slice1)            # the counter increases by the length of the slice

        sample_start_pre = []
        counter = 0
        if len(partial_slices_pre_fdn) != 0:     # ensures the partial_slices_pre_fdn is a non-empty set
            for slice2 in reversed(partial_slices_pre_fdn):    # the partial_slices_pre_fdn slices are in the correct time order, but we want to work backwards
                counter = counter - len(slice2)    # counter decrements by the length of the current slice before appending the number
                sample_start_pre.append(counter + fdn_sample_loc)

        if len(sample_start_pre) != 0:
            sample_start_pre.reverse()   # appended in the backwards direction, so we need to reverse the list now

        sample_start_list = sample_start_pre + sample_start_post   # concatenate the two lists


        """PRINTING USEFUL OUTPUT FOR SANITY CHECK START"""
        # print("tab length = " + str(tab_len) + "     datatype: " + str(type(tab_len)))
        # print("len(song_slices_tab_indexed) = " + str(len(song_slices_tab_indexed)) + "     datatype of object: " + str(type(song_slices_tab_indexed)))
        # print("song_slices_tab_indexed[0].shape = " + str(song_slices_tab_indexed[0].shape) + "     datatype of [0]: " + str(type(song_slices_tab_indexed[0])))
        # print("np.array(song_slices_tab_indexed).shape = " + str(np.array(song_slices_tab_indexed, dtype='object').shape))
        # print(f'len(sample_start_list) = {len(sample_start_list)}')
        """PRINTING USEFUL OUTPUT FOR SANITY CHECK END"""


        # Force the current "list" into a dictionary so that pandas ALWAYS INTERPRETS IT AS A LIST OF ARRAYS, not just one single large 3D matrix
        tab_dict = {'song slice': song_slices_tab_indexed, 'sample start' : sample_start_list} # HARD CODED name
        song_slices_df = pd.DataFrame(data=tab_dict)

        return song_slices_df
    # END OF DATAFRAME PROCESSING FUNCTIONS


    # START OF TAB STRING PROCESSING FUNCTIONS
    def expand_tab(self, tab):
        """
        Creates and inputs the blank drum lines in each human-readable music line that are necessary for proper future alignment

        Args:
            tab [list]: list array of strings for each \n format

        Returns:
            array: the expanded tab, in the list array of strings for each \n format
        """
        # Assign the previously used variable names to the constant configs variable names
        master_format_dict = MASTER_FORMAT_DICT
        tk_label, measure_char, blank_char = TK_LABEL, MEASURE_CHAR, BLANK_CHAR

        # tk_label is the time-keeping label is two spaces, used to denote the beginning of a time-keeping line
        # measure_char is the character used to split the measures and otherwise make the tabulature human readable
        # blank_char is the character used to denote a blank in a drum piece line

        tk_chars = tk_label + measure_char   # the time-keeping line starts with two chars (here, two spaces) and then the measure_char

        # grab a list of all drum pieces used at least once in the tab
        drum_pieces_used = []                             # empty array to be appended
        for line in tab:                                  # loop over every line in tab
            for value in master_format_dict.values():     # loop over all the values in master_format_dict, which are the 2 char labels
                if line.startswith(value + measure_char) and value not in drum_pieces_used:
                    drum_pieces_used.append(value)      # line must start with the 2 char label+measure_char AND NOT already be in the list

        # change the structure of the tab object so one can more easily expand tab later
        split_on_tk_reversed = self.reverse_then_split_on_tk(tab)

        # now we have an array of strings, where each string either begins with tk_chars, or is the first element in the array
        # For each element that starts with tk_chars, we check which drum piece labels exist in that element, and
        # add in the appropriate length empty line for each missing drum piece, and then combine
        # This is handled by the subfunction add_empty_lines
        expanded_split_on_tk_reversed = self.add_empty_lines(split_on_tk_reversed, drum_pieces_used)

        # To make the future appending function easier, we want to sort the drum piece labels so that each tk line has the same ordered drum lines after.
        ordered_ex_split_on_tk_reversed = self.order_tk_lines(expanded_split_on_tk_reversed, drum_pieces_used)

        # now we want to undo the structural changes made to the tab
        combined_ordered_ex_reversed = ''.join(ordered_ex_split_on_tk_reversed)          # combine the tk split elements into one string
        ordered_expanded_reversed_tab = combined_ordered_ex_reversed.splitlines(keepends=True) # separate into different individual lines, keeping the newline character

        expanded_tab = list(reversed(ordered_expanded_reversed_tab))    # reverse the tab entirely

        return expanded_tab

    def reverse_then_split_on_tk(self, tab):
        """
        Reverses a tab and then splits it on the time-keeping lines to create time-keeping block sections of the tab

        Args:
            tab [list]: list array of strings for each \n format

        Returns:
            array: a tab that is reversed and then split along the time-keeping lines, creating time-keeping block sections
        """

        tk_label, measure_char, blank_char = TK_LABEL, MEASURE_CHAR, BLANK_CHAR
        tk_chars = tk_label + measure_char   # the time-keeping line starts with two chars (here, two spaces) and then the measure_char

        reversed_tab = list(reversed(tab))                       # make a reverse tab object so we can execute the next code more easily with time-keeping lines coming before the drum piece lines
        combined_reversed_tab = ''.join(reversed_tab)              #combine it into one string so we can split on the time-keeping chars
        split_on_tk_reversed = combined_reversed_tab.split(tk_chars) # split into parts based on the tk chars
        for idx in range(1, len(split_on_tk_reversed)):             # for loop to add the delimiter tk chars back into the proper string elements
            split_on_tk_reversed[idx] =  tk_chars + split_on_tk_reversed[idx] # append the tk chars at the beginning of the proper elements

        return split_on_tk_reversed

    def add_empty_lines(self, split_on_tk_reversed, drum_pieces_used):
        """
        Adds the correct length empty lines to ensure time-keeping sections have the full amount of drum piece label lines in a tab

        Args:
            split_on_tk_reversed [list]: list of strings, the tab that has in the reverse tk sections
            drum_pieces_used [list]: list of chars, specifically the 2 char drum piece labels of the drums used in this tab

        Returns:
            list: an array of strings, most of which start on the tk chars, that contain a "full" tk section of drum events and blanks
        """

        tk_label, measure_char, blank_char = TK_LABEL, MEASURE_CHAR, BLANK_CHAR
        tk_chars = tk_label + measure_char     # useful variable

        expanded_split_on_tk_reversed = []     # build up this array to be returned

        for string in split_on_tk_reversed:    # loop through all elements of the array
            if not string.startswith(tk_chars):    # if element does not start with tk_chars, we don't want to change it
                expanded_split_on_tk_reversed.append(string)  # immediately append it, and then move onto next string
            else:                                  # in the case where element starts with tk_chars
                drums_in_tk = []                   # build up the drums already found in this tk element
                for drum in drum_pieces_used:      # loop over all drums found in the drum tabs
                    if (drum + measure_char) in string:  # checks if the string of drums+measure_char is in the tk element
                        drums_in_tk.append(drum)         # if so, add it to the list of drums in the tk
                drums_not_in_tk = list(x for x in drum_pieces_used if x not in drums_in_tk) # grabs all the drum pieces not in tk element but used in overall drum tab

                tk_line = string[len(tk_label) : string.find('\n')]   # grabs the entirety of the time-keeping line, without the \n character at the end of it
                blank_line = ''                      # build up a blank line string equal in length to tk_line (without the initial tk label or the \n char)
                for char in tk_line:                 # loop through chars in current tk line
                    if char is measure_char:         # if any of the chars are the measure char, keep it there
                        blank_line = blank_line + measure_char
                    else:                            # if any of the chars are anything but measure chars, "replace" it with blank char
                        blank_line = blank_line + blank_char

                blank_drums = ''                     # the entire set of drum piece lines in one string
                for drum in drums_not_in_tk:         # utilize the correct length blank line to add the correct blank lines for each drum
                    blank_drum_line = ''.join((drum, blank_line, '\n'))  # construct the blank drum piece line, adding the new line character
                    blank_drums = blank_drums + blank_drum_line          # add this blank drum piece line to running blank drums string
                expanded_string = string[:string.find('\n')+1] + blank_drums + string[string.find('\n')+1:] # use str slicing to make a new expanded string, placing the new blank drum lines immediately after the tk line
                expanded_split_on_tk_reversed.append(expanded_string)          # append the expanded tk string to the running array.

        return expanded_split_on_tk_reversed

    def order_tk_lines(self, expanded_split_on_tk_reversed, drum_pieces_used):
        """
        Orders the drum piece lines in a time-keeping section with respect to the desired order

        Args:
            expanded_split_on_tk_reversed [list]: array of strings that is the tab in its current processing step
            drum_pieces_used [list]: array of chars, specifically the 2 char drum piece labels of the drums used in this tab

        Returns:
            list: the tab in a list of strings, where the drum piece lines are in the same order with respect to each other in different tk sections
        """

        tk_label, measure_char, blank_char = TK_LABEL, MEASURE_CHAR, BLANK_CHAR
        tk_chars = tk_label + measure_char     # useful variable

        # If you don't want to use alphabetical order, but some other order for your drums, the function get_desired_order
        # will return a dictionary that contains the order key based on master format labels and the coder's (my) liking
        desired_order_dict = DESIRED_ORDER_DICT
        drum_pieces_used.sort(key=desired_order_dict.get)    # sorting drum pieces used based on the desired order dictionary

        ordered_tk_lines = []     # build up this array to be returned

        for string in expanded_split_on_tk_reversed:    # loop through all elements of the array
            if not string.startswith(tk_chars):         # if element does not start with tk_chars, we don't want to change it
                ordered_tk_lines.append(string)         # immediately append it, and then move onto next string
            else:                                       # in the case where element starts with tk_chars
                exploded_tk_string = string.splitlines(keepends=True) # splits the tk elements on the newline character, and keeps it
                dct = {x+measure_char: i for i, x in enumerate(drum_pieces_used)}  # setting up a key dictionary to be used later to compare and sort ("BD|" = 0, "CC|" = 1, etc.)
                drums_in_etk_string = [x for x in exploded_tk_string if x[:len(tk_chars)] in dct]  # grabs only the lines in exploded tk string that contains drum piece labels
                drums_in_etk_string.sort(key = lambda x: dct.get(x[:len(tk_chars)]))               # sort the drum line strings by the
                iterator = iter(drums_in_etk_string)              #create the iterator for the next line
                ordered_exploded_tk_string = [next(iterator) if x[:len(tk_chars)] in dct else x for x in exploded_tk_string]  # orders the drum piece lines without affecting any other line
                ordered_tk_string = ''.join(ordered_exploded_tk_string)                       # connects the exploded strings back together
                ordered_tk_lines.append(ordered_tk_string)                            # appends the new ordered string to the running array

        return ordered_tk_lines

    def combine_lines(self, expanded_tab):
        """
        Combines a tab from its expanded form to its entirely-on-one-line form (the machine-friendly tab)

        Args:
            expanded_tab [list]: array of strings, the tab in its expanded and ordered normal form

        Returns:
            list: list of strings, where each element is an entire drum piece label, or the note, garbage, or time-keeping line. That is, the machine friendly form
        """

        # Prelim- Get the special characters because we'll need one of them
        master_format_dict = MASTER_FORMAT_DICT
        tk_label, measure_char, blank_char = TK_LABEL, MEASURE_CHAR, BLANK_CHAR
        tk_chars = tk_label + measure_char

        # grab the drum pieces used in the piece, and then sort them by the desired order.
        # This could probably be more efficiently done considering every music line now contains all of drum pieces used, but whatever
        drum_pieces_used = []                             # empty array to be appended
        for line in expanded_tab:                                  # loop over every line in tab
            for value in master_format_dict.values():     # loop over all the values in master_format_dict, which are the 2 char labels
                if line.startswith(value + measure_char) and value not in drum_pieces_used:
                    drum_pieces_used.append(value)      # line must start with the 2 char label+measure_char AND NOT already be in the list
        desired_order_dict = DESIRED_ORDER_DICT
        drum_pieces_used.sort(key=desired_order_dict.get)    # sorting drum pieces used based on the desired order dictionary

        # First we'll get rid of any duplicate, consecutive newline's that are ultimately not needed, to simplify processing later
        exp_tab = [x for i,x in enumerate(expanded_tab) if i ==0 or x != '\n' or x != expanded_tab[i-1]]

        # Now we'll make the assumption that there's only one line of useful text above any music line, if any, and later attempt to
        # append it to the correct spot above that measure in the eventual single long music line.

        # reverse and split on the time-keeping chars so we have an array of strings, almost all of which start with tk_chars
        exp_sotk_rev_tab = self.reverse_then_split_on_tk(exp_tab)

        num_drums = len(drum_pieces_used)      # total number of DRUM piece lines that are on attached underneath each tk line

        mf_tab = []         # final array to build up. Should have 1 tk line, num_drums num of lines, and then 1 notes on drums line, and then 1 garbage text line
        extra_lines = 3     # HARD CODED number of extra lines
        total_lines = num_drums + extra_lines

        for idx in range(total_lines):   # as described above: 1tk line, then drum lines, then note line, then garbage line.
            mf_tab.append('')

        for string in exp_sotk_rev_tab:         # loop over all the tk elements
            if string.startswith(tk_chars):           # if the string starts with tk_chars, then it contains all the drum piece lines "underneath" it, already in order
                exploded_tk_string = string.splitlines(keepends=False) # splits the tk elements on the newline character, loses the newline because we don't need it anymore
                total_chars_in_tk_line = len(exploded_tk_string[0])    # grab this for later use
                for line_num, line in enumerate(exploded_tk_string):   # grabs the index and line of every line in a tk element
                    if line_num < total_lines-2:        # accessing a tk line or drum piece line
                        mf_tab[line_num] = line[len(tk_chars):] + mf_tab[line_num]   # get a slice of after the first 3 chars, and append it to the beginning, because we are going backwards
                    elif line_num == total_lines-2:     # accessing a note line
                        if (total_chars_in_tk_line - len(line) - len(tk_chars)) >= 0:   # If note line extends nearly the full length, the slicing is slightly different
                            mf_tab[line_num] = line + (" " * (total_chars_in_tk_line - len(line) - len(tk_chars))) + mf_tab[line_num]    # fills in the note line with whitespace so that the other note line notes will align properly
                        else:
                            mf_tab[line_num] = line[len(tk_chars):] + (" " * (total_chars_in_tk_line - len(line))) + mf_tab[line_num]
                    else:                               # in a case where the tk element had more drum_piece + extra lines than expected, so the rest can be added to garbage line
                        mf_tab[-1] = line + " " + mf_tab[-1]
            else:                                    # the case where our tk element did not start with tk_chars
                mf_tab[-1] = string + " " + mf_tab[-1]    # it gets added to the garbage line

        # Now we add back in the important line labels at the beginning of each long line
        for idx in range(len(mf_tab)):  # loop through each long, combined line in mf_tab
            if idx == 0:
                mf_tab[idx] = tk_chars + mf_tab[idx]    # start the tk line with the tk_chars
            elif idx < 1+num_drums:                 # will enter this a total number of drums times
                mf_tab[idx] = drum_pieces_used[idx-1] + measure_char + mf_tab[idx]  # start the drum line with the drum pieces used 2 chars + measure_char
            elif idx == (num_drums+1):   # we are on note line
                mf_tab[idx] = " "*len(tk_chars) + mf_tab[idx]  # adds spaces*tk_chars to realign all the notes to their proper places
                if len(mf_tab[idx]) > len(mf_tab[0]):   # somehow the note line is longer, so we shall cut it
                    slice_me = mf_tab[idx]
                    mf_tab[idx] = slice_me[0:len(mf_tab[0])]
            else:    # we are in the garbage line
                mf_tab[idx] = " "*len(tk_chars) + mf_tab[idx]  # to be consistent with everything else

        mf_tab_final = list(reversed(mf_tab))   # back into the "human friendly" version of the mostly machine friendly text: Garbage line, note line, and then desired order until tk line

        return mf_tab_final
    # END OF TAB STRING PROCESSING FUNCTIONS

# START OF MAT_df CREATION, CLEANING, ENCODING
def create_FullSet_df(songs_file_path):
    '''
    Function used to create a FullSet_df to inspect summary statistics on an entire initial dataset of dfs, or to randomly look at different songs

    Args:
        songs_file_path [str]: Filepath to the songs folder

    Returns:
        Dataframe: the FullSet dataframe of all songs' dataframes stacked on top of each other (and outputs information to the user display)
    '''

    # need to create a MusicAlignedTab object on every song in the song_folder_path given to us, then extract Dataframe, then concatenate
    MATDF_dict = {}  # dictionary where the keys are the string of the name of the song, and the values are the dataframe from the MAT class object

    subdirs = [os.path.join(songs_file_path, o) for o in os.listdir(songs_file_path) if os.path.isdir(os.path.join(songs_file_path,o))] # grab all the subdirectories in the song_folder_path
    list_of_songs = [os.path.basename(os.path.normpath(x)) for x in subdirs]  # grabbing only the end of the song folders (that is, the song title)
    # print(f'subdirs = {subdirs}')
    # print(f'list_of_songs = {list_of_songs}')

    for song in list_of_songs:                # go through all the songs folders
        MAT_class = MusicAlignedTab(song)     # create the MAT class object
        MATDF_dict[song] = MAT_class.MAT      # extract and keep only the dataframe of the class object

    # get blank char to use in this function
    blank_char = BLANK_CHAR

    # stacks all the dataframes one on top of another in the rows, ignoring indices of each dataframe, and giving each frame an extra "key" layer
    print("...Concatenating all music-aligned dataframes")
    output_df = pd.concat(MATDF_dict, axis=0, ignore_index = False, join = 'outer', sort = False)

    print("...Replacing NaNs with " + blank_char + " for output")
    output_df = output_df.fillna(blank_char)               # replace NaN with the blank_char, for columns that didn't exist in one tab but existed in others

    MusicAlignedTab.labels_summary(output_df)

    return output_df

def one_hot_encode(df):
    """
    Encoder for the class labels of a tab dataframe. Encoded as a multi-label one hot vector: that is, a one hot vector that can be 1 in multiple classes for each example
    Note that the labels from the time-keeping line are classifed as "beats" and "downbeats", not "offbeats" and "downbeats"
    Note that the labels names are "encoded" (col + '_' + label) into the column names of the dataframe, regardless of how many labels you might have in each column originally

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

def create_configs_dict(df):
    '''
    Creates a dictionary  that contains a dictionary of the index to the class labels, among other important config choices/values

    Args:
        df [Dataframe]: encoded FullSet_df that would contain all the information of the class names

    Returns:
        dict: the configs dictionary that has a bunch of useful information in it concerning the current model's configs
    '''

    class_names = [x for x in list(df.columns) if '_' in x]
    num_features = 2*N_MELS if INCLUDE_FO_DIFFERENTIAL else N_MELS
    num_channels = 3 if INCLUDE_LR_CHANNELS else 1
    month_date = date.today().strftime("-%b-%d")

    configs_dict = {'class_names_dict': {idx: val for idx, val in enumerate(class_names)},
                    'num_classes'   : len(class_names),
                    'num_features'  : num_features,
                    'num_channels'  : num_channels,
                    'n_mels'        : N_MELS,
                    'model_type'    : MODEL_TYPE,
                    'window_size'   : WINDOW_SIZE,
                    'fmax'          : FMAX,
                    'hop_size'      : HOP_SIZE,
                    'shift_to_db'   : SHIFT_TO_DB,
                    'n_context_pre' : N_CONTEXT_PRE,
                    'n_context_post': N_CONTEXT_POST,
                    'include_fo_differential'  : INCLUDE_FO_DIFFERENTIAL,
                    'positive_window_fraction' : POSITIVE_WINDOW_FRACTION,
                    'negative_window_fraction' : NEGATIVE_WINDOW_FRACTION,
                    'tolerance_window' : TOLERANCE_WINDOW,
                    'classification_dict' : {'clean_date' : CLEAN_DATA, 'keep_dynamics': KEEP_DYNAMICS, 'keep_bells': KEEP_BELLS,
                                             'keep_toms_seperate' : KEEP_TOMS_SEPARATE, 'hihat_classes' : HIHAT_CLASSES, 'cymbal_classes' : CYMBAL_CLASSES},
                    'month_date' : month_date
                    }

    return configs_dict
# END OF MAT_df CREATION, CLEANING, ENCODING

# START OF HUMAN-FACING UTILITY FUNCTIONS
# END OF HUMAN-FACING UTILITY FUNCTIONS

# START OF MODEL SAVING, LOADING, AND INFERENCE FUNCTIONS
def save_drum_tabber_model(drum_tabber, model_name, saved_models_path, configs_dict):
    '''
    Saves a keras.Model drum-tabber to the correct location, along with the configs dictionary used to build that model

    Args:
        drum_tabber [keras.Model]: the trained model for automatically tabbing drums in songs
        model_name [str]: custom name given to the model to be saved. Determines the subfolder name
        saved_models_path [str]: filepath to the folder where the models are saved
        configs_dict [dict]: dictionary that has compiled configuration options

    Returns:
        None (writes to disk)
    '''

    model_name_folder_path = os.path.join(saved_models_path, model_name)

    if not os.path.exists(model_name_folder_path):
        os.mkdir(model_name_folder_path)   # makes the model name folder

    # saves the model in the new folder for that model
    drum_tabber.save(filepath = model_name_folder_path)

    # saves the configs_dict in the same folder
    # TODO: check that you can load a model if something else has been saved in that folder
    with open(os.path.join(model_name_folder_path, model_name + '-configs.json' ), 'w') as outfile:
        json.dump(configs_dict, outfile, indent=4)

    return None

def load_drum_tabber_model(model_name, saved_models_path):
    '''
    Simple human-facing helper function to load drum_tabber models more easily. Assumes the saved model was
    created using the save_drum_tabber_model function (and thus will have a valid configs_dict in the folder)

    Args:
        model_name [str]:
        saved_models_path [str]:

    Returns:
        keras.model: drum_tabber model loaded from the saved file directory
        dict: configs_dict associated with the saved model
    '''
    drum_tabber = tf.keras.models.load_model(os.path.join(saved_models_path, model_name))

    with open(os.path.join(saved_models_path, model_name, model_name+'-configs.json'), 'r') as json_file:
        configs_dict = json.load(json_file)

    return drum_tabber, configs_dict

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
    if configs_dict['model_type'] == 'Context-CNN':
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

def detect_peaks(prediction, peak_pick_parameters = {'pre_max' : 2, 'post_max' : 2, 'pre_avg' : 20, 'post_avg' : 20, 'delta' : 0.5, 'wait' :5}):
    '''
    Detects peaks of the prediction output of the TF model. Powered by librosa.util.peak_pick

    Args:
        prediction [np.array]: output of an inference of a drum-tabber model. Has shape of (n_examples, n_classes)
        peak_pick_parameters [dict]: dictionary containing the parameters to be used in the librosa peak_pick function
                                    Default dictionary is {'pre_max' : 2, 'post_max' : 2, 'pre_avg' : 20, 'post_avg' : 20, 'delta' : 0.5, 'wait' :5}

    Returns:
        np.array: same shape as prediction array, but with 0s and 1s, where 1s identify the peaks for each class
    '''

    n_examples, n_classes = prediction.shape
    detected_peaks = np.zeros(shape = prediction.shape)

    for idx in range(n_classes):  # lb.util.peak_pick can accept only 1D data, so must use for loop to go through each class one at a time
        peaks_idx = lb.util.peak_pick(x = prediction[:,idx],
                                    pre_max = peak_pick_parameters['pre_max'], post_max = peak_pick_parameters['post_max'],
                                    pre_avg = peak_pick_parameters['pre_avg'], post_avg =  peak_pick_parameters['post_avg'],
                                    delta =  peak_pick_parameters['delta'], wait =  peak_pick_parameters['wait'])
        if len(peaks_idx) != 0:   # ensures no error occurs if no peaks are found and peak_pick returns an empty set (which means all peaks are 0, already)
            detected_peaks[peaks_idx, idx] = 1    # assign 1 at sample_idx where peak is detected, all other values are 0

    # TODO: if I wanted to implement CLASS-DEPENDENT peak pick parameters, then I would need to rewrite most of this function to introduce a
    #       idx-dependent parameters in the for loop. Additionally, code to check if the dictionary passed is applicable to all or class-dependent
    #       Class-dependent peak pick parameters basically would manually rectify the poor onset prediction performance in the more sparse classes

    return detected_peaks

def peaks_to_tab(detected_peaks, configs_dict):
    '''
    Changes the detected peaks array into a tab-like array by morphing the spectrogram-sized array into a time-based array.
    Heavily relies on detected peaks of beats being "correct" (to infer the time).

    Args:
        detected_peaks [np.array]: a 0s/1s array of shape (n_samples, n_classes) that denotes location of peaks. Output of detect_peaks function
        configs_dict [dict]:
    '''

    return None

# END OF MODEL SAVING, LOADING, AND INFERENCE FUNCTIONS
