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
import numpy as np
import pandas as pd
from pydub import AudioSegment   # main class from pydub package used to upload mp3 into Python and then get a NumPy array
import IPython.display as ipd    # ability to play audio in Jupyter Notebooks if needed
import librosa as lb             # loads the librosa package

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

    # START OF MAIN HIGH LEVEL FUNCTIONS
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
        with open(self.filepaths['tab'], 'r') as tab_file:
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
    # END OF MAIN HIGH LEVEL FUNCTIONS

    # HUMAN-FACING CLASS UTILITY FUNCTIONS
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
                drop_MAT = MAT_df.drop(columns = ['song slice'])          # drop the slices column so we are left with only tab
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

    def labels(self):
        '''

        '''
        return None

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

        lb_song, _ = lb.core.load(song_title, sr=None, mono=channel_mono) # uses librosa to output a np.ndarray of shape (n,) or (2,n) depending on the channels

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
            Dataframe: contains 'song slice', 'tk' and drum piece label columns, aligned properly. Basically the sum of 'song slice' should be most of the song if converted back to audio and played
        """
        tk_label, measure_char, blank_char = TK_LABEL, MEASURE_CHAR, BLANK_CHAR # grab the special chars, we'll need the blank_char and measure_char

        # TODO: Allow Tempo Changes into the set of tabs/music that can be processed
        if alignment_info['tempo change'] == True: # checks if the tab has an tempo changes
            print("This song has a tempo change, rejecting the tab for now.")
            # return tab_df    # If so, return the tab dataframe unchanged because we don't want to deal with that right now

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
        print("All the following prints are from combine_tab_and_song function:")
        print("first drum note row = " + str(fdn_row_index))
        tab_len = len(tab_df.index)     # total length of the drum tab dataframe

        # slice up raw audio AFTER the first drum note, correcting for potential misalignment due to lopping off remainder of sample delta, and handling the triplet case
        song_slices_post_fdn = self.get_post_fdn_slice(song, fdn_sample_loc, sample_num, sample_delta, decimal_delta, triplets_bool, tab_df, fdn_row_index)

        # slice up raw audio BEFORE the first drum note, correcting for potential misalignment due to lopping off remainder of sample delta
        song_slices_pre_fdn = self.get_pre_fdn_slice(song, fdn_sample_loc, sample_num, sample_delta, decimal_delta)

        """PRINTING USEFUL OUTPUT FOR SANITY CHECK"""
        print('# of song slices post fdn = ' + str(len(song_slices_post_fdn)))
        print('# of song slices pre fdn = ' + str(len(song_slices_pre_fdn)))
        print("Produced number of song slices = " + str(len(song_slices_pre_fdn) + len(song_slices_post_fdn)))
        print("Expected number of song slices (should be same for non-triplet songs) = " + str(sample_num/(sample_delta+decimal_delta)))

        # take the two song slices, and the location of the first drum note index, and produce a dataframe of the same length as the tab frame,
        # with the song slices in the correct index position
        song_slices_df = self.slices_into_df(song_slices_pre_fdn, song_slices_post_fdn, fdn_row_index, tab_len)

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

    def slices_into_df(self, slices_pre_fdn, slices_post_fdn, fdn_row_index, tab_len):
        """
        Helper subfunction in combine_tab_and_song that pushes the song slices into its own df.

        Args:
            slices_pre_fdn [list]: list of np.arrays that are the raw audio of the song slices pre first drum note
            slices_post_fdn [list]: list of np.arrays that are the raw audio of the song slices post first drum note
            fdn_row_index [int]: index of the row of the first drum note in the tab dataframe
            tab_len [int]: length of the tab dataframe

        Returns:
            Dataframe: dataframe of one column named 'song slice' that contains all the rows of the song slices, in the correct index position, to afterwards be immediately adjoined with the tab
        """

        partial_slices_post_fdn = slices_post_fdn[0: tab_len - fdn_row_index] # grabs the first tab_len - fdn_row_index of slices post fdn
        partial_slices_pre_fdn = slices_pre_fdn[len(slices_pre_fdn) - fdn_row_index :]  # grabs the last fdn_row_index number of slices from slices_pre_fdn
        song_slices_tab_indexed = partial_slices_pre_fdn + partial_slices_post_fdn   # concatenates the two arrays

        """PRINTING USEFUL OUTPUT FOR SANITY CHECK"""
        print("tab length = " + str(tab_len) + "     datatype: " + str(type(tab_len)))
        print("len(song_slices_tab_indexed) = " + str(len(song_slices_tab_indexed)) + "     datatype of object: " + str(type(song_slices_tab_indexed)))
        print("song_slices_tab_indexed[0].shape = " + str(song_slices_tab_indexed[0].shape) + "     datatype of [0]: " + str(type(song_slices_tab_indexed[0])))
        print("np.array(song_slices_tab_indexed).shape = " + str(np.array(song_slices_tab_indexed).shape))

        # Force the current "list" into a dictionary so that pandas ALWAYS INTERPRETS IT AS A LIST OF ARRAYS, not just one single large 3D matrix
        tab_dict = {'song slice': song_slices_tab_indexed} # HARD CODED name
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

def create_FullSetMAT(songs_file_path):
    '''
    Function used to create a FullSetMAT to inspect summary statistics on an entire initial dataset of MATs, or to randomly look at different

    Args:
        songs_file_path [str]: Filepath to the songs folder

    Returns:
        Dataframe: the FullSet dataframe of all songs' dataframes stacked on top of each other (and outputs information to the user display)
    '''

    # need to create a MusicAlignedTab object on every song in the song_folder_path given to us, then extract Dataframe, then concatenate
    MATDF_dict = {}  # dictionary where the keys are the string of the name of the song, and the values are the MAT class object

    subdirs = [os.path.join(songs_file_path, o) for o in os.listdir(songs_file_path) if os.path.isdir(os.path.join(songs_file_path,o))] # grab all the subdirectories in the song_folder_path
    list_of_songs = [os.path.basename(os.path.normpath(x)) for x in subdirs]  # grabbing only the end of the song folders (that is, the song title)
    print(f'subdirs = {subdirs}')
    print(f'list_of_songs = {list_of_songs}')

    for song in list_of_songs:                # go through all the songs folders
        MAT_class = MusicAlignedTab(song)     # create the MAT class object
        MATDF_dict[song] = MAT_class.MAT      # extract and keep only the dataframe of the class object

    # get blank char to use in this function
    blank_char = BLANK_CHAR

    # stacks all the dataframes one on top of another in the rows, ignoring indices of each dataframe, and giving each frame an extra "key" layer
    print("...Concatenating all music-aligned dataframes")
    output_df = pd.concat(MATDF_dict, axis=0, ignore_index = False, join = 'outer', sort = False)

    print("...Replacing NaNs with " + blank_char + " for output")
    output_df = output_df.fillna(blank_char)               # replace NaN with the blank_char

    print("...Dropping the song slices for ease of display \n")
    full_df = output_df.drop(columns = ['song slice'])    # drop the song slice info because we don't care about them right now

    print("---fullset.describe() without blank_chars---")
    print(full_df[full_df != blank_char].describe(), '\n')

    print("Unique values and frequencies in column __:")
    for col in full_df:
        naf_series = full_df[col].value_counts()
        print(str(naf_series.to_frame().T))
        print()

    return output_df
