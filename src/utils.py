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
from src.configs import *


class MusicAlignedTab(object):
    """
    MusicAlignedTab is the class used to store the pre-processed object that will eventually become a part of the training set
    This class is mainly used to give the user a view into the alignments, augmentation effects, and other attributes that
    might need user verification before further processing is done.
    """
    def __init__(self, song_name):   # song_name is the string in the form of the folder names ('forever_at_last')
        self.song_folder_fp = os.path.join(SONG_PATH, song_name)
        # TODO: maybe add try except block here on importing of json file?
        self.json_fp = os.path.join(self.song_folder_fp, song_name+'.json')
        self.json_dict = self.parse_json_file()              # json_dict has 'song', 'tab_file', 'tab_char_labels', and 'alignment_info' as keys
        self.song_fp = os.path.join(self.song_folder_fp, self.json_dict['song'])
        self.tab_fp = os.path.join(self.song_folder_fp, self.json_dict['tab_file'])
        self.hf_tab = self.import_tab()
        self.mf_tab = hf_to_mf(self.hf_tab)
        self.df_MAT = self.align_tab_with_music()

    def parse_json_file(self):
        """
        Parses the JSON file information in the song folder to be utilized in the MusicAlignedTab

        Args:
            self [MusicAlignedTab]: can access the attributes of the class

        Returns:
            dict: dictionary derived from the JSON file
        """
        with open(self.json_fp, 'r') as json_file:
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
        tab_char_labels = self.json_dict   # getting the tab_char_labels dictionary for the object's json_dict
        tab_conversion_dict = {}      # dictionary whose keys are the CURRENT TAB's char labels and whose values are the MASTER FORMAT's char labels
        for key in tab_char_labels:
            if key in MASTER_FORMAT_DICT:   # ensures the keys are in both sets, which they should be
                tab_conversion_dict[tab_char_labels[key]] = MASTER_FORMAT_DICT[key]

        # read in the tab .txt file
        with open(self.tab_fp, 'r') as tab_file:
            tab_to_convert = tab_file.readlines()  # returns a python array of the text file with each line as a string, including the \n new line character at the end of each string

        # build this array tab up and return it
        tab = []
        for line in tab_to_convert:                 # loop over each element in tab (code line in tab)
            for key in tab_conversion_dict:          # At each line, loop over each key in the tab_conversion_dict
                if line.startswith(key):             # If a line starts with the key, reassign it to be the new label from dict...
                    tab.append(tab_conversion_dict[key] + line[len(key):])   # ...and concatenate with the rest of that line
                else:
                    tab.append(line)
        return tab

    def hf_to_mf(self, tab):
        """
        High level function that changes a tab in human-friendly format to machine-friendly formatting

        Args:
            self [MusicAlignedTab]: can access the attributes of the class
            tab [list]:

        Returns:
            list: list of strings, each of which . Represents the entire machine-friendly tab
        """

        # change the tab to include "white space" drum piece lines whenever there are tab lines WITHOUT a drum piece label
        expanded_tab = expand_tab(tab)

        # align all the drum piece lines in the entire tab with each other into one long line for every drum piece line
        mf_tab = combine_lines(expanded_tab)

        return mf_tab

    def expand_tab(tab):
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
        split_on_tk_reversed = reverse_then_split_on_tk(tab)

        # now we have an array of strings, where each string either begins with tk_chars, or is the first element in the array
        # For each element that starts with tk_chars, we check which drum piece labels exist in that element, and
        # add in the appropriate length empty line for each missing drum piece, and then combine
        # This is handled by the subfunction add_empty_lines
        expanded_split_on_tk_reversed = add_empty_lines(split_on_tk_reversed, drum_pieces_used)

        # To make the future appending function easier, we want to sort the drum piece labels so that each tk line has the same ordered drum lines after.
        ordered_ex_split_on_tk_reversed = order_tk_lines(expanded_split_on_tk_reversed, drum_pieces_used)

        # now we want to undo the structural changes made to the tab
        combined_ordered_ex_reversed = ''.join(ordered_ex_split_on_tk_reversed)          # combine the tk split elements into one string
        ordered_expanded_reversed_tab = combined_ordered_ex_reversed.splitlines(keepends=True) # separate into different individual lines, keeping the newline character

        expanded_tab = list(reversed(ordered_expanded_reversed_tab))    # reverse the tab entirely

        return expanded_tab

    def reverse_then_split_on_tk(tab):
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

    def add_empty_lines(split_on_tk_reversed, drum_pieces_used):
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

    def order_tk_lines(expanded_split_on_tk_reversed, drum_pieces_used):
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

    def combine_lines(expanded_tab):
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
        exp_sotk_rev_tab = reverse_then_split_on_tk(exp_tab)

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
