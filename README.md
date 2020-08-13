# Drum-Tabber

This repository is for an automatic drum transcription project created by me. The main purpose of the project is to develop and display my data science and machine learning skills and knowledge with a challenging, lofty goal. I am interested in creating an automatic drum tabber by using real music with labelled drums as a training set. The training set and labels would be derived from currently existing, freely available drum tabulatures, aligned to the music properly such that it can assign a tiny slice of that song with a label. After that, a convolutional neural network architecture is used to train a model to predict the drum onset events in a song. Note that the goal of the project is not to produce unique drum rhythms, but only to classify drums in a song. 

August 12th Update: 
After taking a break from this project to work on a different machine learning neural network based project, I've learned quite a few things from that project that I can apply to this Drum-Tabber project. Here are the things I believe I want to change. 
1. **Data augmentation implemented in the Training/Dataset Pipeline**: Similar to the Drum-Tabber project, my other project was very data-limited. So I needed to create some data augmentation functions to help with dealing with that problem. That project was a computer vision object detection problem, so there were plenty of suggestions online with dealing with augmenting images and their bounding box labels. Drum-Tabber is, eventually, a computer vision problem because it turns audio data into a spectrogram, which is effectively a 2D image for data processing purposes. However, there is a key difference between the two projects with respect to data augmentation techniques. In a computer vision application, when implementing, for example, a random brightness function, when testing the function you can output the augmented image to check with your own eyes whether that augmentation makes sense. For random brightness, there could be brightness values that don't make sense because it distorts the image *too* much, thereby giving the model garbage data (which is bad). A spectrogram, although treated as a 2D image by the computer, has no real significant, visual meaning to a *human*. Augmenting the spectrogram with computer vision-like functions (changing brightness/contrast/translation/horizontal flip/vertical flip/etc.) is not guaranteed to produce something that makes sense aurally. **We need to ensure that the augmented training data being fed into the model is not garbage data.** The way to ensure this is to augment the *audio data* with audio filters *before* passing it into the functions used to convert audio data into spectrograms. In this way, a human could listen to the augmented audio data to ensure the data isn't *too* augmented, analagous to the brightness function situation described above. We assume that an augmented song's spectrogram will be meaningfully different enough from its non-augmented spectrogram, but not different enough that it won't represent how drum events are encoded into a spectrogram. In this way the data will better represent the general structure of a drum event for any drum class. 

2. **Using current code as quality-checking for new input data, and creating new code for X and Y training set creation (not based on the drum tab grid).**: The current code that is used to construct the X and Y datasets from the all_songs_dataframe will more than likely be thrown out. However, all the code up to that point, where you can randomly sample a part of a song and align it with the drum tabs to check that the song slices line up with the drum tabs, should continue to be used as a quality-checking code, especially when the augmentation functions are implemented. After you run a song through an audio filter, you can check the tab alignment to ensure that the augmentation didn't change anything too much (with respect to the audio itself representing a drum event, or the possibility of disturbing the alignments if a time-affecting augmentation is used). Furthermore, related to the point 1 above, new code needs to be created to make the training set so that it has the data augmentation functions in the training pipeline. 

3. **Training on Google Colab or Microsoft Azure cloud based GPUs**: The other project involved a tutorial which at some point showed how to train using Google Colab. After doing that and realizing that training was **way** faster, I pretty much need to train using any of those. Because these resources are limited, I should code everything and start training using my computer, and then transfer over everything when the code is proven to work and a model can be trained. 

4. **Convert general options across different functions to a configs.py file**: The other project used a "configs.py" file that contains a bunch of "global variables" that are named with all upper case (TRAIN_EPOCHS for example). I really like this concept and it is probably standard in other projects. The other .py files import this file so that it has access to all the configuration options that have existed in your code. I already can imagine which options to include in this configs.py file. 
