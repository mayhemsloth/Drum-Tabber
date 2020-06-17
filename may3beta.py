song = 'may3beta.mp3'
tab_file= 'may3beta_tab.txt'
tab_char_labels = { 'bass drum'     :'B ',
                    'snare drum'    :'S ',
                    'high tom'      :'sT',
                    'mid tom'       :'mT',
                    'low tom'       :'FT',
                    'hi-hat'        :'HH',
                    'ride cymbal'   :'R ',
                    'crash cymbal'  :'C ',
                    'crash cymbal 2':'C2',     # extra cymbals if needed, classification will ultimately collapse later
                    'crash cymbal 3':'C3',     # extra cymbals if needed, classification will ultimately collapse later
                    'crash cymbal 4':'C4',     # extra cymbals if needed, classification will ultimately collapse later
                    'splash cymbal' :'SC',
                    'china cymbal'  :'CH'
                    }
alignment_info = {'triplets' : False, 
                        'tempo change' : False,
                        'BPM' : 135,
                        'first drum note onset' : 1.770}  # this is accurate information for our test song may3beta