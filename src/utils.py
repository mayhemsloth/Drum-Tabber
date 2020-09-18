#================================================================
#
#   File name   : utils.py
#   Author      : Thomas Hymel
#   Created date: 9-2-2020
#   GitHub      : https://github.com/mayhemsloth/Drum-Tabber
#   Description : Tools for pre-processing and post-processing of data
#
#================================================================


from src.configs import *



'''
TAKEN FROM CODE OF RICHARD VOGL from "TOWARDS MULTI-INSTRuMENT DRUM TRANSCRIPTION" 2018 paper
Used to see an implementation of the peak picking code referenced in the paper
'''
def peak_picking(activations, threshold, smooth=None, pre_avg=0, post_avg=0,
                 pre_max=1, post_max=1):
    """
    Perform thresholding and peak-picking on the given activation function.

    Parameters
    ----------
    activations : numpy array
        Activation function.
    threshold : float
        Threshold for peak-picking
    smooth : int or numpy array, optional
        Smooth the activation function with the kernel (size).
    pre_avg : int, optional
        Use `pre_avg` frames past information for moving average.
    post_avg : int, optional
        Use `post_avg` frames future information for moving average.
    pre_max : int, optional
        Use `pre_max` frames past information for moving maximum.
    post_max : int, optional
        Use `post_max` frames future information for moving maximum.

    Returns
    -------
    peak_idx : numpy array
        Indices of the detected peaks.

    See Also
    --------
    :func:`smooth`

    Notes
    -----
    If no moving average is needed (e.g. the activations are independent of
    the signal's level as for neural network activations), set `pre_avg` and
    `post_avg` to 0.
    For peak picking of local maxima, set `pre_max` and  `post_max` to 1.
    For online peak picking, set all `post_` parameters to 0.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Markus Schedl,
           "Evaluating the Online Capabilities of Onset Detection Methods",
           Proceedings of the 13th International Society for Music Information
           Retrieval Conference (ISMIR), 2012.

    """
    # smooth activations
    activations = smooth_signal(activations, smooth)
    # compute a moving average
    avg_length = pre_avg + post_avg + 1
    if avg_length > 1:
        # TODO: make the averaging function exchangeable (mean/median/etc.)
        avg_origin = int(np.floor((pre_avg - post_avg) / 2))
        if activations.ndim == 1:
            filter_size = avg_length
        elif activations.ndim == 2:
            filter_size = [avg_length, 1]
        else:
            raise ValueError('`activations` must be either 1D or 2D')
        mov_avg = uniform_filter(activations, filter_size, mode='constant',
                                 origin=avg_origin)
    else:
        # do not use a moving average
        mov_avg = 0
    # detections are those activations above the moving average + the threshold
    detections = activations * (activations >= mov_avg + threshold)
    # peak-picking
    max_length = pre_max + post_max + 1
    if max_length > 1:
        # compute a moving maximum
        max_origin = int(np.floor((pre_max - post_max) / 2))
        if activations.ndim == 1:
            filter_size = max_length
        elif activations.ndim == 2:
            filter_size = [max_length, 1]
        else:
            raise ValueError('`activations` must be either 1D or 2D')
        mov_max = maximum_filter(detections, filter_size, mode='constant',
                                 origin=max_origin)
        # detections are peak positions
        detections *= (detections == mov_max)
    # return indices
    if activations.ndim == 1:
        return np.nonzero(detections)[0]
    elif activations.ndim == 2:
        return np.nonzero(detections)
    else:
        raise ValueError('`activations` must be either 1D or 2D')
