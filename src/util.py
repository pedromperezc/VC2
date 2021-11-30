import numpy as np

def rle_to_mask(lre, shape=(1600, 256)):
    '''
    params:  rle   - run-length encoding string (pairs of start & length of encoding)
             shape - (width,height) of numpy array to return

    returns: numpy array with dimensions of shape parameter
    '''
    # the incoming string is space-delimited
    runs = np.asarray([int(run) for run in lre.split(' ')])

    # we do the same operation with the even and uneven elements, but this time with addition
    runs[1::2] += runs[0::2]
    # pixel numbers start at 1, indexes start at 0
    runs -= 1

    # extract the starting and ending indeces at even and uneven intervals, respectively
    run_starts, run_ends = runs[0::2], runs[1::2]

    # build the mask
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    for start, end in zip(run_starts, run_ends):
        mask[start:end] = 1

    # transform the numpy array from flat to the original image shape
    return mask.reshape(shape)