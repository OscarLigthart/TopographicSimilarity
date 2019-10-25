import itertools
import numpy as np
from typing import Optional
import pickle

# 3 colors, 3 shapes, 2 sizes, 3 position y, 3 position x
SHAPES_ATTRIBUTES = [3, 3, 2, 3, 3]


def one_hot(a, n_cols: Optional[int] = None):
    if n_cols is None or n_cols < a.max() + 1:
        n_cols = a.max() + 1
    out = np.zeros((a.size, n_cols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (n_cols,)
    return out


def generate_dataset(atttribute_vector: list = SHAPES_ATTRIBUTES, split: int = 0):
    """
    Generates a dataset based on the vector of attributes passed
    """

    possible_values = []
    for attribute in atttribute_vector:
        possible_values.append(list(range(attribute)))

    # get all possible values
    all_possible_values = np.array(list(itertools.product(*possible_values)))

    # one hot encode
    one_hot_derivations = one_hot(all_possible_values).reshape(
        all_possible_values.shape[0], -1
    )

    # compress one hot encoding (remove all 0 only columns)
    remove_idx = np.argwhere(np.all(one_hot_derivations[..., :] == 0, axis=0))
    one_hot_derivations = np.delete(one_hot_derivations, remove_idx, axis=1)

    # randomly samply from possible combinations
    # idxs = np.random.choice(range(len(one_hot_derivations)), size, replace=True)

    # optionally split dataset
    if split:
        one_hot_derivations = split_data(split, one_hot_derivations)

    return one_hot_derivations


def split_data(split, one_hot_derivations):
    """
    This function splits data based on an amount of co-occurring attributes/features
    The split samples are saved in a file which can be loaded for the generalization
    :return: dataset without split samples
    """

    # take a number of co-occurrences
    k = split

    # remove a random coocccurring sample (for instance, "red square")
    # we remove all instances of [1 0 0 1 0 0]
    # needs to be variable across different attribute sizes
    # so take the first attribute and find the first k amount of ones
    sample = one_hot_derivations[0]
    attr = np.where(sample == 1)[0]
    occurrences = attr[:k]

    # initialize empty list in which we store the samples for the split set
    samples = {}

    # find all elements where the occurrences are the same
    for i, s in enumerate(one_hot_derivations):
        attr = np.where(s == 1)[0]
        occ = attr[:k]
        if np.array_equal(occurrences, occ):

            # add sample to split set, keep index as well
            samples[i] = s

            # delete sample from dataset
            one_hot_derivations = [j for j in one_hot_derivations if not np.array_equal(j, s)]

    # get name for samples split
    name = 'data/generalize_split_{}.p'.format(split)

    # save selected samples
    pickle.dump(samples, open(name, "wb"))

    # cast to numpy array
    one_hot_derivations = np.array(one_hot_derivations)

    return one_hot_derivations
