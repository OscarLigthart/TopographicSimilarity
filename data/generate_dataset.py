import itertools
import numpy as np
from typing import Optional
import pickle
from itertools import combinations

# 3 colors, 3 shapes, 2 sizes, 3 position y, 3 position x
SHAPES_ATTRIBUTES = [3, 3, 2, 3, 3]


def one_hot(a, n_cols: Optional[int] = None):
    if n_cols is None or n_cols < a.max() + 1:
        n_cols = a.max() + 1
    out = np.zeros((a.size, n_cols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (n_cols,)
    return out


def generate_dataset(atttribute_vector: list = SHAPES_ATTRIBUTES, split: int = 0, pair: int = 1):
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
        one_hot_derivations = split_data(split, one_hot_derivations, pair)

    return one_hot_derivations


def split_data(split, one_hot_derivations, pair=1):
    """
    Required for zero-shot generalization. It removes samples from the dataset based
    on an amount of co-occurring attributes/features.
    The split samples are saved in a file which can be loaded for the generalization
    :param: split:                  the amount of co-occurring attributes (default = 2)
    :param: one_hot_derivations:    the dataset consisting of one-hot vectored object attributes
    :param: pair:                   The object pair to remove from the dataset (default = 1, e.g. "red square")
    :return: dataset without split samples
    """

    # take a number of co-occurrences
    k = split

    # we need to keep track of the filtered attributes
    filtered_attr = []

    # start by using the first target (get by index)
    index = 0

    # initialize empty list in which we store the samples for the split set
    samples = {}

    # extract all attribute pairs
    while index < len(one_hot_derivations):

        # remove a co-occurring sample (for instance, "red square")
        # we remove all instances of [1 0 0 1 0 0]
        # so take the first attribute and find the first k amount of ones
        sample = one_hot_derivations[index]
        attr = np.where(sample == 1)[0]

        # find the co occurring samples in the dataset
        for j in range(sum(one_hot_derivations[0]) - k):
            occurrences = attr[j:j + k]

            # check if we have not filtered the same combination of attributes before
            if tuple(occurrences) not in filtered_attr:
                filtered_attr.append(tuple(occurrences))

        # next time we take the next target
        index += 1

    # get the requested pair
    occurrences = np.array(filtered_attr[pair-1])

    # find all elements where the occurrences are the same
    for i, s in enumerate(one_hot_derivations):
        attr = np.where(s == 1)[0]

        # find the co occurring samples in the dataset
        for j in range(sum(one_hot_derivations[0]) - k):
            occ = attr[j:j + k]

            # check if sample holds the attributes
            if np.array_equal(occurrences, occ):

                # add sample to split set, keep index as well
                samples[i] = s

    # delete sample from dataset
    for s in samples.values():
        one_hot_derivations = [j for j in one_hot_derivations if not np.array_equal(j, s)]

    # get name for samples split
    name = 'data/splits/split_{}_attr_{}_pair_{}.p'.format(split, sum(one_hot_derivations[0]), pair)

    # save selected samples
    pickle.dump(samples, open(name, "wb"))

    # cast to numpy array
    one_hot_derivations = np.array(one_hot_derivations)

    return one_hot_derivations
