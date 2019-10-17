import numpy as np
import random
import os
from scipy import spatial
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler

from .generate_dataset import generate_dataset


dir_path = os.path.dirname(os.path.realpath(__file__))

class ReferentialDataset(Dataset):
    """
    Referential Game Dataset
    """

    def __init__(self, data):
        self.data = data.astype(np.float32)

    def __getitem__(self, indices):

        target_idx = indices[0]
        distractors_idxs = indices[1:]

        distractors = []
        for d_idx in distractors_idxs:
            distractors.append(self.data[d_idx])

        target_img = self.data[target_idx]

        return (target_img, distractors)

    def __len__(self):
        return self.data.shape[0]


class ReferentialSampler(Sampler):
    def __init__(self, data_source, samples, related: bool = False, k: int = 3, shuffle: bool = False):
        self.data_source = data_source

        # beforehand, for every sample, we gather a list of objects that are fundamentally similar
        # to that sample. We sample randomly from this list to gather distractors --> should be a dictionary
        self.samples = samples
        self.related = related

        self.n = len(data_source)
        self.k = k
        self.shuffle = shuffle
        assert self.k < self.n

    def __iter__(self):
        indices = []

        targets = list(range(self.n))
        if self.shuffle:
            random.shuffle(targets)

        for t in targets:

            # if we only want to sample related targets we use the enclosed dictionary holding these
            if self.related:

                # target in first position with k random distractors following
                indices.append(
                    np.array(
                        [t]
                        + random.sample(
                            self.samples[t], self.k
                        ),
                        dtype=int,
                    )
                )

            # if we want completely random targets we just randomly sample from the entire dataset
            else:

                # target in first position with k random distractors following
                indices.append(
                    np.array(
                        [t]
                        + random.sample(
                            list(range(t)) + list(range(t + 1, self.n)), self.k
                        ),
                        dtype=int,
                    )
                )

        return iter(indices)

    def __len__(self):
        return self.n


def get_attributes(nr_attributes):

    # configure attributes
    attributes = [3, 3, 3, 3, 3, 2, 2, 2]

    # decide how many attributes to generate
    gen_attr = attributes[:nr_attributes]

    # first calculate the amount of samples created
    total_attr = np.prod(gen_attr)

    # make sure the dataset holds at least 150 samples, by adding dimensions to attributes
    while total_attr < 150:  # and all(i >= 2 for i in gen_attr):
        index = np.argmin(gen_attr)
        gen_attr[index] += 1
        total_attr = np.prod(gen_attr)

    return gen_attr


def get_close_samples(dataset, threshold=0.2):
    """
    This function takes the dataset filled with symbolic objects and
    finds closely related samples for each sample. It then returns a
    dictionary holding all samples along with a list of closely
    related samples for each sample.
    :param dataset: the set of samples
    :param gen_attr: the amount of attributes describing the data samples
    :param threshold: value with which we decide the manner of required
                      similarity to make it as a distractor (value of 0.2 only allows one attribute to be different)
    :return: every target will have a dictionary holding all of the attributes, within these attributes there will
             be a list consisting of the distractors that differ from the target on only that attribute.
             thus, shape will be:
                    {target: [attr1: [distractors], attr2: [distractors]]}
    """

    # create dictionary in which we save possible distractors for every target,
    samples = defaultdict(list)

    # convert to index, for every index, get close samples
    for t_index, target in enumerate(dataset):

        # go through all possible distractors
        for d_index, distractor in enumerate(dataset):

            # check if we're not comparing the same sample
            if t_index == d_index:
                continue

            # get distance between target and possible distractor object
            dist = spatial.distance.hamming(target, distractor)

            # if target and distractor are closely related, add the distractor
            # to the target in dict
            if dist < threshold:

                # append sample
                samples[t_index].append(d_index)

    return samples


def get_referential_dataloader(
    file_name: str, gen_attr: list, batch_size: int = 32, shuffle: bool = False, k: int = 3,
        split=False, related=False):
    """
    Splits a pytorch dataset into different sizes of dataloaders
    Args:
        file_name: filename to load
        gen_attr: number of attributes to generate
        batch_size : number of examples per batch_size
        shuffle (bool): whether to shuffle dataset
        k (int): number of distractors
        split: whether to keep some samples apart for purpose of testing generalization
        related: whether to save sample distractors based on similarity to targets
    Returns:
        dataloader
    """
    # load if already exists
    file_path = dir_path + "/" + file_name

    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        print("Generating dataset...")
        # create the attribute vector here
        data = generate_dataset(gen_attr, split)

        # save locally
        # np.save(file_path, data)

    # create the dictionary filled with closely related samples (if required)
    if related:
        samples = get_close_samples(data)
    else:
        samples = None

    dataset = ReferentialDataset(data)
    return DataLoader(
        dataset,
        pin_memory=True,
        batch_sampler=BatchSampler(
            ReferentialSampler(dataset, samples, related=related, k=k, shuffle=shuffle),
            batch_size=batch_size,
            drop_last=False,
        ),
    )
