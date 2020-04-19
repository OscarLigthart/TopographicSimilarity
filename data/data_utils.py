import numpy as np
import random
import pickle
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

        target = self.data[target_idx]

        return (target, distractors)

    def __len__(self):
        return self.data.shape[0]


class ReferentialSampler(Sampler):
    def __init__(self, data_source, samples, related: bool = False, k: int = 3, shuffle: bool = False,
                 split: int = 0, attr: int = 0, pair: int = 1, generalize: bool = False):
        self.data_source = data_source

        # beforehand, for every sample, we gather a list of objects that are fundamentally similar
        # to that sample. We sample randomly from this list to gather distractors --> should be a dictionary
        self.samples = samples
        self.related = related
        self.generalize = generalize

        # keep track of whether the dataloader should use a subset of targets (after having split the data)
        self.split = split
        self.attr = attr
        self.nr_attr = len(attr)
        self.pair = pair

        self.n = len(data_source)
        self.k = k
        self.shuffle = shuffle
        assert self.k < self.n

    def __iter__(self):
        indices = []

        # get the targets
        targets = list(range(self.n))

        # if we want to generalize we have custom targets, we need those indices
        if self.split:
            # load the generated split
            name = 'data/splits/split_{}_attr_{}_pair_{}.p'.format(self.split, self.nr_attr, self.pair)
            sub_targets = pickle.load(open(name, 'rb'))

            # get the keys of the split, they represent the indices
            sub_targets = list(sub_targets.keys())

            # replace the targets with the splits
            targets = sub_targets

        # shuffle the targets if requested
        if self.shuffle:
            random.shuffle(targets)

        # loop through targets to add distractors to every one of them
        for t in targets:

            # if we only want to sample related targets we use the enclosed dictionary holding these
            if self.related and self.generalize:

                # extract the samples
                samples = []

                for s in self.samples[t]:
                    samples.append(s[0])

                # target in first position with k random distractors following
                indices.append(
                    np.array(
                        [t]
                        + random.sample(
                            samples, self.k
                        ),
                        dtype=int,
                    )
                )

            elif self.related:

                # loop over the amount of attributes
                for a in range(self.nr_attr):

                    # initialize empty sample array
                    samples = []


                    # get the indices that should be the same
                    for s, i in self.samples[t]:

                        # check if the sample differs in the same attribute
                        if i == a:
                            samples.append(s)

                    # target in first position with k random distractors following
                    indices.append(
                        np.array(
                            [t]
                            + random.sample(
                                samples, self.k
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


def get_attributes(nr_attributes, related = False):

    # configure attributes
    attributes = [3, 3, 3, 3, 3, 2, 2, 2]

    # decide how many attributes to generate
    gen_attr = attributes[:nr_attributes]

    # first calculate the amount of samples created
    total_attr = np.prod(gen_attr)

    # make sure the dataset holds at least 243 samples (like baseline), by adding dimensions to attributes
    while total_attr < 243:  # and all(i >= 2 for i in gen_attr):
        index = np.argmin(gen_attr)
        gen_attr[index] += 1
        total_attr = np.prod(gen_attr)

    # create bigger dataset if we use stricter setup
    if related or len(gen_attr) == 4:
        gen_attr = [x+1 for x in gen_attr]

    return gen_attr


def get_close_samples(dataset, threshold=0.2, one_attribute = False):
    """
    This function takes the dataset filled with symbolic objects and
    finds closely related samples for each sample. It then returns a
    dictionary holding all samples along with a list of closely
    related samples for each sample.
    :param dataset: the set of samples
    :param threshold: value with which we decide the manner of required
                      similarity to make it as a distractor (value of 0.2 only allows one attribute to be different)
    :param generalize: choose whether the samples should only differ on the novel object properties
    :return: every target will have a dictionary holding all of the attributes, within these attributes there will
             be a list consisting of the distractors that differ from the target on only that attribute.
             thus, shape will be:
                    {target: [attr1: [distractors], attr2: [distractors]]}
    """

    # create dictionary in which we save possible distractors for every target,
    samples = defaultdict(list)

    # determine the amount of properties an attribute can take on
    attr_val = len(dataset[0]) / sum(dataset[0])

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

                # find which attribute differs
                diff_index = [i for i, p in enumerate(target) if p != distractor[i]][0]

                # convert index to attribute
                attr, _ = divmod(diff_index, attr_val)

                # for zero-shot generalization, only pick targets that differ on the
                # novel object traits (color and shape in our case)
                if one_attribute:

                    # if the other attributes are different, append it as target
                    # todo softcode --> should be able to do this while training
                    # todo while training, re-run a target with all available attributes
                    if np.array_equal(target[10:], distractor[10:]):
                        samples[t_index].append((d_index, attr))

                else:
                    # append sample
                    samples[t_index].append((d_index, attr))

    return samples


def get_referential_dataloader(
    file_name: str, gen_attr: list, batch_size: int = 32, shuffle: bool = False, k: int = 3,
        split=0, related=False, pair=0):
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
        data = generate_dataset(gen_attr, split, pair=pair)

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
            ReferentialSampler(dataset, samples, related=related, k=k, shuffle=shuffle, attr=gen_attr),
            batch_size=batch_size,
            drop_last=False,
        ),
    )
