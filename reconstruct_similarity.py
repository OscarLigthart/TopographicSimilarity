"""
Calculates all the RSA for all run files
"""
import glob
import pickle
import os
import sys
import argparse
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from scipy import signal
import scipy
from scipy import spatial
import sklearn
from metrics import rsa
from data import one_hot
from tqdm import tqdm
import numpy as np


def parse_arguments(args):
    """
    Determines on which experiment RSA needs to be calculated
    :param args:
    :return:
    """
    parser = argparse.ArgumentParser(
        description="Trains a Sender/Receiver Agent in a referential game"
    )
    parser.add_argument(
        "--same-data",
        type=bool,
        default=False,
        help="determine whether data is created on same seed or not"
    )
    parser.add_argument(
        "--attributes",
        type=int,
        default=5,
        help="determine whether data is created on same seed or not"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="determine whether data is created on same seed or not"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=10,
        metavar="N",
        help="max sentence length allowed for communication (default: 10)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=25,
        metavar="N",
        help="Size of vocabulary (default: 25)",
    )
    parser.add_argument(
        "--distractors",
        help="Decide on the amount of distractors to use",
        type=int,
        default=3
    )
    parser.add_argument(
        "--related",
        help="Decide whether to use distractors that are semantically similar to the targets",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--split",
        help="Decide whether to use all generated samples or keep some apart for testing generalization",
        type=bool,
        default=False
    )

    args = parser.parse_args(args)
    return args


def main(args):
    """
    This function calculates Cross-Seed RSA
    :param args: arguments determining for which experiment RSA needs to be calculated
    :return: Calculated RSA values
    """

    # retrieve arguments
    args = parse_arguments(args)

    # make vocab global so other functions can access it as well
    global VOCAB
    VOCAB = args.vocab_size + 3

    # determine path to calculate Cross-Seed RSA on
    path = "runs/lstm_max_len_{}_vocab_{}".format(args.max_length, args.vocab_size)

    # get correct path name, based on parameters
    if args.same_data:
        path += "_same_data"
    path += "_attr_{}".format(args.attributes)
    if args.related:
        path += "_related"
    if args.split:
        path += "_split"

    metric_files = glob.glob(f"{path}/*/*.pkl")

    # Calculate Cross-Seed RSA for all
    seed_folders = glob.glob(f"{path}/*")

    RESULTS = {}

    all_distances = []
    # compare every seed to all others
    for s1, s2 in combinations(seed_folders, 2):

        seed1 = s1.split("/")[-1]
        seed2 = s2.split("/")[-1]

        if len(seed1 + seed2) < 5:

            RESULTS[seed1 + seed2] = {}

        # load all metric files
        files_s1 = glob.glob(f"{s1}/*.pkl")

        # get the first file
        for f1 in files_s1:

            # parse the path to get the second file
            metric_file = f1.split("/")[-1]
            iteration = int(metric_file.split("_")[-1].split(".")[0])
            if iteration != 9800:
                continue

            # get second file
            f2 = f"{s2}/{metric_file}"

            if os.path.isfile(f2):

                # load the different rsa analysis files
                m1 = pickle.load(open(f1, "rb"))
                m2 = pickle.load(open(f2, "rb"))

                # get receiver values
                h_receiver_1 = m1['h_receiver']
                h_receiver_2 = m2['h_receiver']

                # get similarity values for all instances
                distances = []

                for i in range(len(h_receiver_1)):
                    dis = spatial.distance.cosine(h_receiver_1[i], h_receiver_2[i])
                    distances.append(dis)

                RESULTS[seed1 + seed2] = np.mean(distances)

    print(RESULTS)



if __name__ == "__main__":
    main(sys.argv[1:])
