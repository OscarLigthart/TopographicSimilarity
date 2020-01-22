"""
Calculates all the RSA for all run files
"""
import glob
import pickle
import os
import sys
import argparse
from itertools import combinations
from scipy import spatial
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
        action="store_true",
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
        "--hidden-size",
        type=int,
        default=64,
        metavar="N",
        help="hidden size for hidden layer (default: 64)",
    )
    parser.add_argument(
        "--related",
        help="Decide whether to use distractors that are semantically similar to the targets",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--split",
        help="Decide whether to use all generated samples or keep some apart for testing generalization",
        type=int,
        default=0
    )
    parser.add_argument(
        "--freeze-sender",
        help="Freeze the sender in the setup, so the language stays constant",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--freeze-receiver",
        help="Freeze the receiver in the setup, so the interpretation stays constant",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--freeze-seed", type=int, default=2, help="random seed (default: 1)"
    )

    args = parser.parse_args(args)
    return args


def flatten_cos(a, b) -> float:
    return spatial.distance.cosine(a.flatten(), b.flatten())


def on_hot_hamming(a, b):
    return spatial.distance.hamming(
        one_hot(a, n_cols=VOCAB).flatten(), one_hot(b, n_cols=VOCAB).flatten()
    )

def compute_jaccard_distance_score(x, y):
    """
    Jaccard Similarity J (A,B) = | Intersection (A,B) | /
                                    | Union (A,B) |
    """
    intersection_cardinality = len(set(x).intersection(set(y)))
    union_cardinality = len(set(x).union(set(y)))

    # 1 - result since we want the distance
    return 1 - (intersection_cardinality / float(union_cardinality))

def levenshtein_ratio_and_distance(s, t):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows, cols), dtype=int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # If we choose to calculate the ratio the cost of a substitution is 2.
                cost = 2

            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions

    # Computation of the Levenshtein Distance Ratio
    Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
    return Ratio


DIST = {
    #"h_sender": spatial.distance.cosine,
    #"h_rnn_sender": flatten_cos,
    #"h_receiver": spatial.distance.cosine,
    #"h_rnn_receiver": flatten_cos,
    "targets": spatial.distance.hamming,
    "messages": compute_jaccard_distance_score
}

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
    #path = "runs/lstm_h_{}_max_len_{}_vocab_{}".format(args.hidden_size, args.max_length, args.vocab_size)

    # get correct path name, based on parameters
    if args.same_data:
        path += "_same_data"
    path += "_attr_{}".format(args.attributes)
    if args.related:
        path += "_related"
    if args.split:
        path += "_split_{}".format(args.split)

    # check if sender is frozen
    if args.freeze_sender:
        path += "/" + str(args.freeze_seed) + "/freezes_sender"

    # check if receiver is frozen
    if args.freeze_receiver:
        path += "/" + str(args.freeze_seed) + "/freezes_receiver"

    # # within seed RSA for all
    metric_files = glob.glob(f"{path}/*/*.pkl")

    for file in tqdm(metric_files):
        m = pickle.load(open(file, "rb"))
        for (space_x, space_y) in combinations(list(DIST.keys()), 2):
            rsa_title = f"RSA:{space_x}/{space_y}"
            #if rsa_title not in m:
            r = rsa(m[space_x], m[space_y], DIST[space_x], DIST[space_y], number_of_samples=args.samples)
            m[rsa_title] = r

        pickle.dump(m, open(file, "wb"))

    # Calculate Cross-Seed RSA for all
    seed_folders = glob.glob(f"{path}/*")

    # we are not interested in cross-seed RSA for targets
    DIST.pop('targets')
    #DIST.pop('messages')

    #DIST["ham_messages"] = on_hot_hamming
    #DIST["lev_messages"] = levenshtein_ratio_and_distance


    RESULTS = {}

    for sp in DIST:

        # use key messages for hamming and lev distance, since data is saved in that way
        if sp == "ham_messages" or sp == "lev_messages":
            space = "messages"

        # use regular keys for other datapoints
        else:
            space = sp

        RESULTS[sp] = {}

        # compare every seed to all others
        for s1, s2 in tqdm(combinations(seed_folders, 2)):

            seed1 = s1.split("/")[-1]
            seed2 = s2.split("/")[-1]

            RESULTS[sp][seed1 + '-' + seed2] = {}

            # load all metric files
            files_s1 = glob.glob(f"{s1}/*.pkl")

            # get the first file
            for f1 in files_s1:

                # parse the path to get the second file
                metric_file = f1.split("/")[-1]

                if 'generalize_metrics.pkl' in f1:
                    iteration = 'generalize'
                else:
                    iteration = int(metric_file.split("_")[-1].split(".")[0])

                # get second file
                f2 = f"{s2}/{metric_file}"

                if os.path.isfile(f2):

                    # load the different rsa analysis files
                    m1 = pickle.load(open(f1, "rb"))
                    m2 = pickle.load(open(f2, "rb"))

                    # use actual space for to extract data and sp to differentiate message similarity spaces
                    r = rsa(m1[space], m2[space], DIST[sp], DIST[sp], number_of_samples=args.samples)
                    RESULTS[sp][seed1 + '-' + seed2][iteration] = r

    # save results
    pickle.dump(RESULTS, open(f"{path}/rsa_analysis.pkl", "wb"))


if __name__ == "__main__":
    main(sys.argv[1:])
