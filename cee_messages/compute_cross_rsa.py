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
from tqdm import tqdm
import numpy as np
import itertools
import numpy as np
import scipy.stats
import warnings
from typing import Sequence, Callable, Optional


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
        "--samples",
        type=int,
        default=None,
        help="determine whether data is created on same seed or not"
    )

    args = parser.parse_args(args)
    return args


def rsa(
    space_x: Sequence[Sequence],
    space_y: Sequence[Sequence],
    distance_function_x: Callable[[Sequence, Sequence], float],
    distance_function_y: Callable[[Sequence, Sequence], float],
    number_of_samples: Optional[int] = None,
) -> float:
    """
    Calculates RSA using all possible pair combinations in both space given distance functions
    Args:
        space_x: representations in space x
        space_y: representations in space y (note these represensations must match in the 1st dimension)
        distance_function_x: distance function used to measure space x
        distance_function_y: distance function used to measure space y
        number_of_samples (int, optional): if passed, uses a random number of pairs instead of all combinations
    Returns:
        topographical_similarity (float): correlation between similarity of pairs in both spaces
    """
    assert len(space_x) == len(space_y)

    N = len(space_x)

    # if no number of sample is passed
    # using all possible pair combinations in space
    if number_of_samples is None:
        combinations = list(itertools.combinations(range(N), 2))
    else:
        combinations = np.random.choice(
            np.arange(N), size=(number_of_samples, 2), replace=True
        )

    sim_x = np.zeros(len(combinations))
    sim_y = np.zeros(len(combinations))

    for i, c in enumerate(combinations):
        s1, s2 = c[0], c[1]

        sim_x[i] = distance_function_x(space_x[s1], space_x[s2])
        sim_y[i] = distance_function_y(space_y[s1], space_y[s2])

    # check if standard deviation is not 0
    if sim_x.std() == 0.0 or sim_y.std() == 0.0:
        warnings.warn("Standard deviation of a space is 0 given distance function")
        rho = 0.0
    else:
        rho = scipy.stats.pearsonr(sim_x, sim_y)[0]

    return rho


def one_hot(a, n_cols: Optional[int] = None):
    if n_cols is None or n_cols < a.max() + 1:
        n_cols = a.max() + 1
    out = np.zeros((a.size, n_cols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (n_cols,)
    return out


def flatten_cos(a, b) -> float:
    return spatial.distance.cosine(a.flatten(), b.flatten())


def on_hot_hamming(a, b):
    return spatial.distance.hamming(
        one_hot(a, n_cols=VOCAB).flatten(), one_hot(b, n_cols=VOCAB).flatten()
    )


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
    distance = np.zeros((rows,cols),dtype = int)

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
    "ham_messages" : on_hot_hamming,
    "lev_messages": levenshtein_ratio_and_distance
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

    #todo check actual vocab size
    VOCAB = 5


    # metric_files = glob.glob(f"/*/*")
    #
    # for file in tqdm(metric_files):
    #     m = pickle.load(open(file, "rb"))
    #     for (space_x, space_y) in combinations(list(DIST.keys()), 2):
    #         rsa_title = f"RSA:{space_x}/{space_y}"
    #         #if rsa_title not in m:
    #         r = rsa(m[space_x], m[space_y], DIST[space_x], DIST[space_y], number_of_samples=args.samples)
    #         m[rsa_title] = r
    #
    #     pickle.dump(m, open(file, "wb"))

    # Calculate Cross-Seed RSA for all
    seed_folders = glob.glob(f"*")

    RESULTS = {}

    for sp in tqdm(DIST):

        RESULTS[sp] = {}

        # compare every seed to all others
        for s1, s2 in combinations(seed_folders, 2):

            seed1 = s1.split("/")[-1]
            seed2 = s2.split("/")[-1]

            RESULTS[sp][seed1 + seed2] = {}

            # load messages from specific agent
            files_s1 = glob.glob(f"{s1}/*/*")

            # loop through all agents
            for a1 in files_s1:

                # retrieve current agent pair
                agent = a1.split("_")[2]

                # retrieve the string
                agent1 = a1.split("/")[1:]
                sym = '/'
                agent1 = sym.join(agent1)

                # get second file, they have the same name so just ask agent 1
                a2 = f"{s2}/{agent1}"

                if os.path.isfile(a2):

                    # load the different rsa analysis files
                    m1 = pickle.load(open(a1, "rb"))
                    m2 = pickle.load(open(a2, "rb"))

                    # use actual space for to extract data and sp to differentiate message similarity spaces
                    r = rsa(m1, m2, DIST[sp], DIST[sp], number_of_samples=args.samples)
                    RESULTS[sp][seed1 + seed2][agent] = r

    # save results
    pickle.dump(RESULTS, open(f"cross_rsa_analysis.pkl", "wb"))


if __name__ == "__main__":
    main(sys.argv[1:])
