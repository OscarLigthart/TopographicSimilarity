"""
Calculates all the RSA for all run files
"""
import glob
import pickle
import os
import sys
import random
import argparse
from itertools import combinations
from scipy import spatial
from metrics import rsa
from data import one_hot
from tqdm import tqdm
from utils import *
from data import *
from models import *
from metrics import *
from main import parse_arguments


VOCAB = 28  # Vocab size + 3 special case tokens (eos, sos, pad)

def main(args):
    """
    This function calculates Cross-Seed SVCCA
    :param args: arguments determining for which experiment SVCCA needs to be calculated
    :return: Calculated SVCCA values
    """

    # retrieve arguments
    args = parse_arguments(args)

    vocab = AgentVocab(args.vocab_size)
    gen_attr = get_attributes(args.attributes)

    # load all models
    all_models = load_trained_models(args, vocab, gen_attr)
    print(len(all_models))
    #random.shuffle(all_models)

    # create the dataloader
    valid_data = get_referential_dataloader("shapes", gen_attr)

    # during this batch, load all models and get their hidden states on a batch
    for i, model in enumerate(all_models):

        # get all hidden states
        metrics = evaluate(model, valid_data)

        if i == 0:
            # get first batch of primary hidden sender (the one to which we compare everything)
            pr_h_sender = metrics['h_sender'][:args.batch_size]
        else:
            h_sender = metrics['h_sender'][:args.batch_size]

            # apply svcca to both hidden states
            output = get_cca_similarity(pr_h_sender, h_sender, threshold=.98, epsilon=1e-6, compute_dirns=True,
                                        verbose=False)

            print(output['mean'])
            #print(output['x_idxs'])
            #print(output['y_idxs'])

    # for each model, run it and extract hidden state

    # calculate

if __name__ == "__main__":
    main(sys.argv[1:])









