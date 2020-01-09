import argparse
import sys
import os
import torch
import pickle
import re

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler

from utils import *
from data import *
from models import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments(args):
    # Training settings
    parser = argparse.ArgumentParser(
        description="Trains a Sender/Receiver Agent in a referential game"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        metavar="N",
        help="number of batch iterations to train (default: 10k)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=64,
        metavar="N",
        help="embedding size for embedding layer (default: 64)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        metavar="N",
        help="hidden size for hidden layer (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        metavar="N",
        help="input batch size for training (default: 32)",
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
        "--lr",
        type=float,
        default=1e-3,
        metavar="N",
        help="Adam learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--same-data",
        help="decide whether same seed should be used to shuffle data",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--attributes",
        help="Decide on the amount of attributes to use",
        type=int,
        default=5
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
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--split",
        help="Decide whether to use all generated samples or keep some apart for testing generalization,"
             " value decides the amount of cooccurences to be removed",
        type=int,
        default=0
    )
    parser.add_argument(
        "--pair",
        help="Decide on which attribute pair to split on, (currently integer)",
        type=int,
        default=1
    )

    args = parser.parse_args(args)
    return args


def main(args):
    args = parse_arguments(args)

    # use the same seed for the data collection
    seed_torch()

    model_name = get_filename(args)

    create_folder_if_not_exists("runs")
    create_folder_if_not_exists("runs/" + model_name)

    run_folder = "runs/" + model_name + "/" + str(args.seed)
    create_folder_if_not_exists(run_folder)

    model_path = run_folder + "/model.p"

    # remove related
    #model_path = re.sub('_related', '', model_path)

    vocab = AgentVocab(args.vocab_size)

    # get objects attribute early, to determine input size of models
    gen_attr = get_attributes(args.attributes, args.related)

    # get sender and receiver models
    sender = Sender(
        vocab,
        args.max_length,
        input_size=sum(gen_attr),
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        greedy=True,
    )

    receiver = Receiver(
        vocab,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        output_size=sum(gen_attr),
    )

    # initialize trainer
    model = ReferentialTrainer(sender, receiver)

    # load model
    epoch, iteration = load_model_state(model, model_path)
    print(f"Loaded model. Resuming from - epoch: {epoch} | iteration: {iteration}")

    # get the data
    print("Generating dataset...")

    # create the dataset
    data = generate_dataset(gen_attr, 0)

    # load regular dataset, replace the targets with the saved split

    # check whether we need distractor samples
    if args.related:
        samples = get_close_samples(data, one_attribute=True)
    else:
        samples = None

    # create dataloader
    dataset = ReferentialDataset(data)

    valid_data = DataLoader(
        dataset,
        pin_memory=True,
        batch_sampler=BatchSampler(
            ReferentialSampler(dataset, samples, related=args.related, k=args.distractors, shuffle=False,
                               split=args.split, attr=gen_attr, pair=args.pair),
            batch_size=args.batch_size,
            drop_last=False,
        ),
    )

    # evaluate the model
    metrics = evaluate(model, valid_data)

    print(metrics['acc'])

    # save the metrics
    # pickle.dump(
    #     metrics, open(run_folder + f"/generalize_metrics.pkl", "wb")
    # )


if __name__ == "__main__":
    main(sys.argv[1:])
