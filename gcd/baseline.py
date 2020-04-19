# import modules
import argparse
import sys
import os
import torch
import pickle
import diagnnose

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '..')

from data import *
from receiver import *





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments(args):
    # Training settings
    parser = argparse.ArgumentParser(
        description="Trains a Sender/Receiver Agent in a referential game"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10001,
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
        default=128,
        metavar="N",
        help="hidden size for hidden layer (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
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
        default=20,
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
        "--log-interval",
        type=int,
        default=1000,
        metavar="N",
        help="number of iterations between logs (default: 1000)",
    )
    parser.add_argument(
        "--resume",
        help="Resume the training from the saved model state",
        action="store_true",
        default=False
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
        default=4
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
    parser.add_argument(
        "--freeze-sender",
        help="Freeze the sender in the setup, so the language stays constant",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--freeze-receiver",
        help="Freeze the receiver in the setup, so the language interpretation remains constant",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--freeze-seed", type=int, default=2, help="random seed (default: 1)"
    )

    args = parser.parse_args(args)

    return args


def main(args):
    """
    Runs the baseline experiment
    :param: args: experiment settings
    """

    # get settings
    args = parse_arguments(args)

    # set seed straight away
    seed_torch(seed=args.seed)

    # create datafile structure
    model_name = get_filename(args)
    create_folder_if_not_exists("runs")
    create_folder_if_not_exists("runs/" + model_name)

    # create dataset
    run_folder = "runs/" + model_name + "/" + str(args.seed)
    create_folder_if_not_exists(run_folder)
    model_path = run_folder + "/model.p"

    # determine the vocabulary size
    # TODO insert the size of the dataset
    vocab = AgentVocab(args.vocab_size)

    # get objects attribute early, to determine input size of models
    gen_attr = get_attributes(args.attributes, args.related)

    ####################
    # INITIALIZE MODEL #
    ##############################################################

    # initialize receiver
    receiver = Receiver(
        vocab,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        output_size=sum(gen_attr),
    )

    # initialize model
    model = ReceiverTrainer(receiver)

    # send model to device
    model.to(device)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ###################
    # DATA SET LOADER #
    #############################################################

    # set dataseed and dataset here if you wish to use same data
    if args.same_data:
        seed_torch()

    # initialize dataset
    train_data = get_referential_dataloader("shapes", gen_attr, k=args.distractors,
                                            batch_size=args.batch_size, shuffle=True,
                                            related=args.related, split=args.split, pair=args.pair)

    valid_data = get_referential_dataloader("shapes", gen_attr, k=args.distractors,
                                            batch_size=args.batch_size, related=args.related,
                                            split=args.split, pair=args.pair)

    ###############
    # TRAIN MODEL #
    ################################################################

    epoch, iteration = 0, 0
    while iteration < args.iterations:
        for (messages, targets, distractors) in train_data:

            # normal training
            train_one_batch(model, optimizer, messages, targets, distractors)

            # evaluate the model on log_interval
            if iteration % args.log_interval == 0:
                print(f"{iteration}/{args.iterations}\r")
                metrics = evaluate(model, valid_data)
                save_model_state(model, model_path, epoch, iteration)

                # normal dump
                pickle.dump(
                    metrics, open(run_folder + f"/metrics_at_{iteration}.pkl", "wb")
                )

                print(f"\t\t acc: {metrics['acc']:.3f}\r", end="")

            iteration += 1
            if iteration >= args.iterations:
                break

        epoch += 1

if __name__ == "__main__":
    main(sys.argv[1:])