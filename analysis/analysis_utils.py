import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import glob
from collections import defaultdict
from analysis import *

def load_metrics(path):
    """
        This function loads metrics
    """

    # load the data
    seed_folders = glob.glob(f"{path}/*")

    # save results
    generalize_result = {}
    train_result = {}

    # run through all seed
    for s in seed_folders:

        # get seed index
        seed = s.split("/")[-1]

        # make sure to ignore the rsa analysis for now
        if seed == 'rsa_analysis.pkl':
            continue

        # get all metric files
        metric_files = glob.glob(s + "/*.pkl")

        for file in metric_files:

            # load files
            m1 = pickle.load(open(file, "rb"))

            # check if file is generalize or train metric
            if file.find('generalize') == -1:
                if file.find('10000') != -1:
                    train_result[seed] = m1
            else:
                generalize_result[seed] = m1

    return train_result, generalize_result


def show_messages(path, metrics, show_results=True):
    """
    This function shows the messages for different seeds of a certain experiment.

    :param path: decides for which files the messages must be printed
    :param metrics: checkpoints to show the message
    :param show_results: boolean to decide whether or not to print results
    :return:
    """

    # get the files to be analyzed
    metric_files = glob.glob(f"{path}/*/*.pkl")
    seed_folders = glob.glob(f"{path}/*")

    # load one file to find the amount of samples in the data
    m1 = pickle.load(open(metric_files[0], "rb"))
    nindex = len(m1['messages'])

    # generate 10 samples to be checked
    indices = np.random.choice(nindex, 10, replace=False)
    result = {}
    # run through all seed
    for s in seed_folders:

        # get seed index
        seed = s.split("/")[-1]

        if seed == 'rsa_analysis.pkl':
            continue

        result[seed] = defaultdict(list)

        # run through selected metric iterations
        for metric in metrics:

            # combine file path
            file_path = s + "/metrics_at_{}.pkl".format(metric)

            # load files
            m1 = pickle.load(open(file_path, "rb"))

            # extract selected messages
            messages = m1['messages'][indices]
            targets = m1['targets'][indices]

            # use targets as key to save messagess
            for i in range(len(messages)):
                result[seed][str(targets[i])].append(messages[i])

    # show the results if requested
    if show_results:
        # pretty print the results
        for tar, mess in result['1'].items():
            print('Messages for target:')
            print(tar)
            print()
            for i in range(len(mess)):
                if i == 0:
                    print('iteration | ', end='')
                    for seed in result.keys():
                        print("{:<13}".format('    seed ' + seed + ''), end="")
                        print(' | ', end='')
                    print()

                print("{:<9}".format(str(metrics[i])), end='')
                print(' | ', end="")
                for seed in result.keys():
                    print(result[seed][tar][i], end='')
                    print(' | ', end='')
                print()
            print()
    return


def unique_messages(path, metrics, show_results=True):
    """
    This function counts the amount of unique messages for a certain experiment
    :param path: the path to the files that need to be checked
    :param metrics: iterations at which to show results
    :param show_results: boolean to decide whether or not to print results
    :return: nr of unique messages
    """

    # load the data
    seed_folders = glob.glob(f"{path}/*")

    result = {}
    # run through all seed
    for s in seed_folders:

        # get seed index
        seed = s.split("/")[-1]

        if seed == 'rsa_analysis.pkl':
            continue

        result[seed] = {}

        # run through selected metric iterations
        for metric in metrics:
            # combine file path
            file_path = s + "/metrics_at_{}.pkl".format(metric)

            # load files
            m1 = pickle.load(open(file_path, "rb"))

            # loop through messages to gather unique ones
            um = set()
            for message in m1['messages']:
                um.add(tuple(message))

            # append result
            result[seed][metric] = len(um)

    if show_results:
        # show the amount of unique message
        for seed, metric in result.items():
            print('Showing results for seed ' + str(seed))
            print('Iteration: \t | \t Unique messages:')
            for i, count in metric.items():
                if i < 1000:
                    print('\t' + str(i) + '\t\t\t\t\t' + str(count))
                else:
                    print('\t' + str(i) + '\t\t\t\t' + str(count))

            print()

    return result


def unique_tokens(path, iterations, show_results=True):
    """
    This function counts the amount of unique messages for a certain experiment
    :param path: the path to the files that need to be checked
    :param iterations: iterations at which to show results
    :param show_results: boolean to decide whether or not to print results
    :return: nr of unique messages
    """

    # load the data
    seed_folders = glob.glob(f"{path}/*")

    result = {}
    result_tokens = {}
    # run through all seed
    for s in seed_folders:

        # get seed index
        seed = s.split("/")[-1]

        if seed == 'rsa_analysis.pkl':
            continue

        result[seed] = {}

        all_tokens =set()

        # run through selected metric iterations
        for it in iterations:
            # combine file path
            file_path = s + "/metrics_at_{}.pkl".format(it)

            # load files
            m1 = pickle.load(open(file_path, "rb"))

            # loop through messages to gather unique tokens
            tokens = set()
            for message in m1['messages']:
                for token in message:
                    tokens.add(token)
                    all_tokens.add(token)

            # append result
            result[seed][it] = len(tokens)

        result_tokens[seed] = all_tokens

    if show_results:
        # show the amount of unique message
        for seed, metric in result.items():
            print('Showing results for seed ' + str(seed))
            print('Iteration: \t | \t Unique tokens:')
            for i, count in metric.items():
                if i < 1000:
                    print('\t' + str(i) + '\t\t\t\t\t' + str(count))
                else:
                    print('\t' + str(i) + '\t\t\t\t' + str(count))

            print()

    return result
