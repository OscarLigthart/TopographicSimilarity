import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import glob
from collections import defaultdict

vocab = pickle.load(open('data/dict_5.pckl', "rb"))

# load interesting metrics
path = "runs/lstm_max_len_5_vocab_5_attr_5"

metric_files = glob.glob(f"{path}/*/*.pkl")
seed_folders = glob.glob(f"{path}/*")

# generate 10 samples to be checked
indices = np.random.choice(243, 10, replace=False)
print(indices)


# add metric file, set checkpoints
metrics = [0, 200, 400, 600, 800, 1000, 2800, 5000, 7200, 9800]


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

# retrieve messages for specific targets

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


