import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import glob
from collections import defaultdict
from analysis import *

# load interesting metrics
path = "runs/lstm_max_len_5_vocab_5_attr_5_related"

# add metric file, set checkpoints
metrics = [0, 200, 400, 600, 800, 1000, 2800, 5000, 7200, 9800]

#show_messages(path, metrics)

um = unique_messages(path, metrics)









