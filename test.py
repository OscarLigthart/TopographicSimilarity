import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

vocab =pickle.load( open( "data/dict_3.pckl", "rb" ) )

print(vocab)