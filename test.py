import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

RSA_2_ATTR = pickle.load(open(f'runs/lstm_h_64_lr_0.001_max_len_10_vocab_25_attr_2/rsa_analysis.pkl', 'rb'))
RSA_SAME_DATA = pickle.load(open(f'runs/lstm_h_64_lr_0.001_max_len_10_vocab_25_same_data_attr_5/rsa_analysis.pkl', 'rb'))

for key,value in RSA_SAME_DATA.items():

    for key2,value2 in value.items():
        print(key2)
        print(value2)


# for key,value in RSA_2_ATTR.items():
#     print(key)
#     for key2,value2 in value.items():
#         print(key2)
#         print(value2)
#         break

print(RSA_2_ATTR)
