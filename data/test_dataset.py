from data import *
import pickle

x = pickle.load(open('splits/split_2_attr_5_pair_1.p', 'rb'))

for value in x.values():
    print(value)
