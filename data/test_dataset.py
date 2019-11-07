from data import *
import pickle

x = pickle.load(open('generalize_split_2_attr_5.p', 'rb'))

for value in x.values():
    print(value)
