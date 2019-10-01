#!/bin/bash

#echo "Computing RSA's"
#for attribute in {2..5}
#  do
#    python compute_rsas.py --attributes $attribute --samples 1000 --max-length 5 --vocab-size 5
#  done

python compute_rsas.py --samples 1000 --max-length 5 --vocab-size 5 --distractors 3
python compute_rsas.py --samples 1000 --max-length 5 --vocab-size 5 --distractors 7
python compute_rsas.py --samples 1000 --max-length 5 --vocab-size 5 --distractors 11
