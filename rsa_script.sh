#!/bin/bash

#echo "Computing RSA's"
#for attribute in {2..5}
#  do
#    python compute_rsas.py --attributes $attribute --samples 1000 --max-length 5 --vocab-size 5
#  done

echo "Computing RSAs"
python compute_rsas.py --samples 1000 --vocab-size 3 --max-length 5
 
python compute_rsas.py --samples 1000 --vocab-size 5 --max-length 5

python compute_rsas.py --samples 1000 --vocab-size 5 --max-length 5 --related True

python compute_rsas.py --samples 1000 --vocab-size 6 --max-length 5

python compute_rsas.py --samples 1000 --vocab-size 25 --max-length 10 --attributes 2

python compute_rsas.py --samples 1000 --vocab-size 25 --max-length 10 --attributes 3

python compute_rsas.py --samples 1000 --vocab-size 25 --max-length 10 --attributes 4

python compute_rsas.py --samples 1000 --vocab-size 25 --max-length 10

python compute_rsas.py --samples 1000 --vocab-size 25 --max-length 10 --attributes 6

python compute_rsas.py --samples 1000 --vocab-size 25 --max-length 10 --attributes 7

python compute_rsas.py --samples 1000 --vocab-size 25 --max-length 10 --attributes 8

python compute_rsas.py --samples 1000 --same-data True

python compute_rsas.py --samples 1000 --vocab-size 50 --max-length 10



