#!/bin/bash

echo "Running referential game"

#for seed in {1..10} 
#  do
#    python3 main.py --seed $seed --vocab-size 5 --max-length 5 --same-data --attributes 4 --hidden-size 256
#    python3 main.py --seed $seed --vocab-size 5 --max-length 5 --attributes 4 --hidden-size 256
#  done

echo "Computing RSAs"
python3 compute_rsas.py --samples 2500 --vocab-size 5 --max-length 5 --attributes 4 --same-data
python3 compute_rsas.py --samples 2500 --vocab-size 5 --max-length 5 --attributes 4


