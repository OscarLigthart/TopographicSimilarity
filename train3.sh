#!/bin/bash

# same data shuffles
echo "Running data split task"

for seed in {1..10} 
  do
    python main.py --seed $seed --split 2 --vocab-size 3 --max-length 4
    python generalize.py --seed $seed --split 2 --vocab-size 3 --max-length 4
  done


echo "Computing RSAs"
python compute_rsas.py --samples 1000 --split 2 --vocab-size 3 --max-length 4
