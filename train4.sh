#!/bin/bash

# same data shuffles

for seed in {1..10} 
  do
    python generalize.py --seed $seed --split 2 --vocab-size 5 --max-length 5
    python generalize.py --seed $seed --same-data --split 2 --vocab-size 5 --max-length 5
  done

