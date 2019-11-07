#!/bin/bash

# same data shuffles
echo "Running split task"


for seed in {1..10} 
  do
    python main.py --seed $seed --split 2 --attributes 3 --log-interval 1000
  done

echo "Computing RSAs"
python compute_rsas.py --samples 1000 --split 2 --attributes 3

