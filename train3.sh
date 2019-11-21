#!/bin/bash

# same data shuffles
echo "Running split task"

for pair in {1..20}
  do
  for seed in {1..10} 
    do
      python main.py --seed $seed --split 2 --pair $pair --same-data True --log-interval 1000
      python generalize.py --split 2 --pair $pair --seed $seed --same-data True
    done
  done
