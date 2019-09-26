#!/bin/bash

echo "Running baseline"
for seed in {1..10} 
  do
    python main.py --seed $seed --vocab-size 5 --max-length 5
  done

echo "Computing RSAs"
python compute_rsas.py --vocab-size 5 --max-length 5 --samples 1000
