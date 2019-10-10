#!/bin/bash

# baseline experiment
echo "Running baseline"
for seed in {1..10} 
  do
    python main.py --seed $seed --vocab-size 5 --max-length 5 --related True
  done

echo "Computing RSAs"
python compute_rsas.py --samples 1000 --vocab-size 5 --max-length 5 --related True





