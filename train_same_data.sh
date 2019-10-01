#!/bin/bash

# same data shuffles
echo "Running same data"
for seed in {1..10} 
  do
    python main.py --seed $seed --same-data True
  done

echo "Computing RSAs"
python compute_rsas.py --samples 1000 --same-data True
