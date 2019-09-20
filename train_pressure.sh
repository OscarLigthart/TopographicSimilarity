#!/bin/bash

echo "Running..."
for attr in {2..5}
  do
    for seed in {1..10}
      do
        python main.py --seed $seed --attributes $attr --vocab-size 5 --max-length 5
      done
    
    echo "Computing RSAs"
    python compute_rsas.py --attributes $attr --samples 1000
  done
                     
