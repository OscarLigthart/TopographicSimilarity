#!/bin/bash

echo "Running..."
for attr in {2..5}
  do
    for seed in {1..10}
      do
        python main.py --seed $seed --attributes $attr --vocab-size 50 --max-length 15
      done
    
    echo "Computing RSAs"
    python compute_rsas.py --attributes $attr --samples 1000 --vocab-size 50 --max-length 15
  done
                     
