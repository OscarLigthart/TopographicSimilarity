#!/bin/bash

# same data shuffles
echo "Running same data"

array=( 4 6 )
for i in "${array[@]}"
  do
    for seed in {1..10} 
      do
        python main.py --seed $seed --attributes $i --same-data True
      done

    echo "Computing RSAs"
    python compute_rsas.py --samples 1000 --attributes $i --same-data True
  done
