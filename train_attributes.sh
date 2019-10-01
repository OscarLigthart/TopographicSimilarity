#!/bin/bash

# lower amount of attributes
echo "Running different attributes"
for attribute in {2..4}
  do
    for seed in {1..10} 
      do
        python main.py --seed $seed --attributes $attribute
      done

    echo "Computing RSAs"
    python compute_rsas.py --samples 1000 --attributes $attribute
  done


# different amount of attributes
echo "Running different attributes"
for attribute in {6..8}
  do
    for seed in {1..10} 
      do
        python main.py --seed $seed --attributes $attribute
      done

    echo "Computing RSAs"
    python compute_rsas.py --samples 1000 --attributes $attribute
  done

