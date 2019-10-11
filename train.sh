#!/bin/bash

# same data shuffles
echo "Running hard task"

for seed in {1..10} 
  do
    python main.py --seed $seed --attributes 6 --embedding-size 128 --hidden-size 128 --related True
  done

echo "Computing RSAs"
python compute_rsas.py --samples 1000 --attributes 6 --related True

# related with lower attributes
echo "Running baseline"
for seed in {1..10} 
  do
    python main.py --seed $seed --attributes 4 --related True
  done

echo "Computing RSAs"
python compute_rsas.py --samples 1000 --attributes 4 --related True



