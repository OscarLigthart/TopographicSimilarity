#!/bin/bash

# same data shuffles
echo "Running hard task"

for seed in {1..10} 
  do
    python main.py --seed $seed --attributes 5 --split True
  done

echo "Computing RSAs"
python compute_rsas.py --samples 1000 --attributes 5 --split True

# related with lower attributes
#echo "Running baseline"
#for seed in {1..10} 
#  do
#    python main.py --seed $seed --attributes 4 --embedding-size 128 --hidden-size 128
#  done

#echo "Computing RSAs"
#python compute_rsas.py --samples 1000 --attributes 4 --embedding-size 128 --hidden-size 128



