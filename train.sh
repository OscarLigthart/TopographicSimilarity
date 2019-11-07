#!/bin/bash

# same data shuffles
echo "Running split task"

#for split in {1..3}
#  do
#    for seed in {1..10} 
#      do
#        python main.py --seed $seed --split $split --log-interval 1000
#      done

#    echo "Computing RSAs"
#    python compute_rsas.py --samples 1000 --split $split
#  done

for seed in {1..10} 
  do
    python main.py --seed $seed --split 2 --log-interval 1000
    python generalize.py --split 2 --seed $seed
  done



#echo "Computing RSAs"
#python compute_rsas.py --split 2 --same-data True --samples 1000



