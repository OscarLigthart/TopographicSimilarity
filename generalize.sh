#!/bin/bash

#for split in {1..3}
#  do
#    for seed in {1..10} 
#      do
#        python generalize.py --seed $seed --split $split
#      done
#  done

for seed in {1..10} 
  do
    python generalize.py --seed $seed --split 2 --attributes 3
  done


