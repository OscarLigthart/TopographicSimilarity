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
    python main.py --seed $seed --split 2 --vocab-size 3 --max-length 3
    python generalize.py --split 2 --seed $seed --vocab-size 3 --max-length 3
  done



echo "Computing RSAs"
python compute_rsas.py --samples 1000 --split 2 --vocab-size 3 --max-length 3
#python compute_generalize_rsa.py --split 2 --same-data --samples 1000



