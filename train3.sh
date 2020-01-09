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
    python main.py --seed $seed --vocab-size 5 --max-length 5 --log-interval 200 --same-data
    #python generalize.py --split 2 --seed $seed --same-data --vocab-size 5 --max-length 5
  done



echo "Computing RSAs"
python compute_rsas.py --samples 1000 --vocab-size 5 --max-length 5 --same-data
#python compute_generalize_rsa.py --split 2 --same-data --samples 1000

