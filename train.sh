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
    #python main.py --seed $seed --split 2 --vocab-size 5 --max-length 5 --same-data #--related --attributes 4
    python generalize.py --split 2 --seed $seed --vocab-size 5 --max-length 5 #--same-data #--related --attributes 4
  done


#echo "Computing RSAs"
#python compute_rsas.py --samples 1000 --split 2 --vocab-size 3 --max-length 5
#python compute_generalize_rsa.py --split 2 --same-data --samples 1000



