#!/bin/bash

# same data shuffles
echo "Running referential game"

for seed in {1..10} 
  do
    #python main.py --seed $seed --split 2 --vocab-size 5 --max-length 5 --same-data --related --attributes 4 --hidden-size 128
    python generalize.py --split 2 --seed $seed --vocab-size 5 --max-length 5 --same-data --related --attributes 4 --hidden-size 128
    python generalize.py --split 2 --seed $seed --vocab-size 5 --max-length 5 --same-data --attributes 4 --hidden-size 128
  done

#for seed in {1..10}
#  do
#    python main.py --seed $seed --vocab-size 5 --max-length 5 --same-data --related --attributes 4 --hidden-size 128 
#  done

#echo "Computing RSAs"
#python compute_rsas.py --samples 2500 --vocab-size 5 --max-length 5 --related --attributes 4 --same-data --split 2
#python compute_rsas.py --samples 2500 --vocab-size 5 --max-length 5 --attributes 4 --same-data --split 2
#python compute_generalize_rsa.py --split 2 --same-data --samples 1000



