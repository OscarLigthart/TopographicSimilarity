#!/bin/bash

# same data shuffles
echo "Running low hidden space task"


for seed in {1..10} 
  do
    python main.py --hidden-size 16 --seed $seed --log-interval 1000 --same-data
  done



echo "Computing RSAs"
python compute_rsas.py --hidden-size 16 --same-data --samples 1000
#python compute_generalize_rsa.py --split 2 --same-data --samples 1000
