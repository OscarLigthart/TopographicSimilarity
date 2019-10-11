#!/bin/bash

# related with lower vocabulary
#echo "Running baseline"
#for seed in {1..10} 
#  do
#    python main.py --seed $seed --vocab-size 5 --max-length 5 --related True
#  done

#echo "Computing RSAs"
#python compute_rsas.py --samples 1000 --vocab-size 5 --max-length 5 --related True

# related with baseline vocabulary
#echo "Running baseline"
#for seed in {1..10} 
#  do
#    python main.py --seed $seed --related True
#  done

#echo "Computing RSAs"
#python compute_rsas.py --samples 1000 --related True

# related with higher attributes
#echo "Running baseline"
#for seed in {1..10} 
#  do
#    python main.py --seed $seed --attributes 6 --related True
#  done

#echo "Computing RSAs"
#python compute_rsas.py --samples 1000 --attributes 6 --related True

# related with lower attributes
echo "Running baseline"
for seed in {1..10} 
  do
    python main.py --seed $seed --attributes 4 --related True
  done

echo "Computing RSAs"
python compute_rsas.py --samples 1000 --attributes 4 --related True




