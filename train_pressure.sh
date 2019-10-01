#!/bin/bash

# higher pressure runs
echo "Running higher pressure"
# create vocab array
array=( 3 5 )
for i in "${array[@]}"
  do
    for seed in {1..10} 
      do
        python main.py --seed $seed --vocab-size $i --max-length 5
      done

    echo "Computing RSAs"
    python compute_rsas.py --samples 1000 --vocab-size $i --max-length 5
  done



# lower pressure runs
echo "Running lower pressure"
for seed in {1..10} 
  do
    python main.py --seed $seed --vocab-size 50
  done

echo "Computing RSAs"
python compute_rsas.py --samples 1000 --vocab-size 50

