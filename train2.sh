#!/bin/bash

# same data shuffles
echo "Running receiver freeze task"


for globseed in {1..3}
  do
    echo "Training model"

    # train new model
    python main.py --seed $globseed --split 2 --log-interval 1000

    echo "Freezing receiver and retraining sender..."

    # freeze sender models and retrain receiver
    for seed in {1..10} 
      do
        #python main.py --seed 4 --resume --freeze-sender --freeze-seed $seed --split 2 --log-interval 1000
        python main.py --seed $globseed --resume --freeze-receiver --freeze-seed $seed --split 2 --log-interval 1000
      done


    echo "Computing RSAs"
    #python compute_rsas.py --samples 2500 --split 2 --freeze-sender --freeze-seed 4
    python compute_rsas.py --samples 2500 --split 2 --freeze-receiver --freeze-seed $globseed
  done

