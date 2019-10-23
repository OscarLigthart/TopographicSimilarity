# same data shuffles
echo "Running generalization"

for seed in {1..10} 
  do
    echo "Training model"
    python main.py --seed $seed --split True --log-interval 9800
    echo "Generalizing"
    python generalize.py --seed $seed
  done

echo "Computing RSAs"
python compute_rsas.py --samples 1000 --attributes 5 --split True


