# same data shuffles
echo "Running hard task"

for seed in {1..10} 
  do
    python main.py --seed $seed --attributes 5 --related True --hidden-size 16
  done

