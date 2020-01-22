#!/bin/bash

echo "Computing RSAs"
python compute_rsas.py --samples 1000 --vocab-size 5 --max-length 5 --related --attributes 4 --split 2 --same-data

