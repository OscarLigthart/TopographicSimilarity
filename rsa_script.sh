#!/bin/bash

echo "Computing RSA's"
for attribute in {2..5}
  do
    python compute_rsas.py --attributes $attribute --samples 1000
  done

python compute_rsas.py --same-date True --samples 1000
