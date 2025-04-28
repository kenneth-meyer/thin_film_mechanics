#!/bin/bash

#declare -a nu_list=(0.3 0.4 0.45 0.49)

for i in {0.3,0.4,0.45,0.49}
do
    python run_substrate_bvp_standalone.py $i
done