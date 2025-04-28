#!/bin/bash

h=0.02
H=2
E_f=10000 # 10 GPa

for E_s in {0.1,1,10}
do
    for nu_s in {0.4,0.45,0.49}
    do
        for nu_f in {0.3,0.4}
        do
            python parameter_investigation.py $h $H $E_s $nu_s $E_f $nu_f
        done
    done
done