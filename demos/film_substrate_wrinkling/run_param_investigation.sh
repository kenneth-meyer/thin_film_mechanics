#!/bin/bash

h=0.02
H=2
E_f=10000 # 10 GPa

h=0.0001    # 5.0 um
# H=1.0      # 1000 um
w=0.112
D=0.0561
E_s=2.97   # 10 MPa
nu_s=0.495
E_f=7300 # 7300 MPa
nu_f=0.35

python parameter_investigation.py $h $D $E_s $nu_s $E_f $nu_f

# for E_s in {0.1,1,10}
# do
#     for nu_s in {0.4,0.45,0.49}
#     do
#         for nu_f in {0.3,0.4}
#         do
#             python parameter_investigation.py $h $H $E_s $nu_s $E_f $nu_f
#         done
#     done
# done