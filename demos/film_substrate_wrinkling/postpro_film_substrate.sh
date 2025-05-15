#!/bin/bash

# TESTING PARAMETERS
# h=0.2
# H=2
# E_s=2.97   # 10 MPa
# nu_s=0.495
# E_f=7300 # 7300 MPa
# nu_f=0.35

# E_s=10.0   # 10 MPa
# nu_s=0.4
# E_f=10000 # 10 GPa
# nu_f=0.45

# material parameters
h=0.0002    # 5.0 um
# H=1.0      # 1000 um
w=0.112
D=0.056
E_s=2.97   # 10 MPa
nu_s=0.495
E_f=7300 # 7300 MPa
nu_f=0.35

# can scale the initial membrane force (not used for dirichlet-based compression)
f_scale=1.005

# these change whenever the material params and geometry changes; better to compute on the fly.
# critical force corresponding to these gus ^
#N_c=0.245
# pre-computed initial membrane force N_0_11
#N_0_11=0.25

# running one at a time right now
python ../../runfiles/postpro_film_substrate_prestress.py $h $w $D $E_s $nu_s $E_f $nu_f $f_scale

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