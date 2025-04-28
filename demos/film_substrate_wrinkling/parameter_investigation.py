"""
    Investigate the critical wavelength and membrane force
    required to wrinkle a film-substrate system with prescribed moduli
"""
import meshio
import os
import jax.numpy as np
import numpy as onp
import jax

from thin_film_src.film_substrate_helpers import FilmSubstrate

import time
import pickle
import sys
   
if __name__ == "__main__":
    # allow the user to vary poisson's ratio to examine effects of it changing
    try:
        h = float(sys.argv[1])
        H = float(sys.argv[2])
        E_s = float(sys.argv[3])
        nu_s = float(sys.argv[4])
        E_f = float(sys.argv[5])
        nu_f = float(sys.argv[6])
        
    except:
        print(sys.argv)
        raise Exception("Invalid input! ")
        # E_s = 10
        # nu_s = 0.4
        
    # name of the job we're running
    RUN_NAME ="substrate_bvp"
    data_dir_top = "/workspace/kmeyer/research/cardiax_testing/wrinkling/subrate_bvp"

    # determine the case of the problem to set up the geometry
    CASE = '2D'

    if CASE == '2D':
        Nx, Ny = 4, 2
        Lx, Ly = 1.0, H
        vec = 2
        dim = 2
    else:
        Nx, Ny, Nz = 40, 7, 40   
        Lx, Ly, Lz = 1.0, H, 1.0
        vec = 3
        dim = 3
    
    # h_f, H_s, E_s, nu_s, E_f, nu_f
    film_subrate_obj = FilmSubstrate(h, H, E_s, nu_s, E_f, nu_f)
    
    # compute various quantities that will be helpful for logging in overleaf/slides
    lmbda_c_thin = film_subrate_obj.get_wavelength_thin_limit()
    lmbda_c_thick = film_subrate_obj.get_wavelength_thick_limit()
    f_c_thin = film_subrate_obj.get_critical_force_thin_limit()
    f_c_thick = film_subrate_obj.get_critical_force_thick_limit()
    
    # save this to a .txt file (a log file would be better)
    with open("log_param_investigation.txt","a") as f:
        f.write(f"h: {h}, H:{H}, E_s:{E_s}, nu_s:{nu_s}, E_f:{E_f}, nu_f:{nu_f}\n")
        f.write("lmbda_c thick | f_c thick || lmbda_c thin | f_c thin\n")
        f.write(f"{lmbda_c_thick} | {f_c_thick} || {lmbda_c_thin} | {f_c_thin}\n")