"""
    Sets up and runs a boundary value problem (BVP) for a substrate-only system.

    This is going to be used to test my ability to deform the surface of a substrate in a sinusoidal manner,
    my main concern is trying to determine what the element size needs to be to capture wrinkling (if it even matters)

    This also serves as a good first example of a sample workflow that uses CARDIAX.

    NOTE: `cardiax.input_file_handler.py` might need to be updated as:
        1. I don't think it's fully functional
        2. I'm using non-cardiax based classes that might be better off staying outside of cardiax for now
"""


import sys
import yaml
import meshio
from pathlib import Path
import jax.numpy as np
import numpy as onp

from thin_film_src.thinfilm_inputfile_handler import Film_FE_Handler

try:
    input_file = sys.argv[1]
except:
    input_file = "/workspace/kmeyer/school/thin_film_mechanics/demos/substrate_bvp/inputs.yaml"

fe_handler = Film_FE_Handler(input_file)

# solves a problem; I need to define boundary conditions appropriately.
sol, info = fe_handler.solver.solve(**fe_handler.solver_params["solver_params"])
