"""
This script is made to run an input file.
Call (python run.py input_file.yaml)
"""

import sys
import yaml
import meshio
from pathlib import Path
import jax.numpy as np
import numpy as onp

from cardiax.input_file_handler import FE_Handler

try:
    input_file = sys.argv[1]
except:
    input_file = "/home/bthomas/Desktop/Research/NNFE_code/CARDIAX/demos/input_file/inputs.yaml"

fe_handler = FE_Handler(input_file)
sol, info = fe_handler.solver.solve(**fe_handler.solver_params["solver_params"])

