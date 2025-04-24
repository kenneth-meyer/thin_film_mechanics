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

from cardiax.problem import Problem
from cardiax.solver import Newton_Solver
from cardiax.input_file_helper import *
import sys

class FE_Handler():

    def __init__(self, input_file):
        with open(input_file) as f:
            params = yaml.safe_load(f)

        self.parent = Path(params['directory'])
        
        self.fe_params = params['fe_info']
        self.pde_params = params['pde_info']
        self.solver_params = params['solver_info']
        self.bc_params = params['bc_info']
        self.plot_params = params["plot_info"]

        self.fe_loader()
        self.bc_loader()
        self.problem_loader()
        self.solver_loader()
        return

    def fe_loader(self):
        mesh = meshio.read((self.parent / self.fe_params['mesh_path']).resolve())
        mesh.points = mesh.points.astype(np.float64)

        if self.fe_params["framework"] == "FEM":
            from cardiax.fe import FiniteElement
            self.fe_params["fe_params"]["dim"] = mesh.points.shape[-1]
            self.fe = FiniteElement(mesh, **self.fe_params["fe_params"])

        elif self.fe_params["framework"] == "IGA":

            pass

        else:
            raise ValueError("Framework not supported. Please choose between FEM and IGA.")
        return
    
    def bc_loader(self):
        # Generic function to check if a point is on a surface
        def surf_tag(tagged_nodes, point):
            return np.isclose(point, tagged_nodes, atol=1e-6).all(axis=1).any()

        def value_fn(value, point):
            return np.array(value)

        self.dirichlet_bc_info = []
        self.location_fns = []
        self.surface_kernels = []
        for bc_num in self.bc_params:
            bc = self.bc_params[bc_num]
            if bc["type"] == "Dirichlet":
                assert len(bc["component"]) == len(bc["value"])
                tagged_nodes = self.fe.mesh.points[self.fe.mesh.point_data[bc["surface_tag"]].astype(bool)].astype(np.float64)

                dirichlet_tag = fctls.partial(surf_tag, tagged_nodes)

                value_fns = []
                for i in range(len(bc["value"])):
                    part_value_fn = fctls.partial(value_fn, bc["value"][i])
                    value_fns.append(part_value_fn)

                bc_temp = [[dirichlet_tag]*len(bc["value"]), bc["component"], value_fns]
                self.dirichlet_bc_info.append(bc_temp)
                continue

            elif bc["type"] == "Neumann":
                assert len(bc["value"]) == self.fe.dim
                tagged_nodes = self.fe.mesh.points[self.fe.mesh.point_data[bc["surface_tag"]].astype(bool)].astype(np.float64)

                neumann_tag = fctls.partial(surf_tag, tagged_nodes)

                self.location_fns.append(neumann_tag)
                self.surface_kernels.append({"type": "Neumann", "value": bc["value"]})
                continue

            elif bc["type"] == "Robin":
                continue
            elif bc["type"] == "Spring":
                continue
            elif bc["type"] == "Pressure":
                continue
            else:
                raise ValueError("BC type not supported. Please choose between Dirichlet, Neumann, Robin, Spring, and Pressure.")
        return

    def problem_loader(self):
        if self.pde_params["pde_type"] == "Hyperelasticity":
            if self.pde_params["material_model"] == "NeoHookean":
                from cardiax.material_models.NeoHookean import Hyperelasticity
            # elif pde_params["material_model"] == "MooneyRivlin":
            #     from cardiax.material_models.MooneyRivlin import Hyperelasticity
            else:
                raise ValueError("Material model not supported. Please choose between NeoHookean,...")

            Problem_class = Hyperelasticity
            pde_constants = self.pde_params["material_constants"]

        elif self.pde_params["pde_type"] == "Laplacian":
            # Problem_class = Laplacian
            pass
        elif self.pde_params["pde_type"] == "Poisson":
            # Problem_class = Poisson
            pass

        else:
            raise ValueError("PDE type not supported. Please choose between Hyperelasticity, Laplacian, and Poisson.")

        Problem_class = add_surface_kernels(Problem_class, self.surface_kernels)

        self.problem = Problem_class(self.fe, dirichlet_bc_info=[self.dirichlet_bc_info], location_fns=[self.location_fns])
        self.problem.set_params(pde_constants)
        return
    
    def solver_loader(self):
        if self.solver_params["solver_type"] == "Newton":
            self.solver = Newton_Solver(self.problem, np.zeros(self.problem.num_total_dofs_all_vars))
        return

    def plotting(self, sol):
        if self.plot_params["plot"]:
            from cardiax.plotting import visualize_sol
            visualize_sol(self.fe, sol, self.parent / self.plot_params["filename"])
        return
