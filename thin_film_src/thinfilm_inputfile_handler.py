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
from cardiax.generate_mesh import box_mesh, get_ele_type, rectangle_mesh
import sys

from thin_film_src.material_models import (
    LinearElastic,
    LinearElastic_Traction
)

def tag_mesh(mesh, space, Lx, Ly, Lz=None):
    """
        Tags a mesh depending the space we're looking in

        Need to append the tags as it is possible for a single point to have
        multiple tags...

        Assumes Lx and other 
    """

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)
    # x == Lx
    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)
    
    # y == 0
    def front(point):
        return np.isclose(point[1], 0., atol=1e-5)
    # y == Ly
    def back(point):
        return np.isclose(point[1], Ly, atol=1e-5)

    bc_list = [left, right, front, back]

    # only define these if we're working in 3D
    if space == '3D':
        # z == 0
        def bottom(point):
            return np.isclose(point[2], 0.0, atol=1e-5)
        # z == Lz
        def top(point):
            return np.isclose(point[2], Lz, atol=1e-5)
        
        bc_list.append([bottom, top])

    # need to vmap to do this well
    def check_point(point, bc_fun):
        return bc_fun(point)

    check_point_vmap = jax.vmap(check_point, in_axes=(0,None))

    for bc_fn in bc_list:
        # list of booleans
        idx = check_point_vmap(mesh.points, bc_fn)
        # assign the name of the function to the point
        # not sure if there's a way to do this without using a for loop
        mesh.point_data["tags"]

        # need to complete this; I might be abandoning this for today to get a result quickly


# need to partial this function
def substrate_sinusoid(component, params, point):
    # params[0] : A
    # params[1] : k
    return params[0] * np.cos(params[1] * point[component])

def get_bc_type(bc):
    # only supports a bc being applied in 1 direction
    # (this is technically all that is required for the sinusoid problem)
    bc_type = bc['value'][0]
    params = bc['params']
    component = bc['component'][0]
    if bc_type == "substrate_sinusoid":
        my_bc_fun = fctls.partial(substrate_sinusoid, component, *params)

    return my_bc_fun

def create_2d_mesh(Nx, Ny, Lx, Ly, deg):
    # note: only linear elements can be used in this case as this is a custom
    #       mesh gen function from JAX-FEM
    ele_type = 'quad'
    mesh = rectangle_mesh(Nx, Ny, Lx, Ly)
    return mesh, ele_type

def create_3d_mesh(Nx, Ny, Nz, Lx, Ly, Lz, deg, data_dir):

    ele_type = get_ele_type(deg, 'hex')
    mesh = box_mesh(Nx,Ny,Nz,Lx=Lx,Ly=Ly,Lz=Lz,data_dir=data_dir,ele_type=ele_type)
    return mesh, ele_type

def get_problem(problem_type):
    """
        Returns a problem instance of a specified type
    """

    if problem_type == "LinearElastic":
        problem = LinearElastic
    elif problem_type == "LinearElastic_Traction":
        problem = LinearElastic_Traction
    else:
        raise Exception("The material model you requested was not found! oops")

    return problem

class Film_FE_Handler():

    def __init__(self, input_file):
        with open(input_file) as f:
            params = yaml.safe_load(f)

        self.parent = Path(params['directory'])
        
        # get all the required parameters
        self.mesh_params = params['mesh_info']
        self.fe_params = params['fe_info']
        self.pde_params = params['pde_info']
        self.solver_params = params['solver_info']
        self.bc_params = params['bc_info']
        self.plot_params = params["plot_info"]

        # load the appropriate mesh, FE, Problem, and Solver objects
        self.mesh_loader()
        self.fe_loader()
        self.bc_loader()
        self.problem_loader()
        self.solver_loader()
        return
    
    def get_value_fns(self,bc):

        # function to handle creating the value functions
        def value_fn(value, point):
            return np.array(value)
        
        value_fns = []
       
        # if the bc is a string, load the boundary condition from a predefined function
        if type(bc["value"][0]) == type("a"):
            # for now, only allows for bc to be specified in 1 direction
            my_value_fn = get_bc_type(bc)
            value_fns.append(my_value_fn)
        else:
            for i in range(len(bc["value"])):
                part_value_fn = fctls.partial(value_fn, bc["value"][i])
                value_fns.append(part_value_fn)
        
        return value_fns

    def mesh_loader(self):
        """
            Loads (generates) an FE mesh for a specific problem

            Saves mesh as self.mesh    
        """

        # call functions required to set up a FE object...
        space = self.fe_params["mesh_params"]["space"]
        if space == '2D':
            self.mesh = create_2d_mesh(**self.fe_params["mesh_params"])
        elif space == '3D':
            self.mesh = create_3d_mesh(**self.fe_params["mesh_params"])
        else:
            raise NotImplementedError("Only '2D' and '3D' cases are supported.")

        # tag each surface per existing conventions.
        self.mesh = tag_mesh(self.mesh, space)

    def fe_loader(self):
        #mesh = meshio.read((self.parent / self.fe_params['mesh_path']).resolve())
        #mesh.points = mesh.points.astype(np.float64)

        if self.fe_params["framework"] == "FEM":
            from cardiax.fe import FiniteElement
            self.fe_params["fe_params"]["dim"] = self.mesh.points.shape[-1]
            self.fe = FiniteElement(self.mesh, **self.fe_params["fe_params"])

        elif self.fe_params["framework"] == "IGA":

            pass

        else:
            raise ValueError("Framework not supported. Please choose between FEM and IGA.")
        return
    
    def bc_loader(self):
        # Generic function to check if a point is on a surface
        def surf_tag(tagged_nodes, point):
            return np.isclose(point, tagged_nodes, atol=1e-6).all(axis=1).any()

        self.dirichlet_bc_info = []
        self.location_fns = []
        self.surface_kernels = []
        for bc_num in self.bc_params:
            bc = self.bc_params[bc_num]
            if bc["type"] == "Dirichlet":
                assert len(bc["component"]) == len(bc["value"])
                tagged_nodes = self.fe.mesh.points[self.fe.mesh.point_data[bc["surface_tag"]].astype(bool)].astype(np.float64)

                dirichlet_tag = fctls.partial(surf_tag, tagged_nodes)

                # use a function to allow for abstraction of how we generate
                # dirichlet boundary conditions
                value_fns = self.get_value_fns(bc)

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
            
            Problem_class = get_problem(self.pde_params["material_model"])
            
            # if self.pde_params["material_model"] == "NeoHookean":
            #     from cardiax.material_models.NeoHookean import Hyperelasticity
            # # elif pde_params["material_model"] == "MooneyRivlin":
            # #     from cardiax.material_models.MooneyRivlin import Hyperelasticity
            # elif 
            # else:
            #     raise ValueError("Material model not supported. Please choose between NeoHookean,...")

            # Problem_class = Hyperelasticity
            
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
