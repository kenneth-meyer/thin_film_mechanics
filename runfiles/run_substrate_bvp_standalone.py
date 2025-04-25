# Kenneth Meyer
# 4/13/25
# Hertzian contact, of a rigid indenter and an elastic half-plane.
# the half-plane is being modeled as a relatively large and refined rectangular prism.
#
# adaptive mesh refinement, or a mesh that is not of uniform refinement, could be very useful here.

import meshio
import os
import jax.numpy as np
import numpy as onp
import jax

# will convert the unittest code that generates geometries in one shot (without)
from cardiax.fe import FiniteElement
from cardiax.iga import BSpline
from cardiax.generate_mesh import generate_any_mesh, get_ele_type
from cardiax.problem import Problem
from cardiax.solver import Newton_Solver
from cardiax.utils import save_sol

from thin_film_src.thinfilm_inputfile_handler import create_2d_mesh, create_3d_mesh
from thin_film_src.material_models import LinearElastic

import time
import pickle

def get_A():
    raise NotImplementedError

def get_k():
    raise NotImplementedError

# executable portion of the file
if __name__ == "__main__":
    
    # name of the job we're running
    RUN_NAME ="substrate_bvp"
    data_dir_top = "/workspace/kmeyer/research/cardiax_testing/wrinkling/subrate_bvp"

    # determine the case of the problem to set up the geometry
    CASE = '2D'

    if CASE == '2D':
        Nx, Ny = 40,40
        Lx, Ly = 1.0,1.0
        vec = 2
        dim = 2
    else:
        Nx, Ny, Nz = 40, 7, 40   
        Lx, Ly, Lz = 1.0, 0.2, 1.0
        vec = 3
        dim = 3
    
    # GMSH mesh was not behaving well.
    # load a mesh generated from GMSH - is a .msh file I think
    # mesh_file = "stacked_mesh_no_refinement.msh"
    
    # material properties for the substrate
    E_s = 10.
    nu_s = 0.33
    mu_s = E_s/2/(1+nu_s)
    lmbda_s = E_s*nu_s/(1+nu_s)/(1-2*nu_s)

    # material properties for the thin film
    E_f = 10. * 10**4
    nu_f = 0.33
    mu_f = E_f/2/(1+nu_f)
    lmbda_f = E_f*nu_f/(1+nu_f)/(1-2*nu_f)

    # determine the amplitude and wavenumber from the 
    # material and geometry

    # need to define these functions
    A = get_A()
    k = get_k()

    # decide if we want to adaptively step - indenter height is what is being stepped.
    ADAPTIVE_LOAD_STEP = False

    # model params (see load stepper below)
    f_0 = 0.1
    # R = 0.5
    # k_pen_list = [1]
    # h_0 = 0.02

    # film/substrate params (geometric)
    h_s = 0.01

    # load stepping parameters - need to reset after each discretization/problem, hence the use of 'init' variables    
    delta_f_init = 0.1 # needs to decrease!
    delta_f = delta_f_init
    f_step_init = f_0                # initialize the load step
    f_step = f_step_init
    # f_max will be fixed
    f_max = 1
    
    # # define problem geometry - hopefully there are no bugs with 2D...
    # # (try compression first to check maybe?)
    # # keeping this in 3D so I can more easily assess where potential bugs are occuring...
    # # FOR TESTING
    # DEBUG = False
    # if DEBUG:
    #     Nx, Ny, Nz = 1, 7, 1
    # else:
    #     Nx, Ny, Nz = 40, 7, 40   


    # need to make sure we can resolve the film vs. the substrate

    # might want to find a way to define the center of the indenter to be at 0.5,0.5


    # define the types of problems we want to run
    problem_list = ['hex']
    degree_list = [1]
    refinements = [1] # multiplies number of knot spans in each direction

    # dictionary to save data in.
    save_dict = {}
    save_dict['problem_type'] = []
    save_dict['solve_time'] = []
    save_dict['output_dir'] = []
    save_dict['n_dofs'] = []
    save_dict['n_ele'] = []
    save_dict['refinement'] = []
    save_dict['num_load_steps'] = []

    # iterate through each problem type
    for n in refinements:
        
        Nx_n = Nx * n
        Ny_n = Ny * n
        Nz_n = Nz * n
        # grouping simulations by discretization for now.
        data_dir = data_dir_top + "/" + str(Nx_n) + "_" + str(Ny_n) + "_" + str(Nz_n)
        # data_dir = data_dir_top + "/" + str(Nx_n) + "_" + str(Ny_n)

        # make the directory, it's ok if it already exists.
        os.makedirs(data_dir, exist_ok=True)

        # only need to generate one directory for now, everything is saved there
        save_dict['output_dir'].append(data_dir)

        for problem_type in problem_list:
            for deg in degree_list:
                
                # save the problem type to the dict
                save_dict['problem_type'].append(problem_type + " " + str(deg))
                save_dict['refinement'].append(n)
            
                # generate the mesh - problem type is 'hex' or 'bspline'
                # I think this was only working for problems defined in 3D...
                
                # fe, mesh_obj = generate_2d_fe(problem_type, 
                #                 Nx_n, Ny_n, Lx, Ly, deg, vec, dim, data_dir)
                # generate the mesh - problem type is 'hex' or 'bspline'
                
                # fe, mesh_obj = generate_any_mesh(problem_type, 
                #                 Nx_n, Ny_n, Nz_n, Lx, Ly, Lz, deg, vec, dim, data_dir)

                if CASE == '2D':
                    mesh_obj, ele_type = create_2d_mesh(Nx, Ny, Lx, Ly, deg)
                elif CASE == '3D':
                    mesh_obj = create_3d_mesh(Nx, Ny, Nz, Lx, Ly, Lz, deg, data_dir)
                
                # not really sure what gauss order should be here tbh
                fe = FiniteElement(mesh_obj, vec, dim, ele_type, gauss_order = deg)
                
                # Define boundary locations.
                # x == 0
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
                # # z == 0
                # def bottom(point):
                #     return np.isclose(point[2], 0.0, atol=1e-5)
                # # z == Lz
                # def top(point):
                #     return np.isclose(point[2], Lz, atol=1e-5)
                
                # Define Dirichlet boundary values.
                def u_0(point):
                    return 0.
                # only small strain should be required to cause buckling
                # could I load step this if needed?...check D_example to see.
                def u_disp_x(point):
                    return -0.01

                # define dirichlet boundary condition related to surface wrinkling
                def u_cosine(point):
                    return A * np.cos(k*point[0])

                # define dirichlet boundary conditions
                bc0 = [[front]*3, [0,1,2], [u_0]*3]
                bc1 = [[back], [2], [u_cosine]]  
                
                dirichlet_bc_info = [[bc0, bc1]]

                # define the problem
                problem = LinearElastic(fe, dirichlet_bc_info = dirichlet_bc_info)

                # need to define lmbda and mu
                temp_array = onp.ones((problem.num_quads, vec, dim))
                lmbda = lmbda_s * temp_array
                mu = mu_s * temp_array
                
                # do we need to call a 'set' function?
                problem.internal_vars = [lmbda, mu]

                # define the solver
                # need to see if line search works for a hex problem.
                solver = Newton_Solver(problem, np.zeros_like(fe.nodes), line_search_flag=False)

                # might need to allow for a custom solver to be used to better handle contact...
                sol, info = solver.solve(atol=1e-6, max_iter=15)
                
                # save the solution at each iteration.
                if info[0]:
                    status = 'converged'
                else:
                    status = 'failed'

                # write the solution to a file
                sol_file = data_dir + "/substrate_bvp"+ str(f_max) + "_" + str(problem_type) + "_" + str(deg) + '.vtu'
                
                # save_sol only works for hexahedral elements right now...
                # updating this in the next push.
                save_sol(fe, sol, sol_file)
                
                # save dictionary at each step as we might eventually run out of memory for cubic bspline
                with open(RUN_NAME + ".pickle", 'wb') as f:
                    pickle.dump(save_dict, f)
