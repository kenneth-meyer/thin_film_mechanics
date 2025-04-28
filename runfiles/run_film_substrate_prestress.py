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
jax.config.update("jax_enable_x64", True)

# will convert the unittest code that generates geometries in one shot (without)
from cardiax.fe import FiniteElement
from cardiax.iga import BSpline
from cardiax.generate_mesh import generate_any_mesh, get_ele_type
from cardiax.problem import Problem
from cardiax.solver import Newton_Solver
from cardiax.utils import save_sol

from thin_film_src.thinfilm_inputfile_handler import create_2d_mesh, create_3d_mesh
from thin_film_src.material_models import LinearElasticPrestrain, LinearElastic
from thin_film_src.film_substrate_helpers import (
    FilmSubstrate,
    spatially_varying_linear_elastic_moduli,
    film_prestrain,
    refine_mesh
)

import time
import pickle
import sys

# executable portion of the file
if __name__ == "__main__":
    # allow the user to vary poisson's ratio to examine effects of it changing
    try:
        # should prolly read these as a yaml file, oh well
        h = float(sys.argv[1])
        H = float(sys.argv[2])
        E_s = float(sys.argv[3])
        nu_s = float(sys.argv[4])
        E_f = float(sys.argv[5])
        nu_f = float(sys.argv[6])
        f_c_scale = float(sys.argv[7])

    except:
        # raise Exception("Invalid input!")
        h=0.2
        H=2
        E_s=10.0   # 10 MPa
        nu_s=0.4
        E_f=10000 # 10 GPa
        nu_f=0.45
        # can scale the initial membrane force
        f_c_scale=1.005

    # name of the job we're running
    RUN_NAME ="nikravesh_recreation"
    data_dir_top = "/workspace/kmeyer/research/cardiax_testing/wrinkling/nikravesh_recreation"

    # save each run based on the material parameters used!
    data_dir = data_dir_top + "/h_" + str(h) + "_H_" + str(H) + "_Es_"+ str(E_s) + "_nus_"+ str(nu_s) + "_Ef_"+ str(E_f) + "_nuf_" + str(nu_f)    

    # determine the case of the problem to set up the geometry
    CASE = '2D'

    # define element size heuristically from previous wrinkling examples
    num_substrate_cells = 1
    # u_max = 0.02            # 20 um
    u_max = 0.02 * H
    num_ramp_steps = 100             # need to step the dirichlet boundary condition!
    y_ratio = h/num_substrate_cells

    # there are papers that inspect the height-length ratio; I'm avoiding this for now.
    if CASE == '2D':
        # make this a square for now... might be good to make longer though.
        #Nx, Ny = int(H / y_ratio), int((H + h) / y_ratio)
        
        # manually prescribing the number of cells; need to use mesh refinement now.
        Nx, Ny = 100,101
        Lx, Ly = H, H + h
        vec = 2
        dim = 2
        deg = 1 # linear elements
    else:
        Nx, Ny, Nz = 40, 7, 40   
        Lx, Ly, Lz = 1.0, 0.2, 1.0
        vec = 3
        dim = 3
        deg = 1 # linear elements

    # h_f, H_s, E_s, nu_s, E_f, nu_f
    film_substrate_obj = FilmSubstrate(h, H, E_s, nu_s, E_f, nu_f, deg, dim)

    # get critical force, scale by factor set in input file
    N_c = film_substrate_obj.get_critical_force_thin_limit()
    N_0_11 = N_c * f_c_scale

    # determine the pre-strain that is required to induce the required force in the film!
    # note: does this need to be scaled depending on how many elements we have? likely not.
    # needs to be in compression...
    e_11_0 = N_0_11 / (h * film_substrate_obj.E_f_bar)

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
        
        if CASE=='2D':
            Nx_n = Nx * n
            Ny_n = Ny * n
            
        else:
            Nx_n = Nx * n
            Ny_n = Ny * n
            Nz_n = Nz * n
            
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
                    mesh_obj, ele_type = create_2d_mesh(Nx_n, Ny_n, Lx, Ly, deg)
                elif CASE == '3D':
                    mesh_obj = create_3d_mesh(Nx_n, Ny_n, Nz, Lx, Ly, Lz, deg, data_dir)


                # 8:24 PM - this is not working for some reason!!! idk what's going on
                #           but negative numbers in the y direction are appearing!
                #
                #           commenting out the mesh refinement section for now.
                #
                # refine the mesh!
                # pts_new = refine_mesh(mesh_obj, Ly, Ny, h, deg, n_s=num_substrate_cells)
                
                # update the mesh
                # mesh_obj.points = pts_new

                # print(Lx)
                # print(onp.max(pts_new[:,0]))

                # not really sure what gauss order should be here tbh
                fe = FiniteElement(mesh_obj, vec, dim, ele_type, gauss_order = deg)
                
                # Define boundary locations.
                # x == 0
                def left(point, i=0):
                    return np.isclose(point[0], 0., atol=1e-7)
                # x == Lx

                # might need to make this a moving boundary? idk what's going on.
                # def right(point,**kwargs):
                #     return np.isclose(point[0], Lx - kwargs['i'] * u_max / num_ramp_steps, atol=1e-7)
                
                # points won't move, so we don't need to save anything
                # I think things are getting messed up because none of these functions are pure...
                def right(point,Lx):
                    return np.isclose(point[0], Lx, atol=1e-7)
                
                # x ==0 and y == 0
                def front_left(point, i=0):
                    return np.isclose(point[0], 0., atol=1e-7) * np.isclose(point[1], 0., atol=1e-7)
                
                # y == 0
                def front(point,i=0):
                    return np.isclose(point[1], 0., atol=1e-7)
                # y == Ly
                def back(point,i=0):
                    return np.isclose(point[1], Ly, atol=1e-7)
                # # z == 0
                # def bottom(point):
                #     return np.isclose(point[2], 0.0, atol=1e-5)
                # # z == Lz
                # def top(point):
                #     return np.isclose(point[2], Lz, atol=1e-5)

                # this seems to not be updating...
                def u_ramp(point, i, u_max, num_ramp_steps):
                    return -1 * (i + 1) * u_max / num_ramp_steps
                
                # Define Dirichlet boundary values.
                def u_0(point, **kwargs):
                    return 0.
                
                # only small strain should be required to cause buckling
                # could I load step this if needed?...check D_example to see.
                def u_disp_x(point):
                    return -0.01
                
                if RUN_NAME == "nikravesh_recreation":
                    # define dirichlet boundary conditions
                    bc0 = [[left], [0], [u_0]]   
                    bc1 = [[front_left]*2, [0,1], [u_0]*2]
                    bc2 = [[right], [0], [u_ramp]]             
                    dirichlet_bc_info = [[bc0, bc1, bc2]]


                
                # define the problem
                # problem = LinearElasticPrestrain(fe, dirichlet_bc_info = dirichlet_bc_info)
                #problem = LinearElastic(fe, dirichlet_bc_info = dirichlet_bc_info)

                # define lmbda and mu as spatially varying fields
                lmbda, mu = spatially_varying_linear_elastic_moduli(fe, film_substrate_obj, num_cells_in_plane=Nx, num_f_cells = num_substrate_cells, perturb=True)

                # define the pre-stress applied to the film
                substrate_e_11 = 0 # don't apply prestrain to the substrate
                e_11 = film_prestrain(fe, film_substrate_obj, e_11_0, substrate_e_11)


                # define the problem and solver
                # the kwargs don't seem to be getting updated...
                dbc_kwargs = {'i': 0, 'u_max': u_max, 'num_ramp_steps': num_ramp_steps, 'Lx': Lx}
                problem = LinearElastic(fe, dirichlet_bc_info = dirichlet_bc_info, dbc_kwargs=dbc_kwargs)
                
                # problem = LinearElastic(fe, dirichlet_bc_info = dirichlet_bc_info, i=0)
                
                problem.internal_vars = [lmbda, mu]

                solver = Newton_Solver(problem, np.zeros_like(fe.nodes), line_search_flag=False)



                # e_11 = e_11_0 * np.ones_like(lmbda)

                # check the max and min of each value, makes sure they're defined correctly
                print(np.max(e_11[:,:,0,0]))
                print(np.min(e_11[:,:,0,0]))
                print(np.max(lmbda))
                print(np.min(lmbda))
                
                # problem.internal_vars = [lmbda, mu, e_11]
                #problem.internal_vars = [lmbda, mu]

                # define the solver
                # need to see if line search works for a hex problem.
                #solver = Newton_Solver(problem, np.zeros_like(fe.nodes), line_search_flag=True)


                # step the solve!
                # for i in range(len(num_ramp_steps)):
                # do this once for debugging the mesh refinement.
                STEP=True
                sol = np.zeros_like(fe.nodes)
                if STEP:
                    for i in range(num_ramp_steps):
                        # update the boundary condition function
                        # def right_i(point):
                        #     np.isclose(point[0], Lx - (u_max * i / num_ramp_steps), atol=1e-5)

                        # # ^ bc_inds doesn't actually change, look into a better way of doing this
                        # def u_ramp(point):
                        #     return -1 * u_max * (i+1) / num_ramp_steps
                        
                        # print(u_ramp(0))
                        
                        # bc2 = [[right_i], [0], [u_ramp]]             
                        # dirichlet_bc_info = [[bc0, bc1, bc2]]
                        # # might need to make this stateful...
                        
                        # # update problem and solver each time to make this work... EW
                        # problem = LinearElastic(fe, dirichlet_bc_info = dirichlet_bc_info)
                        # problem.internal_vars = [lmbda, mu]
                        # solver = Newton_Solver(problem, sol, line_search_flag=True)

                        # problem.update_dirichlet_info_custom(dirichlet_bc_info, 0)

                        # update the initial guess with the previous solution
                        #solver.initial_guess = sol

                        # update problem too? this might need to be a stateful computation though...
                        # solver.problem = problem
                        # ^ re-loading the problem class at each iteration is far too expensive to do this

                        dbc_kwargs['i'] = i

                        # might need to allow for a custom solver to be used to better handle contact...
                        sol, info = solver.solve(atol=1e-5, max_iter=30, dirichlet_bc_info=dirichlet_bc_info, **dbc_kwargs)

                        # update the initial guess!
                        solver.initial_guess = sol

                        # need to figure out how the solver class interfaces with the problem class to update
                        # the boundary condition stepping..

                    
                        # save the solution at each iteration.
                        if info[0]:
                            status = 'converged'
                        else:
                            status = 'failed'
                            print("Ramp loading failed")
                            break
                else:
                    problem = LinearElastic(fe, dirichlet_bc_info = dirichlet_bc_info)
                    problem.internal_vars = [lmbda, mu]

                    # define the solver
                    # need to see if line search works for a hex problem.
                    #solver = Newton_Solver(problem, np.zeros_like(fe.nodes), line_search_flag=True)

                    # might need to allow for a custom solver to be used to better handle contact...
                    sol, info = solver.solve(atol=1e-6, max_iter=15)


                # write the solution to a file
                sol_file = data_dir + "/" + CASE + "_Nx_" + str(Nx_n) + "_Ny_" + str(Ny_n) + "_Nc_" + str(N_c) + "_u_" + str(u_max) + '.vtu'
                
                # save the mesh, along with material properties and prestrains, to a .vtu file
                # the material properties and prestrain are the same for each point in each cell, can index just one of them
                lmbda_cells = lmbda[:,0,0,0]
                mu_cells = mu[:,0,0,0]
                e_11_cells = e_11[:,0,0,0]

                # create a dictionary to save the cell information
                cell_info = {'lmbda': onp.array(lmbda_cells.tolist()), 'mu': onp.array(mu_cells.tolist()), 'e_11_0': onp.array(e_11_cells.tolist())}
                
                save_sol(fe, sol, sol_file, cell_type=ele_type, cell_infos=[cell_info])
                
                # save dictionary at each step as we might eventually run out of memory for cubic bspline
                with open(RUN_NAME + ".pickle", 'wb') as f:
                    pickle.dump(save_dict, f)
