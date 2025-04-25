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

import time
import pickle

# allowing this to be more modular, where the user can chose from EXISTING material models and
# EXISTING types of loads, could be super cool. not sure how this would work though.

# # material properties for the substrate
# E_s = 10.
# nu_s = 0.33
# mu_s = E_s/2/(1+nu_s)
# lmbda_s = E_s*nu_s/(1+nu_s)/(1-2*nu_s)

# # material properties for the thin film
# E_f = 10. * 10**4
# nu_f = 0.33
# mu_f = E_f/2/(1+nu_f)
# lmbda_s = E_f*nu_f/(1+nu_f)/(1-2*nu_f)

def find_num_points_on_boundary(deg):
    
    if deg == 1:
        # assume linear hex elements
        pts_on_boundary = 4
    elif deg == 2:
        pts_on_boundary = 9
    elif deg == 3:
        pts_on_boundary = 16
    else:
        raise Exception("Only linear, quad, and cubic elements supported for now.")

    return pts_on_boundary

def on_substrate_boundary(cell_points,line,idx,pts_on_boundary):
    """
        determines if a cells lies on a substrate-film interface

        assumes that the mesh is relatively regular/that hexahedral elements are used.
    
    
        Arguments
        ---------
        cell_points : np.ndarray
            the points, in physical space, that are in a given FE cell
        line : float
            the line/plane that represents the film-substrate boundary
        idx : int [0,1,2]
            the parametric plane lives in. xz-plane -> y (1)
        deg : int [1,2,3]
            the degree of the finite element basis functions used to construct
            the given element (assumes the same degree in each direction)
    
    """

    pts_near_boundary = onp.sum(np.isclose(cell_points[:,idx], line))

    # return a boolean that indicates if the point is or isn't on the boundary
    # could a jax.lax.cond be used here as well?...
    return pts_near_boundary == pts_on_boundary


def gen_film_substrate_heights(N, L, h_s, deg, n_s = 2, n_buffer_1 = 4):
    """
        gives y-components for a mesh with a given number of elements in the thickness direction
    """

    # n_s determines the number of elements in the substrate; this can be changed.
    # N is the number of POINTS, which varies depending on the given element type.
    pt_array = onp.zeros(N)
    pt_array[-1] = L
    
    # uniformly space the elements in the 'film' section of the mesh.
    for i in range(n_s*deg):
        pt_array[-2 - i] = L - (i+1) * h_s/n_s * (1./deg)

    # gradually increase the size of the elements...should determine a good way to do this.
    scale = 2
    counter = 0
    for j in range(n_buffer_1*deg):
        # count down...but want to stop at 0!
        pt_array[-2 - n_s*deg - j] = pt_array[-1 - n_s*deg - j] - h_s/n_s * scale * (1./deg)
        counter += 1
        if counter % deg == 0:
            scale *= 2

    # for k in range(N - (n_s + n_buffer_1) + 1):
    
    # have the last chunk be evenly spaced
    n_remaining = N - (n_s + n_buffer_1) * deg
    pt_array[0:n_remaining] = onp.linspace(0, pt_array[n_remaining - 1], n_remaining)

    return pt_array

# a custom function that refines meshes for film-substrate problems.
def refine_mesh(mesh_obj, Ly, Ny, h_s, deg):
    """
        Attributes
        ----------
        mesh_obj : FiniteElement
        Ly : float, length of cube in y
        Ny : int, number of elements in y
        h_s : float, height of the substrate
        deg : degree, degree of the FE mesh
    """
    # # scale the points in the mesh to refine the mesh near the top surface
    pts = mesh_obj.points
    pts_new = onp.array(pts.tolist())
    # VMAP this.
    # identify, and replace, unique points in the y-direction.
    unique_y_pts = onp.linspace(0, Ly, Ny*deg + 1)
    #unique_y_pts = np.unique(pts_new[:,1])
    print(unique_y_pts)
    num_unique_y = len(unique_y_pts)
    # new points:
    new_unique_pts = gen_film_substrate_heights(num_unique_y, Ly, h_s, deg)

    i = 0
    # should probably vmap this...
    for uniq_pt in unique_y_pts:
        # should be in increasing order
        # need to allow for some tolerance.
        idx = onp.where(np.isclose(pts[:,1],uniq_pt,atol=1e-5))[0]
        # need to avoid overwriting points...
        pts_new[idx,1] = new_unique_pts[i]
        i += 1

    return pts_new

class LinearElastic(Problem):
    
    # need to pass a lambda function of some sort to account for a spatially varying material property
    # is there a way to MODEL the interface between the film and the substrate?
    #
    # applying material properties based on mesh tags would be a great thing to implement through this...
    def get_tensor_map(self):
        def stress(u_grad, lmbda, mu):
            epsilon = 0.5 * (u_grad + u_grad.T)
            # lmbda and mu are defined as internal variables using a jax.lax.conditional statement
            sigma = lmbda * np.trace(epsilon) * np.eye(self.dim[0]) + 2 * mu * epsilon
            return sigma
        return stress
    
    def get_surface_maps(self):

        # eventually want this to not be a dead load; should be easy.
        # keeping this as a dead load for right now.
        def traction(u, u_grad, x, p):
            # deadload
            return np.array([p, 0., 0.])

        return [traction]*2
    
    def set_params(self, params):

        # params[0] = R : radius of the indenter (scalar)
        # params[1] = h_0 : the height between the indenter and the undeformed half-plane at x = y= 0 (scalar)
        # params[2] = k_pen : the contact penalty parameter (scalar)

        # set internal vars_surface to [[ [surface_map_1], [surface_map_2] ]]
        self.internal_vars_surfaces = [[[params[0]],[params[1]]]]

# executable portion of the file
if __name__ == "__main__":
    
    # name of the job we're running
    RUN_NAME ="perturbed_element"
    data_dir_top = "/workspace/kmeyer/research/cardiax_testing/wrinkling/perturbed_element"

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

    # decide if we want to adaptively step - indenter height is what is being stepped.
    ADAPTIVE_LOAD_STEP = True

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
    
    # define problem geometry - hopefully there are no bugs with 2D...
    # (try compression first to check maybe?)
    # keeping this in 3D so I can more easily assess where potential bugs are occuring...
    # FOR TESTING
    DEBUG = False
    if DEBUG:
        Nx, Ny, Nz = 1, 7, 1
    else:
        Nx, Ny, Nz = 40, 7, 40   

    Lx, Ly, Lz = 1.0, 0.2, 1.0

    # need to make sure we can resolve the film vs. the substrate

    # might want to find a way to define the center of the indenter to be at 0.5,0.5
    vec = 3
    dim = 3

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
                fe, mesh_obj = generate_any_mesh(problem_type, 
                                Nx_n, Ny_n, Nz_n, Lx, Ly, Lz, deg, vec, dim, data_dir)
                
                pts_new = refine_mesh(mesh_obj, Ly, Ny, h_s, deg)

                print(onp.unique(pts_new[:,1]))

                mesh_new = meshio.Mesh(pts_new, mesh_obj.cells_dict)
                
                # save the mesh for testing purposes
                # mesh_dir = data_dir + "/film_substrate_mesh_undeformed_"+ str(f_max) + "_" + str(problem_type) + "_" + str(deg) + '.vtu'
                # meshio.write(mesh_dir, mesh_new)

                ele_type = get_ele_type(deg, "hex")
                
                # not really sure what gauss order should be here tbh
                fe = FiniteElement(mesh_new, vec, dim,ele_type,gauss_order = 1)

                # use mesh from GMSH!
                # this might automatically make the gauss order 3; might need to use reduced
                # integration rules at some point...
                # fe = FiniteElement(mesh, vec=vec, dim=dim, ele_type='tetra', gauss_order = 1)


                # immediately save the number of dofs and elements there are
                save_dict['n_dofs'].append(fe.num_total_dofs)
                save_dict['n_ele'].append(fe.num_cells)
                
                # Define boundary locations.
                # x == 0
                def left(point):
                    return np.isclose(point[0], 0., atol=1e-5)
                # x == Lx
                def right(point):
                    return np.isclose(point[0], Lx, atol=1e-5)
                
                # only apply the load to the film...
                def left_top(point):
                    return np.isclose(point[0], 0., atol=1e-5) * (point[1] >= Ly-h_s)
                # x == Lx
                def right_top(point):
                    return np.isclose(point[0], Lx, atol=1e-5) * (point[1] >= Ly-h_s)
                
                # y == 0
                def front(point):
                    return np.isclose(point[1], 0., atol=1e-5)
                # y == Ly
                def back(point):
                    return np.isclose(point[1], Ly, atol=1e-5)
                # z == 0
                def bottom(point):
                    return np.isclose(point[2], 0.0, atol=1e-5)
                # z == Lz
                def top(point):
                    return np.isclose(point[2], Lz, atol=1e-5)
                # x == 0 and y == 0
                def front_left(point):
                    return front(point) and left(point)
                
                # Define Dirichlet boundary values.
                def u_0(point):
                    return 0.
                # only small strain should be required to cause buckling
                # could I load step this if needed?...check D_example to see.
                def u_disp_x(point):
                    return -0.01

                # define dirichlet boundary conditions
                bc1 = [[bottom], [2], [u_0]]  
                bc2 = [[top], [2], [u_0]]  
                bc3 = [[front_left]*3, [0,1,2], [u_0]]
                bc4 = [[left], [0], [u_0]]
                bc_disp = [[right]*3, [0,1,2], [u_disp_x, u_0, u_0]]
                #bc3 = [[left]*3,[0,1,2],[u_0]*3]
                dirichlet_bc_info = [[bc1, bc2]]

                # applying a compressive force to the... film only?
                # could also maybe pre-strain the film only...
                location_fns = [[left, right]]

                # define the problem
                problem = LinearElastic(fe, dirichlet_bc_info = dirichlet_bc_info,
                                        location_fns = location_fns)

                # using problem.physical_quad_points, assign lame parameters to each
                # cell/point. gets passed to get_tensor_map

                # use h_s to help figure this out.
                # pseudocode
                # if y >= L - h_s:
                #     pt_lmbda = lmbda_f
                #     pt_mu = mu_f
                # else:
                #     pt_lmbda = lmbda_s
                #     pt_mu = mu_s
                
                # don't think we need a jax.lax.cond function; could try using for practice tho.
                # not totally sure, but might need this to be:
                # (num_cells, num_quads, num_nodes, vec, dim)
                # ...
                # avoiding passing a massive array seems like it would be super helpful...
                # can we pass a scalar and only vmap over u_grads? that would be great.
                # trying it the brute force way for now though.
                #lmbda = onp.zeros((fe.num_cells, fe.num_quads, fe.num_nodes, fe.vec, fe.dim))
                #mu = onp.zeros((fe.num_cells, fe.num_quads, fe.num_nodes, fe.vec, fe.dim))
                

                # use global variables to allow this to be vmapped?...
                num_nodes = fe.num_nodes
                num_quads = fe.num_quads

                # should this be num_quads or num_nodes??? little uncertain...
                temp_array = onp.ones((num_quads, vec, dim))

                film_region = lambda f,s: f * temp_array
                substrate_region = lambda f,s: s * temp_array

                # E_fun = 

                # # this would be super helpful to vmap...and use a conditional...
                # for i in range(fe.num_cells):
                # mu_data = onp.array([mu_f, mu_s])
                # lmbda_data = onp.array([lmbda_f, lmbda_s])
                
                # vec and dim are already defined
                # might want to split this up
                def spatially_varying_field(pts, f, s):
                    field = jax.lax.cond(np.sum(pts[:,1] > (Ly - h_s)) > 0, film_region, substrate_region, f, s)
                    return field
                
                #lmbda = jax.lax.cond(onp.sum(fe.points[fe.cells[i]] > (Ly - h_s)) > 0, film_region, substrate_region, (f, s, num_nodes, num_quads, vec, dim))
                
                # set the params
                lmbda = jax.vmap(spatially_varying_field, in_axes=(0, None, None))(fe.points[fe.cells], lmbda_f, lmbda_s)
                mu = jax.vmap(spatially_varying_field, in_axes=(0, None, None))(fe.points[fe.cells], mu_f, mu_s)

                # the total number of cells with E_f and lmbda_f should be 2
                # with the DEBUG case.
                num_f = 2
                num_f_total = num_f*num_quads*vec*dim

                if DEBUG:
                    total_lmbda = onp.sum(lmbda == lmbda_f)
                    print(f"Total lmbda_f: {total_lmbda}")
                    print(f"lmbda f expected: {num_f_total}")
                    assert(total_lmbda == num_f_total)
                    assert(onp.sum(mu == mu_f) == num_f_total)

                # for a random element in the subtrate that lies on the film-substrate
                # interface, apply the material properties of the film to attempt
                # to introduce an imperfection that should induce buckling
                # there should be a LINE that defines this interface within the material,
                # at L - h_s.
                # on_substrate_boundary(cell_points,line,idx,pts_on_boundary)

                # find all the cells that lie on the substrate and touch this line:
                line = Ly - h_s
                idx = 1             # y-direction for now
                # this is really just (deg + 1)**2...
                pts_on_bndry = find_num_points_on_boundary(deg)
                substrate_boundary_cells = jax.vmap(on_substrate_boundary, in_axes=(0, None,None,None))(fe.points[fe.cells],line, idx, pts_on_bndry)

                # check that there's at least one cell
                assert len(substrate_boundary_cells) > 0

                # choose a cell that is not on the boundary (at least I think)
                idx_list = onp.arange(fe.num_cells)[substrate_boundary_cells]
                # quickly check that we aren't running a test
                if DEBUG:
                    perturb_element_idx = idx_list[0]
                else:
                    perturb_element_idx = idx_list[Nx + Ny + 2]

                print(f"perturbed element: {perturb_element_idx}")

                lmbda.at[perturb_element_idx].set(lmbda_f * onp.ones_like(lmbda[perturb_element_idx]))
                mu.at[perturb_element_idx].set(mu_f * onp.ones_like(mu[perturb_element_idx]))
                    
                # do we need to call a 'set' function?
                problem.internal_vars = [lmbda, mu]

                # define the solver
                # need to see if line search works for a hex problem.
                solver = Newton_Solver(problem, np.zeros_like(fe.nodes), line_search_flag=False)

                # initialize normal vecs
                left_normals = fe.get_surface_normals(left)
                right_normals = fe.get_surface_normals(right)

                # check the shape of the normals to make sure they are properly identified:
                print(f"left normals shape: {left_normals.shape}")
                print(f"right normals shape: {right_normals.shape}")
                # initialize normals variable
                #normals = init_normals

                # initialize shape array
                #force_arr = np.ones_like(normals[:,:,0])
                t_arr_left = -1 * f_0 * np.ones_like(left_normals[:,:,2])
                t_arr_right = f_0 * np.ones_like(right_normals[:,:,2])

                # start a timer
                time_start = time.time()

                # adaptive load stepping parameters
                total_step_attempts = 0
                n_bisections = 0
                sol_old = np.zeros((len(fe.nodes), fe.vec))
                    
                # start load stepping - use code from the CARDIAX timing
                # that double the step size if we have convergence.

                # only the normals in the reference configuration need to get passed
                problem.set_params([t_arr_left,t_arr_right])

                if ADAPTIVE_LOAD_STEP:

                    # step until the max load is reached, or until the load stepping breaks
                    # initialize the state of the previous step; need to know to determine how to update the load.
                    prev_step_converged = True
                    
                    while (f_step < f_max) and (n_bisections < 4):

                        # immediately update the load; initial f_step = 0!!
                        if prev_step_converged:
                            # progess the load
                            f_step = f_step + delta_f
                        else:
                            # halve the load (delta_f is halved if solver fails, see later in this loop)
                            f_step = f_step - delta_f

                        # force last step to be AT the maximal force
                        if f_step > f_max:
                            f_step = f_max

                        # reshape the load to the appropriate size
                        #f_i_full = f_step * force_arr

                        t_arr_left = -1 * f_step * np.ones_like(left_normals[:,:,2])
                        t_arr_right = f_step * np.ones_like(right_normals[:,:,2])      

                        # only the normals in the reference configuration need to get passed
                        problem.set_params([t_arr_left,t_arr_right])

                        # using bisection_solve should:
                        # A) be better
                        # B) save previous solution to self.init_guess, iff!!!!! it doesn't converge at the first iteration.
                        
                        # I think bisection is actually broken right now
                        # was load stepping ever working?
                        #sol, info = solver.bisection_solve()
                        
                        # there's some instability occuring at large displacements.
                        # I think we need to solve a different problem; displacements are huge. Not sure how
                        # this wasn't a problem with shruti's code......
                        sol, info = solver.solve(atol=1e-6, max_iter=15)
                        
                        # save the solution at each iteration.
                        if info[0]:
                            status = 'converged'
                        else:
                            status = 'failed'

                        sol_file = data_dir + "/" + str(problem_type) + "_" + str(deg) + '_C1_1_D1_100_p_' + str(f_max) +  '_step_' + str(f_step) + '_' + status + '.vtu'
                        sol = sol.reshape((len(fe.nodes), fe.vec))
                        
                        # don't save the solution at each step! wasteful.
                        #save_sol(fe, sol, sol_file)
                            
                        # print some info to the screen; # TODO: improve program output messages                            
                        if info[0]:
                            print("SUCCESS FOR PRESSURE: " + str(f_step))
                        else:
                            if n_bisections > 3:
                                raise Exception("Problem did not converge with " + str(f_step) + " N load.")

                        # if the solve converged, step forward
                        if info[0]:                    
                            # let our load stepper know that the previous step converged
                            prev_step_converged = True

                            # save the previous solution as an initial guess 
                            solver.initial_guess = sol

                            # increase the step size if the initial problem residual is 'small'
                            # if info[2] < 1e-3:
                            #     delta_f = delta_f*2
                            #     n_bisections = 0 # reset the number of bisections
                            
                            # increase the step if the previous iteration converged - only incrementing per
                            # the size of the initial residual is sub-optimal
                            delta_f = delta_f*2
                            n_bisections = 0 # reset the number of bisections

                            # should also try to bound the step size by some geometric quantity
                            # check david kamensky's paper for guidance regarding this!!
                            # (his paper is the standard for valvular simulations... beat the standard!!!!!)

                        # decrease the step size and repeat if the solve did not converge
                        # use the same initial guess; no solver parameters need to be updated other than force.
                        else:
                            # the old initial guess will be used, no need to update.
                            # (unless the last saved solution is somehow 'wrong'...)

                            prev_step_converged = False
                            delta_f = delta_f/2.
                            n_bisections += 1

                        total_step_attempts += 1
                else:

                    # might need to allow for a custom solver to be used to better handle contact...
                    sol, info = solver.solve(atol=1e-6, max_iter=15)
                    
                    # save the solution at each iteration.
                    if info[0]:
                        status = 'converged'
                    else:
                        status = 'failed'

                    #print(f"HEIGHT: {h_step} + {status}.")

                # timer
                time_stop = time.time()
                solver_time = time_stop - time_start
                save_dict['solve_time'].append(solver_time)

                # write the solution to a file
                sol_file = data_dir + "/film_substrate_mesh_f_"+ str(f_max) + "_" + str(problem_type) + "_" + str(deg) + '.vtu'
                
                # save_sol only works for hexahedral elements right now...
                # updating this in the next push.
                save_sol(fe, sol, sol_file)
                
                # wait until the solution is written to a file; asynchronous saving can throw an error?...
                time.sleep(1) # I feel like this is not the best approach?...


                # append the total number of iterations!
                save_dict['num_load_steps'].append(total_step_attempts)

                # save dictionary at each step as we might eventually run out of memory for cubic bspline
                with open(RUN_NAME + ".pickle", 'wb') as f:
                    pickle.dump(save_dict, f)
