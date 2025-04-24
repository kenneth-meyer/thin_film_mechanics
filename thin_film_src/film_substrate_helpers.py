"""
    Kenneth Meyer
    4/24/25
    
    Various functions that should help with tasks related to film-substrate systems.
    
    Will include:
        1. checking the critical wavelength, amplitude, and membrane force for a film-substrate system
        2. checking if a given FE mesh is compatible with the information above
        3. generating FE meshes with spatially varying material properties
        4. ...

    Other functionality that might be helpful:
        1. use of input files
        2. classes that set up specific types of problems; avoids the need to change a script each time we run
        ...
"""


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

def spatially_varying_linear_elastic_moduli(fe, fe_info, DEBUG=False):
    """
        Generates spatially varying elastic moduli

        Note: fe_info is a custom class that stores helpful information for the given problem
    """
    # use global variables to allow this to be vmapped?...
    num_nodes = fe.num_nodes
    num_quads = fe.num_quads

    # should this be num_quads or num_nodes??? little uncertain...
    temp_array = onp.ones((num_quads, fe.vec, fe.dim))

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
        field = jax.lax.cond(np.sum(pts[:,1] > (fe_info.Ly - fe_info.h_s)) > 0, film_region, substrate_region, f, s)
        return field
    
    #lmbda = jax.lax.cond(onp.sum(fe.points[fe.cells[i]] > (Ly - h_s)) > 0, film_region, substrate_region, (f, s, num_nodes, num_quads, vec, dim))
    
    # set the params
    lmbda = jax.vmap(spatially_varying_field, in_axes=(0, None, None))(fe.points[fe.cells], fe_info.lmbda_f, fe_info.lmbda_s)
    mu = jax.vmap(spatially_varying_field, in_axes=(0, None, None))(fe.points[fe.cells], fe_info.mu_f, fe_info.mu_s)

    # the total number of cells with E_f and lmbda_f should be 2
    # with the DEBUG case.
    num_f = 2
    num_f_total = num_f*num_quads*fe.vec*fe.dim

    if DEBUG:
        total_lmbda = onp.sum(lmbda == fe_info.lmbda_f)
        print(f"Total lmbda_f: {total_lmbda}")
        print(f"lmbda f expected: {num_f_total}")
        assert(total_lmbda == num_f_total)
        assert(onp.sum(mu == fe_info.mu_f) == num_f_total)

    # for a random element in the subtrate that lies on the film-substrate
    # interface, apply the material properties of the film to attempt
    # to introduce an imperfection that should induce buckling
    # there should be a LINE that defines this interface within the material,
    # at L - h_s.
    # on_substrate_boundary(cell_points,line,idx,pts_on_boundary)

    # find all the cells that lie on the substrate and touch this line:
    line = fe_info.Ly - fe_info.h_s
    idx = 1             # y-direction for now
    # this is really just (deg + 1)**2...
    pts_on_bndry = find_num_points_on_boundary(fe_info.deg)
    substrate_boundary_cells = jax.vmap(on_substrate_boundary, in_axes=(0, None,None,None))(fe.points[fe.cells],line, idx, pts_on_bndry)

    # check that there's at least one cell
    assert len(substrate_boundary_cells) > 0

    # choose a cell that is not on the boundary (at least I think)
    idx_list = onp.arange(fe.num_cells)[substrate_boundary_cells]
    # quickly check that we aren't running a test
    if DEBUG:
        perturb_element_idx = idx_list[0]
    else:
        perturb_element_idx = idx_list[fe_info.Nx + fe_info.Ny + 2]

    print(f"perturbed element: {perturb_element_idx}")

    lmbda.at[perturb_element_idx].set(fe_info.lmbda_f * onp.ones_like(lmbda[perturb_element_idx]))
    mu.at[perturb_element_idx].set(fe_info.mu_f * onp.ones_like(mu[perturb_element_idx]))
    
    return lmbda, mu
