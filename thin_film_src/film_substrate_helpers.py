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

##########################################################################################
####################### Classes describing film-substrate systems ########################
##########################################################################################

# NOTE: any updates made to variables in this class need to be STATEFUL
class FilmSubstrate:
    def __init__(self, h, H, E_s=10, nu_s=0.33, E_f=10**5, nu_f=0.33, deg=1, dim=2):
        
        # geometric quantities for the film and substrate
        self.h = h
        self.H = H

        # degree of the element used and dimension the problem lives in
        self.deg = deg
        self.dim = dim

        # material properties for the substrate
        self.E_s = E_s
        self.nu_s = nu_s
        self.mu_s = self.E_s/2/(1+self.nu_s)
        self.lmbda_s = self.E_s*self.nu_s/(1+self.nu_s)/(1-2*self.nu_s)

        # material properties for the thin film
        self.E_f = E_f
        self.nu_f = nu_f
        self.mu_f = self.E_f/2/(1+self.nu_f)
        self.lmbda_f = self.E_f*self.nu_f/(1+self.nu_f)/(1-2*self.nu_f)

        # E_bar for film and substrate
        self.E_f_bar = self.E_f / (1 - self.nu_f**2)
        self.E_s_bar = self.E_s / (1 - self.nu_s**2)

    # I got a bit confused regarding their analytical solution test
    # and what the amplitude and wavelength of sinusoidal wrinkles
    # in the film are at EQUILIBRIUM for a given system 
    def get_initial_membrane_force(self):
        """
            
        """

        raise NotImplementedError

    def get_wavelength_thick_limit(self):
        """
            Allows us to compute the wavelength without h_f for the
            substrate-only system
        """
        return 2 * np.pi * self.h * np.cbrt(self.E_f_bar / (3*self.E_s_bar))
    
    def get_wavenumber_thick_limit(self):
        """
            k = 2pi/lmbda (wavelength)
        """

        # wavelength
        lmbda = self.get_wavelength_thick_limit()
        return 2 * np.pi / lmbda

    def get_critical_force_thick_limit(self):
        """
            N_c / (h * bar{E}_f)
        """
        return self.h * self.E_f_bar * 0.25 * np.cbrt(3 * self.E_s_bar / self.E_f_bar) ** 2
    
    def get_amplitude_thick_limit(self):
        """
            A_eq / h
        """
        # N_0_11 is defined ANALYTICALLY to derive an analytical solution,
        # and I think it can be defined as an INITIAL CONDITION to compute
        # what the equilibrium amplitude and wavenumber are for a system
        # whose film is under a certain initial load.
        N_0_11 = self.get_initial_membrane_force()
        N_c = self.get_critical_force_thick_limit()
        
        return self.h * np.sqrt(N_0_11 / N_c - 1)

    def get_k_thick_limit(self):
        raise NotImplementedError

    # need to be able to compute the critical force to determine the amplitude...
    # def get_A_thick_limit(self):

    # critical wavelength, membrane force, and wrinkle amplitude for thin substrate
    def get_wavelength_thin_limit(self):
        return 2 * np.pi * self.h * np.sqrt(np.sqrt(self.H * self.E_f_bar * (1 - 2*self.nu_s) / 
                                                    (12 * self.h * self.E_s_bar * (1 - self.nu_s)**2)))
    
    def get_critical_force_thin_limit(self):
        return self.h * self.E_f_bar * np.sqrt(self.h * self.E_s_bar * (1 - self.nu_s)**2 / (3 * self.H * self.E_f_bar * (1 - 2*self.nu_s)))
    
    # also need to get the critical wavelength; need to decide on an initial force though.

##########################################################################################
############## Functions to aid refinement and material param definitions ################
##########################################################################################

def find_num_points_on_boundary(deg, dim):
    if dim == 3:
        if deg == 1:
            # assume linear hex elements
            pts_on_boundary = 4
        elif deg == 2:
            pts_on_boundary = 9
        elif deg == 3:
            pts_on_boundary = 16
        else:
            raise Exception("Only linear, quad, and cubic elements supported for now.")

    elif dim == 2:
        if deg == 1:
            # quad elements
            pts_on_boundary = 2
        else:
            raise NotImplementedError("Only quads (linear elements) are supported in 2D")
        
    return pts_on_boundary

def on_substrate_boundary(cell_points,line,idx,pts_on_boundary, subset):
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

    # pts_near_boundary = onp.sum(np.isclose(cell_points[:,idx], line))
    # we only want to check if the y-coordiante of the points on the top of an element (below the substrate)
    # lie on the line.
    pts_near_boundary = onp.sum(np.isclose(cell_points[subset, idx], line))

    # return a boolean that indicates if the point is or isn't on the boundary
    # could a jax.lax.cond be used here as well?...
    return pts_near_boundary == pts_on_boundary

# checking for cells above and below the boundary will look different!
def get_bottom_face_points(deg, dim):
    if dim == 2:
        if deg == 1:
            bottom_face_points = [0,1]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return bottom_face_points

def get_top_face_points(deg, dim):
    if dim == 2:
        if deg == 1:
            top_face_points = [2,3]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return top_face_points

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
# THIS MIGHT BE DEPRECATED after the updates that were made for the 2D case.
def refine_mesh(mesh_obj, Ly, Ny, h_s, deg, n_s=2):
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
    
    num_unique_y = len(unique_y_pts)
    # new points:
    new_unique_pts = gen_film_substrate_heights(num_unique_y, Ly, h_s, n_s, deg)

    i = 0
    # should probably vmap this...
    for uniq_pt in unique_y_pts:
        # should be in increasing order
        # need to allow for some tolerance.
        idx = onp.where(np.isclose(pts[:,1],uniq_pt,atol=1e-7))[0]
        # need to avoid overwriting points...
        pts_new[idx,1] = new_unique_pts[i]
        i += 1

    return pts_new

def spatially_varying_linear_elastic_moduli(fe, fe_info, num_cells_in_plane, DEBUG=False, num_f_cells = 2, perturb=False):
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

    # vec and dim are already defined
    # might want to split this up
    def spatially_varying_field(pts, f, s):
        # update: FilmSubstrate class is now being used, and
        # H and h are used directly to define the dimensions of the domain.
        field = jax.lax.cond(np.sum(pts[:,1] > fe_info.H) > 0, film_region, substrate_region, f, s)
        return field
    
    #lmbda = jax.lax.cond(onp.sum(fe.points[fe.cells[i]] > (Ly - h_s)) > 0, film_region, substrate_region, (f, s, num_nodes, num_quads, vec, dim))
    
    # set the params
    lmbda = jax.vmap(spatially_varying_field, in_axes=(0, None, None))(fe.points[fe.cells], fe_info.lmbda_f, fe_info.lmbda_s)
    mu = jax.vmap(spatially_varying_field, in_axes=(0, None, None))(fe.points[fe.cells], fe_info.mu_f, fe_info.mu_s)

    # lots of this might be hardcoded; need to check this out.
    if perturb:
        num_f = num_f_cells
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
        #line = fe_info.Ly - fe_info.h_s
        line = fe_info.H
        idx = 1             # y-direction for now
        subset = get_top_face_points(fe_info.deg, fe_info.dim)
        # this is really just (deg + 1)**2...
        pts_on_bndry = find_num_points_on_boundary(fe_info.deg, fe_info.dim)
        substrate_boundary_cells = jax.vmap(on_substrate_boundary, in_axes=(0, None,None,None, None))(fe.points[fe.cells], line, idx, pts_on_bndry, subset)

        # check that there's at least one cell
        assert len(substrate_boundary_cells) > 0

        # choose a cell that is not on the boundary (at least I think)
        idx_list = onp.arange(fe.num_cells)[substrate_boundary_cells]
        # quickly check that we aren't running a test
        if DEBUG:
            perturb_element_idx = idx_list[0]
        else:
            # might need to update this if we're doing a 2D vs. a 3D case
            # also need to check if this is actually on the film-substrate barrier...
            # doing this for a non-refined case is likely a good idea.
            perturb_element_idx = idx_list[len(idx_list)//3] # pick an index halfway along the boundary

        print(f"perturbed element: {perturb_element_idx}")

        lmbda = lmbda.at[perturb_element_idx].set(fe_info.lmbda_f * onp.ones_like(lmbda[perturb_element_idx]))
        mu = mu.at[perturb_element_idx].set(fe_info.mu_f * onp.ones_like(mu[perturb_element_idx]))

    # otherwise, don't modify lmbda and mu, and simply return them.    
    return lmbda, mu

def film_prestrain(fe, fe_info, film_e_11, substrate_e_11):
    """
        Generate a pre-strain in the film

        can be generalized to generate a spatially varying prestrain field.
    """

    # use global variables to allow this to be vmapped?...
    num_nodes = fe.num_nodes
    num_quads = fe.num_quads

    # should this be num_quads or num_nodes??? little uncertain...
    temp_array = onp.ones((num_quads, fe.vec, fe.dim))

    film_region = lambda f,s: f * temp_array
    substrate_region = lambda f,s: s * temp_array

    # vec and dim are already defined
    # might want to split this up
    def spatially_varying_field(pts, f, s):
        # update: FilmSubstrate class is now being used, and
        # H and h are used directly to define the dimensions of the domain.
        field = jax.lax.cond(np.sum(pts[:,1] > fe_info.H) > 0, film_region, substrate_region, f, s)
        return field
    
    #lmbda = jax.lax.cond(onp.sum(fe.points[fe.cells[i]] > (Ly - h_s)) > 0, film_region, substrate_region, (f, s, num_nodes, num_quads, vec, dim))
    
    # set the params
    e_11 = jax.vmap(spatially_varying_field, in_axes=(0, None, None))(fe.points[fe.cells], film_e_11, substrate_e_11)

    # e_11 is a scalar field; it's a component of the small strain tensor \epsilon
    return e_11
