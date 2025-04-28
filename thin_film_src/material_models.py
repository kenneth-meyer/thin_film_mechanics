"""
    Contains a library of different 'material models'
    (really, material models couple with BVP definitions)
    for use with thin-film mechanics related problems.
"""

import meshio
import os
import jax.numpy as np
import numpy as onp
import jax

from cardiax.problem import Problem

# try documenting what PDEs describe these within the comments, would be super cool to include
# as documentation!
# (this is definitely on my todo list AFTER my presentation, and likely paper, are completed.)
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

class LinearElasticPrestrain(Problem):
    
    # need to pass a lambda function of some sort to account for a spatially varying material property
    # is there a way to MODEL the interface between the film and the substrate?
    #
    # applying material properties based on mesh tags would be a great thing to implement through this...
    def get_tensor_map(self):
        def stress(u_grad, lmbda, mu, e_11):

            # small strain tensor
            epsilon = 0.5 * (u_grad + u_grad.T)
            
            # manually prescribe the pre-strain
            # I think lmbda and mu are defined as [dim x dim] tensors,
            # which doesn't change computation (???) but is sketch.
            #pre_strain = np.eye(self.dim[0]).at[0,0].set(e_11[0,0])

            pre_strain = np.array([[1.,0.],[0.,0.]]) * e_11

            # incorporate the prestrain into the model
            epsilon_pre_strain = epsilon + pre_strain
            
            sigma = lmbda * np.trace(epsilon_pre_strain) * np.eye(self.dim[0]) + 2 * mu * epsilon_pre_strain
            return sigma
        return stress

class LinearElastic_Traction(Problem):
    
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
