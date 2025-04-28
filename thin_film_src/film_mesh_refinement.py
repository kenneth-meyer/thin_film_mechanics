"""
    Exploring mesh refinement with thin films

    Great idea: take existing functions from cardiax!    
"""

from thin_film_src.thinfilm_inputfile_handler import create_2d_mesh
import numpy as onp
import meshio

import jax
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from functools import partial

def custom_rectangle_mesh(Nx, Ny, domain_x_0, domain_x, domain_y_0, domain_y):
    dim = 2
    x = onp.linspace(domain_x_0, domain_x, Nx + 1)
    y = onp.linspace(domain_y_0, domain_y, Ny + 1)
    xv, yv = onp.meshgrid(x, y, indexing='ij')
    points_xy = onp.stack((xv, yv), axis=dim)
    points = points_xy.reshape(-1, dim)
    points_inds = onp.arange(len(points))
    points_inds_xy = points_inds.reshape(Nx + 1, Ny + 1)
    inds1 = points_inds_xy[:-1, :-1]
    inds2 = points_inds_xy[1:, :-1]
    inds3 = points_inds_xy[1:, 1:]
    inds4 = points_inds_xy[:-1, 1:]
    cells = onp.stack((inds1, inds2, inds3, inds4), axis=dim).reshape(-1, 4)
    out_mesh = meshio.Mesh(points=points, cells={'quad': cells})
    return out_mesh

def combine_meshes(film_mesh, substrate_mesh, H):
    """
        Stacks a film mesh on top of a substrate mesh!

        Makes a lot of assumptions about how this works...
    """

    # first, find the points that lie on the film-substrate boundary

    film_pts = film_mesh.points[onp.isclose(film_mesh.points[:,1], H)]
    sub_pt_idx = onp.where(onp.isclose(substrate_mesh.points[:,1], H))[0]

    # check that there are more than 0
    assert(len(film_pts) > 0)
    assert(len(sub_pt_idx) > 0)

    # find where the points are the same
    film_pts_on_sub = []
    film_sub_overlap_inds = []
    
    def allclose_pt(point1, point2):
        return onp.allclose(point1, point2)

    for pt in sub_pt_idx:
        # check where the point on the substrate mesh is on the film mesh
        idx = onp.where(onp.sum(film_mesh.points == substrate_mesh.points[pt], axis=1) == 2)[0][0]
        film_pts_on_sub.append(idx)
        # film_pts_on_sub.append(onp.where(jax.vmap(allclose_pt,in_axes=(0, None))(film_mesh.points, substrate_mesh.points[pt]))[0])

    # should have the same length
    assert(len(film_pts_on_sub) == len(sub_pt_idx))

    # we now have a tuple of points that are coupled/the same

    # now, assign new point inds to the film
    film_new_point_inds = []
    # likely won't have the same ordering but that's ok
    film_new_point_inds.append(film_sub_overlap_inds)

    # determine the unique points on the film
    unique_film_points = onp.delete(film_mesh.points, onp.array(film_pts_on_sub), axis=0)

    # create new points using unique points
    new_points = onp.vstack((substrate_mesh.points, unique_film_points))

    # sift through cells in the film to obtain new point tuples
    # def shuffle_film_cell_inds(cell_pts):
    # ^ looping, even vmapping, could be super expensive
    def find_pt_loc(pt):
        return onp.where(new_points == pt)[0]
    
    film_cells_new = []
    film_cells_old = film_mesh.cells_dict['quad']
    # might take crazy long, trying to shift is SUCH a better idea...
    for cell in film_cells_old:
        pts = film_mesh.points[onp.array(cell, dtype=onp.int32)]
        point_loc = []
        for pt in pts:
            point_loc.append(onp.where(new_points == pt)[0][0])
        film_cells_new.append(point_loc)

    
    # film_cells_new = jax.vmap(jax.vmap(find_pt_loc))(film_mesh.points[film_mesh.cells])

    # append the two lists of cells together
    cells_new = substrate_mesh.cells.append(film_cells_new)

    # createa new mesh
    film_substrate_mesh = meshio.Mesh(new_points, cells={'quad':cells_new})

    return film_substrate_mesh


if __name__ == "__main__":

    H = 1.0
    h = 0.001
    
    num_substrate_cells = 4
    Nx, Ny = 10,10 # this is just 1 cell thick for the substrate, need
    Lx, Ly = H, H + h
        
    # mesh_obj, ele_type = create_2d_mesh(Nx, Ny, Lx, Ly, deg=1)

    # remember to remove the points and cells that already exist in the top layer?
    film_mesh = custom_rectangle_mesh(int( H / (h/num_substrate_cells)), num_substrate_cells, 0, H, H, H+h)
    substrate_mesh = custom_rectangle_mesh(100, 100, 0, H, 0, H)

    # combine the meshes!
    mesh = combine_meshes(film_mesh, substrate_mesh, H)

    # save the mesh to a file to make sure that nothing is super broken
    meshio.write("film_substrate_refinement_test.vtu", mesh)

    # note: it would be nice to slowly refine the mesh; check this out.




