# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 14:58:43 2018

@author: maple
"""

import numpy as np

# %%

def hypercubeToSimplex(z, r):
    """
    Takes a length-n vector with components between 0-1 (e.g. sampled from an n-dimensional
    hypercube), and a m x (n+1) dimensional array consisting of the (n+1) vertices
    of the simplex each with m specified coordinates

    Parameters:
    z : array (n-elements)
         vector with components between 0 and 1
    r : array m x (n+1)  (NB: usually m=n)
        vertices of the simplex to be sampled.
    """
    z_sorted = np.concatenate(([0], np.sort(z), [1]))
    bary_coords = np.diff(z_sorted)
    return np.dot(r, bary_coords)


def hypercubeToHermiteSampleFunction(a0_max, a1_limit, a2_limit):
    """
    Returns a function that uniformly maps the hypercube to a distribution on the hermite coefficients satisfying
    a0 < a0_max
    a1 < a0*a1_limit
    a2 < a0*a2_limit
    
    Note this forms a pyramid with square cross sections in the a1-a2 plane. By
    cutting along the line a1/a1_limit = a2/a2_limit, this forms two simplexes.
    
    The strategy is to slice the hypercube according to the volume of the resulting simplices, then remap
    this to two hypercubes in a continuous way (so the hypercubes share a face along the cut plane).
    Then, each of the two hypercubes is independently transformed into a simplex
    using hypercubeToSimplex, such that the shared face of the hypercubes gets mapped
    into the shared face of the simplices.
    """
    
    # Each simplex is a tetrahedron with 4 vertices, one at the origin. Only the
    # 3rd vertex differs between the two simplices
    r0 = [0,0,0]
    r1 = [a0_max, a1_limit*a0_max, a2_limit*a0_max]
    r2 = [a0_max, -a1_limit*a0_max, -a2_limit*a0_max]
    r31 = [a0_max, a1_limit*a0_max, -a2_limit*a0_max]
    r32 = [a0_max, -a1_limit*a0_max, a2_limit*a0_max]
    
    # The volume of the tetrahedra is calculated as 1/6 of the volume of the parallelepiped
    # formed by the vectors pointing to the three non-zero coordinates
    vol1 = np.abs(np.linalg.det(np.array([r1, r2, r31]))/6)
    vol2 = np.abs(np.linalg.det(np.array([r1, r2, r32]))/6)
    
    # The cutpoint is from [0,1] in one dimension such that the ratio of the cut hypercube volumes is
    # equal to the ratio of the simplex volumes
    cutpoint = vol1/(vol1+vol2)
    
    # Sets of vertices. Note that the vertex that differs is in the first slot.
    # This is because z[0]==0 is the face that the hypercubes will share, and so
    # the last vertex is the one that will get set to 0
    tet1 = np.stack((r31, r2, r1, r0)).T
    tet2 = np.stack((r32, r2, r1, r0)).T
    
    def hypercubeToHermiteSample(z):
        if z[0] <= cutpoint:
            z0_new = 1.0 - (z[0] / cutpoint)
            return hypercubeToSimplex(np.array([z0_new, z[1], z[2]]), tet1)
        else:
            z0_new = (z[0]-cutpoint)/(1.0-cutpoint)
            return hypercubeToSimplex(np.array([z0_new, z[1], z[2]]), tet2)
    
    return hypercubeToHermiteSample

# %% Plot an example of the simplex function

"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

f = hypercubeToHermiteSampleFunction(1e5, 0.6, 0.6)

z = np.random.rand(3, 1000)

vals = np.apply_along_axis(f, 0, z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(vals[1,:], vals[2,:], vals[0,:])
ax.set_xlabel('a1')
ax.set_ylabel('a2')
ax.set_zlabel('a0')
plt.axis('square')
"""
