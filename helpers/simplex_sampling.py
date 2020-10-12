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
    vol1 = np.abs(np.linalg.det(np.array([r1, r2, r31]))/6.)
    vol2 = np.abs(np.linalg.det(np.array([r1, r2, r32]))/6.)

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
            z0_new = 1.0 - z[0]/cutpoint
            return hypercubeToSimplex(np.array([z0_new, z[1], z[2]]), tet1)
        else:
            z0_new = (z[0]-cutpoint)/(1.0-cutpoint)
            return hypercubeToSimplex(np.array([z0_new, z[1], z[2]]), tet2)

    return hypercubeToHermiteSample

# TODO unhardcode these paths
prior_bound_means = np.load(r'../data/prior_bound_means.npz')
prior_bound_matrices = np.load(r'../data/prior_bound_matrices.npz')
prior_bound_zeros = np.load(r'../data/prior_bound_zeros.npz')
prior_bound_axes = np.load(r'../data/prior_bound_axes.npz')
prior_bound_inv = {}
for key in prior_bound_matrices:
    transMatrix = prior_bound_matrices[key]
    prior_bound_inv[key] = np.linalg.inv(transMatrix)

def generalizedHypercubeToHermiteSampleFunction(a0_max, n_hermite, scaleFree=True, sqrtPrior=True):
    """
    Similar to the non-generalized version, adds the constraint
    a0 < a0_max

    but constrains the values of an/a0 in a manner that I will explain more fully later.
    n_hermite is the number of hermite polynomials in the fit.
    """

    if n_hermite < 3 or n_hermite > 9:
        raise NotImplementedError('Number of hermite polynomials must be between 3 and 9 inclusive')

    arr_name = 'arr_' + str(n_hermite-3)
    mean = prior_bound_means[arr_name]
    transMatrix = prior_bound_matrices[arr_name]
    zero = prior_bound_zeros[arr_name]
    axes = prior_bound_axes[arr_name]
    axes = np.maximum(axes, [2.0]*len(axes))

    def hypercubeToHermiteSample(z):
        y = 2.0*z[1:]-1.0
        signY = np.sign(np.sign(y) + 1e-2)
        r = np.diag(axes*signY, k=-1)
        r = r[1:,:]

        x = zero + hypercubeToSimplex(np.abs(y), r)
        if scaleFree:
            if sqrtPrior:
                ap = np.sqrt(a0_max)
                if ap < 3:
                    a0 = (z[0]*4)**2
                else:
                    a0 = ((z[0] - 0.75)*4 + ap)**2
            else:
                a0 = np.exp((z[0] - 0.7)*15 + np.log(np.max((np.nan_to_num(a0_max), 1e-3))))

            theta = np.concatenate([[1.0], np.dot(transMatrix, x) + mean]) * a0
        else:
            theta = np.concatenate([[1.0], np.dot(transMatrix, x) + mean]) * a0_max * z[0]
        return theta

    return hypercubeToHermiteSample

def generalizedHypercubeConstraintFunction(cind, n_hermite, bound=1.0):
    """
    TODO: Documentation
    """
    if n_hermite < 3 or n_hermite > 9:
        raise NotImplementedError('Number of hermite polynomials must be between 3 and 9 inclusive')

    arr_name = 'arr_' + str(n_hermite-3)
    mean = prior_bound_means[arr_name]
    invMatrix = prior_bound_inv[arr_name]
    zero = prior_bound_zeros[arr_name]
    axes = prior_bound_axes[arr_name]
    axes = np.maximum(axes, [2.0]*len(axes))

    hypercubeConstraint = lambda theta: (
        bound - np.sum(
            np.abs( np.dot(invMatrix,
                           theta[cind+1:cind+n_hermite]/(theta[cind]+1e-4) - mean) - zero
            )/ axes
        )
    )

    return hypercubeConstraint

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
