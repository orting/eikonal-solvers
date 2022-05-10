'''Implementation of fast marching method (FMM). The implementations is based on

    @article{jones20063d,
    title={3D distance fields: A survey of techniques and applications},
    author={Jones, Mark W and Baerentzen, J Andreas and Sramek, Milos},
    journal={IEEE Transactions on visualization and Computer Graphics},
    volume={12},
    number={4},
    pages={581--599},
    year={2006},
    publisher={IEEE}
    }

and
    @article{rickett1999second,
    title={A second-order fast marching eikonal solver},
    author={Rickett, James and Fomel, Sergey},
    journal={Stanford Exploration Project Report},
    volume={100},
    pages={287--293},
    year={1999}
    }

See also 
    https://github.com/thinks/fast-marching-method 
for a nice description of FMM and a C++ implementation.
'''
# pylint: disable=invalid-name, missing-function-docstring, too-many-arguments, too-many-locals
import heapq
import math
import numpy as np

__all__ = [
    'fmm'
]


def fmm(X, F, Delta=1, max_r=np.inf, force_first_order=False):
    '''Solve the Eikonal equation using the fast marching method.

    Uses second order finite differences. For an isotropic grid, this gives much
    better precision unless grid spacing is very large. For an anisotropic grid
    it will still give better precision, but will underestimate distance along
    the axis of larger spacing.

    If that is the use case, you should compare with `force_first_order=True`
    
    Parameters
    ----------
    X : ndarray
      Calculates distance from X==0 to all other pixels

    F : ndarray, same shape as X
      Speed function. F(x) = 0 implies x cannot be reached.

    Delta : float or sequence of float with len(Delta) == X.ndim (optional)
      Distance between neighboring pixels along each axis

    max_r : float (optional)
      Maximum radius from source to calculate distance for

    force_first_order : bool (optional)
      If True only use first order finite differences.

    Returns
    -------
    ndarray with distance transform, np.inf indicates unreachable indices
      
    '''
    try:
        Delta = (float(Delta), )*X.ndim
    except TypeError:
        pass
    X = np.pad(X, 1, constant_values=1)
    F = np.pad(F, 1, constant_values=0)
    dX = np.full_like(X, np.inf)
    h = []
    ns = neighbors_first_order(X.ndim) 

    def valid(idx):
        return dX[idx] < np.inf
        
    for p in np.argwhere(X == 0):
        p = tuple(p)
        dX[p] = 0
        for ax,n in ns:
            p0 = p[:ax] + (p[ax]+n,) + p[(1+ax):]
            if F[p0] > 0 and X[p0] > 0:
                heapq.heappush(h, (Delta[ax], p0))

    if max_r < np.min(Delta):
        print(max_r, Delta)
        return dX

    while len(h) > 0:
        dp, p = heapq.heappop(h)

        # Skip if already fixed
        if dX[p] < np.inf:
            continue
        
        dX[p] = dp
        for ax,n in ns:
            p0 = p[:ax] + (p[ax]+n,) + p[(1+ax):]
            if F[p0] > 0 and dX[p0] == np.inf:
                dp0 = solve_eikonal(p0, dX, valid, Delta, F[p0], force_first_order)
                if dp0 <= max_r:
                    heapq.heappush(h, (dp0, p0))
    return dX[tuple([slice(1,s-1) for s in dX.shape])]


def solve_eikonal(p0, dX, valid, Delta, f, force_first_order=False, verbose=False):
    '''Solve the Eikonal equation in p0 using the second order stencil if possible.

    Parameters
    ----------
    p0 : tuple of int
      Point to solve for

    dX : nd-array
      Distances calculated so far. `p0` should be a valid index in `dX`

    valid : callable  [tuple of int -> bool]
      Callable that returns true if point should be included in the calculation

    Delta : sequence, length == X.ndim
      Distance between neighboring pixels along each axis

    f : float
      Speed in p0

    verbose : bool (optional)
      If True will print some debug info
    
    Returns
    -------
    dp0 : The distance to p0

    See also
    --------
    solve_eikonal_first_order

    '''
    if force_first_order:
        return solve_eikonal_first_order(p0, dX, valid, Delta, f)
    a = 0
    b = 0
    c = 0
    alpha = [2.25/D**2 for D in Delta] 
    # We get neighbors as (ax, +1, +2), (ax, -1, -2), ...
    for ax, n1, n2 in neighbors_second_order(dX.ndim):
        p1 = get_neighbor(p0, ax, n1) #p0[:ax] + (p0[ax]+n1,) + p0[(1+ax):]
        if valid(p1):
            t = dX[p1]
            p2 = get_neighbor(p0,ax,n2) #p0[:ax] + (p0[ax]+n2,) + p0[(1+ax):]
            if valid(p2):
                t = (4*t - dX[p2])/3
            #print('t', t)
            a += alpha[ax]
            b += alpha[ax]*t
            c += alpha[ax]*t**2
    b = -2*b
    c = c - 1/f**2
    #d = b**2 - 4*a*c
    #
    # At the boundary of the initial shape we can get d < 0. This will happen if p0
    # has two neighbors, x1 and x2, with dX[x1] = 0 and dX[x2] = 1.
    #
    # This can be handled if use a first order estimate. Alternatively, we can safely
    # ignore this calculation and set the distance to infinity as p0 will already be
    # in the queue with distance 1.
    # 
    # A similar issue can occur if there are two neighbors and we only have a second order
    # estimate for one of them, which can happen since we propagate multiple fronts simultaneously
    # For example, 
    # NA  NA   NA   NA  NA
    # NA  NA   x   42.8 NA
    # NA  NA  43.7  NA  NA
    # NA  NA  43.4  NA  NA
    # Assuming unit Delta, we get
    # t = [1/3(4*42.8 - 42.8), 1/3(4*43.7 - 43.4)] = [42.8, 43.8]
    # a = 9/2 = 4.5
    # b = -9/2(42.8 + 43.8) = -389.7
    # c = 9/4(42.8^2 + 43.8^2) = 8437.13
    # d = 389.7^2 - 4*4.5*8437.13 = - 2.25
    # If we instead only use first order estimates we get
    # t = [1/3(4*42.8 - 42.8), 1/3(4*43.7 - 43.7)] = [42.8, 43.7]
    # a = 9/2 = 4.5
    # b = -9/2(42.8 + 43.7) = -389.25
    # c = 9/4(42.8^2 + 43.7^2) = 8417.4425
    # d = 389.25^2 - 4*4.5*8417.4425 = 1.5975
    #
    # We dont want to skip this, so we redo the calculation with only first order info
    try:
        dp0 = solve_quadratic(a,b,c) #(-b+math.sqrt(d))/(2*a)
        if verbose:
            print('Second order', p0, a, b, c, dp0)
    except ValueError:
        if b == -2*sum(alpha):
            if verbose:
                print('Ignored:', p0)
            dp0 = np.inf
        else:
            dp0 = solve_eikonal_first_order(p0, dX, valid, Delta, f)
            if verbose:
                print('First order:', p0, a, b, c, dp0)
    return dp0

def solve_quadratic(a,b,c):
    d = b**2 - 4*a*c
    return (-b+math.sqrt(d))/(2*a)

def get_neighbor(p0, ax, n):
    return p0[:ax] + (p0[ax]+n,) + p0[(1+ax):]


def solve_eikonal_first_order(p0, dX, valid, Delta, f):
    a = 0
    b = 0
    c = 0
    alpha = [1/D**2 for D in Delta]
    for ax, n in neighbors_first_order(dX.ndim):
        pn = p0[:ax] + (p0[ax]+n,) + p0[(1+ax):]
        if valid(pn):
            t = dX[pn]
            a += alpha[ax]
            b += alpha[ax]*t
            c += alpha[ax]*t**2
    b *= -2 
    c -= 1/f**2
    d = b**2 - 4*a*c
    return (-b+math.sqrt(d))/(2*a)


neighbors_first_order_1d = [
    (0, 1),
    (0,-1)
]

neighbors_first_order_2d = [
    (0, 1),
    (0,-1),
    (1, 1),
    (1,-1),
]
neighbors_first_order_3d = [
    (0, 1),
    (0,-1),
    (1, 1),
    (1,-1),
    (2, 1),
    (2,-1),
]
def neighbors_first_order(ndim):
    '''Neighbor offsets for first order calculations

    Parameters
    ----------
    ndim : int
           Dimensionality of space

    Returns
    -------
    list of (axis, offset) pairs.
    '''
    if ndim == 1:
        return neighbors_first_order_1d
    if ndim == 2:
        return neighbors_first_order_2d
    if ndim == 3:
        return neighbors_first_order_3d
    raise NotImplementedError('Only ndim <= 3 implemented. Got ndim=', ndim)

neighbors_second_order_1d = [
    (0, 1, 2),
    (0,-1,-2)
]

neighbors_second_order_2d = [
    (0, 1, 2),
    (0,-1,-2),
    (1, 1, 2),
    (1,-1,-2),
]
neighbors_second_order_3d = [
    (0, 1, 2),
    (0,-1,-2),
    (1, 1, 2),
    (1,-1,-2),
    (2, 1, 2),
    (2,-1,-2),
]

def neighbors_second_order(ndim):
    '''Neighbor offsets for second order calculations

    Parameters
    ----------
    ndim : int
           Dimensionality of space

    Returns
    -------
    list of pairs of axis-index, offset 
    Each pair is the offsets in one direction from the center, e.g. x+1 and x+2.
    The smallest offset is first in each pair.
    '''
    if ndim == 1:
        return neighbors_second_order_1d
    if ndim == 2:
        return neighbors_second_order_2d
    if ndim == 3:
        return neighbors_second_order_3d
    raise NotImplementedError('Only ndim <= 3 implemented. Got ndim=', ndim)
