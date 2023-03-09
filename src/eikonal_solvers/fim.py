'''Implementation of the (Improved) Fast Iterative Method (FIM) for solving eikonal equations
based on

@article{huang2021improved,
  title={Improved Fast Iterative Algorithm for Eikonal Equation for GPU Computing},
  author={Huang, Yuhao},
  journal={arXiv preprint arXiv:2106.15869},
  year={2021}
}

and

@article{jeong2008fast,
  title={A fast iterative method for eikonal equations},
  author={Jeong, Won-Ki and Whitaker, Ross T},
  journal={SIAM Journal on Scientific Computing},
  volume={30},
  number={5},
  pages={2512--2534},
  year={2008},
  publisher={SIAM}
}

'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments, too-many-statements, too-many-locals, too-many-branches
# pylint: disable=too-many-nested-blocks
# pylint: disable=missing-function-docstring
import warnings
import numpy as np

__all__ = [
    'binary_dilation',
    'fim',
    'fim_sequential'
]

def binary_dilation(X, radius):
    '''Binary dilation of X by ball(radius) using FIM method

    Parameters
    ----------
    X : ndarray
      Dilates X != 0
    
    radius : float
      Radius of ball to dilate with

    Returns
    -------
    binary ndarray of same shape as X with dilation of X
    '''
    max_iter = 2*radius + 1 # Seems like a good number...
    F = np.ones_like(X, dtype=float)
    return fim(np.logical_not(X), F, max_iter=max_iter) <= radius


def fim(X, F, Delta=1, epsilon=1e-12, max_iter=np.inf, dtype=np.float64, verbose=False):
    '''Solve the Eikonal equation using the improved fast iterative method.

    Uses first order finite differences. 
    
    Parameters
    ----------
    X : ndarray
      Calculates distance from X==0 to all other pixels

    F : ndarray, same shape as X
      Speed function. F(x) = 0 implies x cannot be reached.

    Delta : float or sequence of float with len(Delta) == X.ndim (optional)
      Distance between neighboring pixels along each axis

    epsilon : float (optional)
      A point is converged if its abs(new value - old value) < epsilon

    max_iter : int (optional)
      Maximum number of iterations. If Delta is isotropic, then this can be used to control the
      maximum distance to calculate as max_iter * Delta
      For large X this can decrease run time significantly. But be aware that it will not necesarily
      mean that all pixels within that distance have converged.

    verbose : bool (optional)
      If True print some information

    Returns
    -------
    ndarray with distance transform, np.inf indicates unreachable indices
      
    '''
    if epsilon < 1e-5 and dtype == np.float32:
        warnings.warn('Using epsilon < 1e-5 and float32 could lead to numerical instability. '
                      'Forcing dtype float64')
        dtype = np.float64
    try:
        Delta = (float(Delta), )* X.ndim
    except TypeError:
        pass

    X = np.pad(X, 1, constant_values=1)
    F = np.pad(F, 1, constant_values=0)
    phi = np.full_like(X, np.inf, dtype=dtype)
    source = np.nonzero(X == 0)
    phi[source] = 0
    active = np.zeros_like(phi, dtype=bool)
    Delta2 = tuple([d**2 for d in Delta])
    if X.ndim == 2:
        ns = (
            np.stack((source[0]-1, source[0]+1, source[0]  , source[0]  )),
            np.stack((source[1]  , source[1]  , source[1]-1, source[1]+1))
        )
    elif X.ndim == 3:
        ns = (
            np.stack((source[0]-1, source[0]+1, source[0]  , source[0]  , source[0]  , source[0] )),
            np.stack((source[1]  , source[1]  , source[1]-1, source[1]+1, source[1]  , source[1] )),
            np.stack((source[2]  , source[2]  , source[2]  , source[2]  , source[2]-1, source[2]+1))
        )
    else:
        raise NotImplementedError('Only 2d and 3d domains supported')
        
    use = np.logical_and(phi[ns] > 0, F[ns] > 0)
    ns = tuple([ax[use] for ax in ns])
    active[ns] = True

    iteration = 0
    while iteration < max_iter:
        iteration += 1
        current = np.nonzero(active)
        if len(current[0]) == 0:
            break

        if X.ndim == 2:
            new_phi, ns = update_fim_2d_parallel(*current, phi, F, *Delta, *Delta2)
        else:
            new_phi, ns = update_fim_3d_parallel(*current, phi, F, *Delta)
        old_phi = phi[current]
        phi[current] = new_phi

        # Check which points have converged
        # Converged points are no longer active and their neighbors should be considered as
        # new active points            
        converged = np.abs(old_phi - new_phi) < epsilon
        current_converged = tuple([ax[converged] for ax in current])
        active[current_converged] = False
        ns = tuple([ax[:,converged] for ax in ns])

        # We never consider source points (phi == 0) or points with zero speed (F == 0)
        use = np.logical_and(phi[ns] > 0, F[ns] > 0)
        ns = tuple([ax[use] for ax in ns])

        # If we have a point that was already active and is in the neighbor set of a
        # converged we need to consider what to do it.
        #
        # If the point is still active we do nothing, since it will be recomputed in the next
        # iteration
        #
        # If the point converged it means it is the neighbor of a point that also converged
        # but this means that the point responsible for adding the new point did not change,
        # so the point has already been calculated with the converged neighborhood.
        # Thus we should not include it in the active list.
        candidates = np.zeros_like(phi)        
        candidates[ns] = 1
        candidates[current] = 0
        ns = np.nonzero(candidates)

        # For each potential new active point, we check if the update decrease distance
        # and if so we apply the update and add them as new active points.
        if X.ndim == 2:
            ns_phi,_ = update_fim_2d_parallel(*ns, phi, F, *Delta, *Delta2)
        else:
            ns_phi,_ = update_fim_3d_parallel(*ns, phi, F, *Delta)
        use = (phi[ns] - ns_phi) > epsilon
        ns = tuple([ax[use] for ax in ns])
        active[ns] = True
        phi[ns] = ns_phi[use]

    if verbose and iteration == max_iter:
        print('Reached max iterations before convergence', iteration)

    return phi[tuple([slice(1,s-1) for s in X.shape])]

def update_fim_2d_parallel(y, x, phi, F, dy, dx, dy2, dx2):
    '''Algorithm 1.2 in \cite{huang2021improved}.
    Note that there is a mistake in Algorithm 1.2, the criteria for choosing how to update
    should be
    \begin{equation}
    \phi^{minx} - \phi^{miny} > \frac{\Delta y}{f} \implies 
      \phi_{ij} = \phi^{miny} + \frac{\Delta y}{f}
    \end{equation}

    instead of
    \begin{equation}
    \phi^{minx} - \phi^{miny} > \sqrt{\frac{\Delta y^ + \Delta x^2}{f^2}} \implies 
      \phi_{ij} = \phi^{miny} + \frac{\Delta y}{f}
    \end{equation}

    The point is to ensure that we go directly from a neighbor to the current point,
    if the distance through that neighbor is smaller than any path that also pass
    through a neighbor in the orthogonal direction.
    
    '''
    f = F[y,x]
    
    ny = np.stack((y-1, y+1, y  , y  ))
    nx = np.stack((x  , x  , x-1, x+1))
    ns = np.stack((ny,nx), axis=-1)

    min_y = np.minimum(phi[ns[0,:,0],ns[0,:,1]], phi[ns[1,:,0],ns[1,:,1]])
    min_x = np.minimum(phi[ns[2,:,0],ns[2,:,1]], phi[ns[3,:,0],ns[3,:,1]])
    diff = min_y - min_x
    dt1 =  dx/f
    dt2 = -dy/f

    idx1 = diff >= dt1
    idx2 = diff <= dt2
    idx3 = np.logical_not(np.logical_or(idx1, idx2))

    new_phi = np.empty_like(diff)
    new_phi[idx1] = min_x[idx1] + dx/f[idx1]
    new_phi[idx2] = min_y[idx2] + dy/f[idx2]
    new_phi[idx3] = (
        min_y[idx3]*dx2 +
        min_x[idx3]*dy2 +
        dy*dx*np.sqrt((dy2+dx2)/f[idx3]**2 - (min_y[idx3] - min_x[idx3])**2)
    )/(dy2 + dx2)
    
    return new_phi, (ny, nx)

def update_fim_3d_parallel(z, y, x, phi, F, dz, dy, dx):
    f = F[z,y,x]

    nz = np.stack((z-1, z+1, z  , z  , z  , z  ))
    ny = np.stack((y  , y  , y-1, y+1, y  , y  ))
    nx = np.stack((x  , x  , x  , x  , x-1, x+1))
    ns = np.stack((nz, ny,nx), axis=-1)

    min_z = np.minimum(phi[ns[0,:,0],ns[0,:,1],ns[0,:,2]], phi[ns[1,:,0],ns[1,:,1], ns[1,:,2]])
    min_y = np.minimum(phi[ns[2,:,0],ns[2,:,1],ns[2,:,2]], phi[ns[3,:,0],ns[3,:,1], ns[3,:,2]])
    min_x = np.minimum(phi[ns[4,:,0],ns[4,:,1],ns[4,:,2]], phi[ns[5,:,0],ns[5,:,1], ns[5,:,2]])

    zy = min_z <= min_y
    zx = min_z <= min_x
    yz = ~zy
    yx = min_y <= min_x
    xz = ~zx
    xy = ~yx

    zc = zy & zx
    yc = yz & yx
    xc = xz & xy

    zb = (yc & zx) | (xc & zy)
    yb = (zc & yx) | (xc & yz)
    xb = (zc & xy) | (yc & xz)

    za = ~(zc | zb)
    ya = ~(yc | yb)
    xa = ~(xc | xb)
    
    c,b,a = np.empty_like(f), np.empty_like(f), np.empty_like(f)
    c[zc], c[yc], c[xc] = min_z[zc], min_y[yc], min_x[xc]
    b[zb], b[yb], b[xb] = min_z[zb], min_y[yb], min_x[xb]
    a[za], a[ya], a[xa] = min_z[za], min_y[ya], min_x[xa]

    dc,db,da = np.empty_like(f), np.empty_like(f), np.empty_like(f)
    dc[zc], dc[yc], dc[xc] = dz, dy, dx
    db[zb], db[yb], db[xb] = dz, dy, dx
    da[za], da[ya], da[xa] = dz, dy, dx
    dc2, db2, da2 = dc**2, db**2, da**2
    
    new_phi = np.empty_like(f)
    
    u = c + dc/f
    idx1 = u <= b
    new_phi[idx1] = u[idx1]
    
    i = ~idx1
    u[i] = (
        b[i]*dc2[i] +
        c[i]*db2[i] +
        dc[i]*db[i]*np.sqrt((db2[i] + dc2[i])/f[i]**2 - (b[i]- c[i])**2)
    )/(db2[i] + dc2[i])
    idx2 = i & (u <= a)
    new_phi[idx2] = u[idx2]

    i = ~(idx1 | idx2)
    A = 1/da2[i] + 1/db2[i] + 1/dc2[i]
    B = a[i]/da2[i] + b[i]/db2[i] + c[i]/dc2[i]
    C = a[i]**2/da2[i] + b[i]**2/db2[i] + c[i]**2/dc2[i] - 1/f[i]**2
    new_phi[i] = (B + np.sqrt(B**2 - A*C))/A

    return new_phi, (nz, ny, nx)
    

def fim_sequential(X, F, Delta=1, epsilon = 1e-12, dtype=np.float64, max_iter=np.inf):
    '''Sequential implementation of FIM for reference. Do NOT use for computations as it is much
    slower than FIM for the same output.
    '''
    try:
        Delta = (float(Delta), )* X.ndim
    except TypeError:
        pass        

    if X.ndim == 2:
        offsets = [
            np.array((0,-1)), np.array((-1,0)),
            np.array((0, 1)), np.array(( 1,0)),
        ]
    else:
        offsets = [
            np.array((0,0,-1)), np.array((0,-1,0)), np.array((-1,0,0)),
            np.array((0,0, 1)), np.array((0, 1,0)), np.array(( 1,0,0)),
        ]

    xsize = np.array(X.shape)
    
    source = np.nonzero(X == 0)
    phi = np.full_like(X, np.inf, dtype=dtype)
    phi[source] = 0
    active = []
    for x in zip(*source):
        for offset in offsets:
            xn = x + offset
            if np.all(xn >= 0) and np.all(xn < xsize):
                xn = tuple(xn)
                if phi[xn] > 0 and xn not in active:
                    active.append(xn)

    iteration = 0
    while len(active) > 0 and iteration < max_iter:
        iteration += 1
        # We do not want to apply updates before all active points have been processed
        # otherwise we introduce a causal relationship
        updates = []

        # Potential new active points are stored, such that we can calculate their updates
        # after all active points have been updated.
        candidates = []
        new_active = []
        for x in active:
            p = phi[x]
            if X.ndim == 2:
                q = update_fim_sequential_2d(*x,phi,F,Delta)
            else:
                q = update_fim_sequential_3d(*x,phi,F,Delta)
            updates.append((x, q))
            if np.abs(p-q) < epsilon:
                for offset in offsets:
                    xn = x + offset
                    if np.all(xn >= 0) and np.all(xn < xsize):
                        xn = tuple(xn)
                        # Any point that is not source is a candidate
                        if phi[xn] > 0 and xn not in candidates:
                            candidates.append(xn)
            else:
                new_active.append(x)

        # Apply all the the updates, before checking candidates
        for x,q in updates:
            phi[x] = q

        # Now check the candidates
        updates = []
        for x in candidates:
            # Although candidates are unique, we might have already added a point
            # that was active to the new_active list. We dont want to recompute the
            # update for those points in this iteration
            if not x in new_active:
                if X.ndim == 2:
                    q = update_fim_sequential_2d(*x,phi,F,Delta)
                else:
                    q = update_fim_sequential_3d(*x,phi,F,Delta)
                # If the distance got smaller, x should be active and the update applied
                if phi[x] > q: 
                    updates.append((x, q))
                    new_active.append(x)
                    
        # Apply all updates from candidates
        for x,q in updates:
            phi[x] = q

        active = new_active
        
    return phi

def update_fim_sequential_2d(i, j, phi, F, Delta):
    f = F[i,j]
    di, dj = Delta
    di2, dj2 = di**2, dj**2
    if i == 0:
        min_i = phi[i+1,j]
    elif i + 1 == phi.shape[0]:
        min_i = phi[i-1,j]
    else:
        min_i = min(phi[i-1,j], phi[i+1,j])

    if j ==  0:
        min_j = phi[i, j+1]
    elif j + 1 == phi.shape[1]:
        min_j = phi[i, j-1]
    else:
        min_j = min(phi[i,j-1], phi[i,j+1])

    diff = min_i - min_j
    if diff >  dj/f:
        return min_j + dj/f
    if diff < -di/f:
        return min_i + di/f
    return (min_i*dj2 + min_j*di2 + di*dj*np.sqrt((di2 + dj2)/f**2 -(min_i - min_j)**2))/(di2 + dj2)


def update_fim_sequential_3d(i, j, k, phi, F, Delta):
    f = F[i,j,k]
    di, dj, dk = Delta

    if i == 0:
        min_i = phi[i+1,j,k]
    elif i + 1 == phi.shape[0]:
        min_i = phi[i-1,j,k]
    else:
        min_i = min(phi[i-1,j,k], phi[i+1,j,k])

    if j ==  0:
        min_j = phi[i, j+1,k]
    elif j + 1 == phi.shape[1]:
        min_j = phi[i, j-1,k]
    else:
        min_j = min(phi[i,j-1,k], phi[i,j+1,k])

    if k == 0:
        min_k = phi[i,j,k+1]
    elif k + 1 == phi.shape[2]:
        min_k = phi[i,j,k-1]
    else:
        min_k = min(phi[i,j,k-1], phi[i,j,k+1])

    if min_i < min_j:
        if min_j < min_k:
            return solve_quadratic(min_i, min_j, min_k, di, dj, dk, f)
        if min_i < min_k:
            return solve_quadratic(min_i, min_k, min_j, di, dk, dj, f)
        return solve_quadratic(min_k, min_i, min_j, dk, di, dj, f)
    
    if min_i < min_k:
        return solve_quadratic(min_j, min_i, min_k, dj, di, dk, f)
    
    if min_j < min_k:
        return solve_quadratic(min_j, min_k, min_i, dj, dk, di, f)
    
    return solve_quadratic(min_k, min_j, min_i, dk, dj, di, f)
        

def solve_quadratic(c,b,a,dc,db,da,f):
    u = c + dc/f
    if u <= b:
        return u
    db2, dc2 = db**2, dc**2    
    u = (b*dc2 + c*db2 + dc*db*np.sqrt((db2 + dc2)/f**2 - (b- c)**2))/(db2 + dc2)
    if u <= a:
        return u
    da2 = da**2
    A = 1/da2 + 1/db2 + 1/dc2 # 3
    B = a/da2 + b/db2 + c/dc2 # (a+b+c)
    C = (a**2/da2 + b**2/db2 + c**2/dc2 - 1/f**2)
    return (B + np.sqrt(B**2 - A*C))/A
