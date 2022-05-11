'''Direct torch translation of numpy version in fim.py'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments, too-many-statements, too-many-locals
# pylint: disable=missing-function-docstring
import warnings
import torch
from torch.nn.functional import pad

__all__ = [
    'fim',
]

def fim(X, F, Delta=1, epsilon=1e-12, max_iter=float('inf'), dtype=torch.float32):
    '''
    X : torch.tensor

    F : torch.tensor
    '''
    if epsilon < 1e-5 and dtype == torch.float32:
        warnings.warn('Using epsilon < 1e-5 and float32 could lead to numerical instability. '
                      'Forcing dtype float64')
        dtype = torch.float64
    try:
        Delta = (float(Delta), )* X.ndim
    except TypeError:
        pass

    X = pad(X, (1,1)*X.ndim, mode='constant', value=1)
    F = pad(F, (1,1)*X.ndim, mode='constant', value=0)

    phi = torch.full_like(X, float('inf'), dtype=dtype)
    source = torch.nonzero(X == 0, as_tuple=True)
    phi[source] = 0
    active = torch.zeros_like(phi, dtype=bool)
    Delta2 = tuple([d**2 for d in Delta])
    if X.ndim == 2:
        ns = (
            torch.stack((source[0]-1, source[0]+1, source[0]  , source[0]  )),
            torch.stack((source[1]  , source[1]  , source[1]-1, source[1]+1))
        )
    elif X.ndim == 3:
        # pylint: disable=line-too-long
        ns = (
            torch.stack((source[0]-1, source[0]+1, source[0]  , source[0]  , source[0]  , source[0]  )),
            torch.stack((source[1]  , source[1]  , source[1]-1, source[1]+1, source[1]  , source[1]  )),
            torch.stack((source[2]  , source[2]  , source[2]  , source[2]  , source[2]-1, source[2]+1))
        )
    else:
        raise NotImplementedError('Only 2d and 3d domains supported')

    use = torch.logical_and(phi[ns] > 0, F[ns] > 0)
    ns = tuple([ax[use] for ax in ns])
    active[ns] = True
    
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        current = torch.nonzero(active, as_tuple=True)
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
        converged = torch.abs(old_phi - new_phi) < epsilon
        current_converged = tuple([ax[converged] for ax in current])
        active[current_converged] = False
        ns = tuple([ax[:,converged] for ax in ns])

        # We never consider source points (phi == 0) or points with zero speed (F == 0)
        use = torch.logical_and(phi[ns] > 0, F[ns] > 0)
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
        candidates = torch.zeros_like(phi)        
        candidates[ns] = 1
        candidates[current] = 0
        ns = torch.nonzero(candidates, as_tuple=True)

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

    if iteration == max_iter:
        print('Reached max iterations before convergence', iteration)

    return phi[tuple([slice(1,s-1) for s in X.shape])]

def update_fim_2d_parallel(y, x, phi, F, dy, dx, dy2, dx2):
    f = F[y,x]
    
    ny = torch.stack((y-1, y+1, y  , y  ))
    nx = torch.stack((x  , x  , x-1, x+1))
    ns = torch.stack((ny,nx), axis=-1)

    min_y = torch.minimum(phi[ns[0,:,0],ns[0,:,1]], phi[ns[1,:,0],ns[1,:,1]])
    min_x = torch.minimum(phi[ns[2,:,0],ns[2,:,1]], phi[ns[3,:,0],ns[3,:,1]])
    diff = min_y - min_x
    dt1 =  dx/f
    dt2 = -dy/f
    
    idx1 = diff >= dt1
    idx2 = diff <= dt2
    idx3 = torch.logical_not(torch.logical_or(idx1, idx2))

    new_phi = torch.empty_like(diff)
    new_phi[idx1] = min_x[idx1] + dx/f[idx1]
    new_phi[idx2] = min_y[idx2] + dy/f[idx2]
    new_phi[idx3] = (
        min_y[idx3]*dx2 +
        min_x[idx3]*dy2 +
        dy*dx*torch.sqrt((dy2+dx2)/f[idx3]**2 - (min_y[idx3] - min_x[idx3])**2)
    )/(dy2 + dx2)

    return new_phi, (ny, nx)

def update_fim_3d_parallel(z, y, x, phi, F, dz, dy, dx):
    f = F[z,y,x]

    nz = torch.stack((z-1, z+1, z  , z  , z  , z  ))
    ny = torch.stack((y  , y  , y-1, y+1, y  , y  ))
    nx = torch.stack((x  , x  , x  , x  , x-1, x+1))
    ns = torch.stack((nz, ny,nx), axis=-1)

    min_z = torch.minimum(phi[ns[0,:,0],ns[0,:,1],ns[0,:,2]], phi[ns[1,:,0],ns[1,:,1], ns[1,:,2]])
    min_y = torch.minimum(phi[ns[2,:,0],ns[2,:,1],ns[2,:,2]], phi[ns[3,:,0],ns[3,:,1], ns[3,:,2]])
    min_x = torch.minimum(phi[ns[4,:,0],ns[4,:,1],ns[4,:,2]], phi[ns[5,:,0],ns[5,:,1], ns[5,:,2]])

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
    
    c,b,a = torch.empty_like(f), torch.empty_like(f), torch.empty_like(f)
    c[zc], c[yc], c[xc] = min_z[zc], min_y[yc], min_x[xc]
    b[zb], b[yb], b[xb] = min_z[zb], min_y[yb], min_x[xb]
    a[za], a[ya], a[xa] = min_z[za], min_y[ya], min_x[xa]

    dc,db,da = torch.empty_like(f), torch.empty_like(f), torch.empty_like(f)
    dc[zc], dc[yc], dc[xc] = dz, dy, dx
    db[zb], db[yb], db[xb] = dz, dy, dx
    da[za], da[ya], da[xa] = dz, dy, dx
    dc2, db2, da2 = dc**2, db**2, da**2
    
    new_phi = torch.empty_like(f)
    
    u = c + dc/f
    idx1 = u <= b
    new_phi[idx1] = u[idx1]
    
    i = ~idx1
    u[i] = (
        b[i]*dc2[i] +
        c[i]*db2[i] +
        dc[i]*db[i]*torch.sqrt((db2[i] + dc2[i])/f[i]**2 - (b[i]- c[i])**2)
    )/(db2[i] + dc2[i])
    idx2 = i & (u <= a)
    new_phi[idx2] = u[idx2]

    i = ~(idx1 | idx2)
    A = 1/da2[i] + 1/db2[i] + 1/dc2[i]
    B = a[i]/da2[i] + b[i]/db2[i] + c[i]/dc2[i]
    C = a[i]**2/da2[i] + b[i]**2/db2[i] + c[i]**2/dc2[i] - 1/f[i]**2
    new_phi[i] = (B + torch.sqrt(B**2 - A*C))/A

    return new_phi, (nz, ny, nx)
