import numpy as np

def point_source(n=50, ndim=2):
    '''A single point in the center of an (2n+1)x(2n+1) domain

    n : int (optional)
      Grid size. Same for all dimensions
    '''
    N = 2*n + 1
    I = np.ones((N, )*ndim)
    I[(n,)*ndim] = 0
    
    return I


def example4(N=101):
    '''Four bars with different length and axis alignment

    N : int
      Size of grid. Same for all dimensions
    '''
    idx0,idx1 = N//4, 3*N//4
    I = np.zeros((N, N))
    I[idx0-1:idx0+2,idx1] = I[idx1-2:idx1+3,idx0] = I[idx0,idx0-3:idx0+4] = I[idx1,idx1-5:idx1+6] = 1
    
    F = np.full_like(I, 1)
    Delta = 1

    return I, F, Delta
