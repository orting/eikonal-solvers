import os
import timeit
import numpy as np
from eikonal_solvers.fim import fim, fim_sequential
from test_data import point_source
import matplotlib.pyplot as plt
    
def setup_fim(n, ndim):
    X = point_source(n, ndim)
    F = np.full_like(X, 1)
    Delta = 1
    return {'X' : X, 'F' : F, 'Delta' : 1}
    
if __name__ == '__main__':
    outdir = 'out'
    os.makedirs(outdir, exist_ok=True)
    ns = [50, 250] #, 500, 1000]
    s_max_ns = 250
    min_res = []
    min_pres = []
    p_n_points = []
    s_n_points = []
    for n in ns:
        params = setup_fim(n, 2)
        if n <= s_max_ns:
            s_n_points.append(params['X'].size)
            res = timeit.repeat('fim_sequential(**params)',
                                repeat=3,
                                number=1,
                                globals=globals())
            print(sorted(res))
            min_res.append(min(res))
            
        p_n_points.append(params['X'].size)
        pres = timeit.repeat('fim(**params)',
                             repeat=3,
                             number=1,
                             globals=globals())
        print(sorted(pres))
        min_pres.append(min(pres))
        
    p_n_points = np.array(p_n_points)
    s_n_points = np.array(s_n_points)
    min_res = np.array(min_res)
    min_pres = np.array(min_pres)
                
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1,2,1)    
    fline, = ax.plot(s_n_points/1e6, min_res, 'k*--', )
    fline.set_label('Sequential')
    pline, = ax.plot(p_n_points/1e6, min_pres, 'ro-')
    pline.set_label('Parallel')
    ax.legend()
    ax.set_xlabel('(Number of points)/10**6')
    ax.set_ylabel('Runtime in seconds')

    
    ax = fig.add_subplot(1,2,2)
    fline, = ax.plot(s_n_points/1e6, 1e3*min_res/s_n_points, 'k*--', )
    fline.set_label('Sequential')
    pline, = ax.plot(p_n_points/1e6, 1e3*min_pres/p_n_points, 'ro-')
    pline.set_label('Parallel')
    ax.legend()
    ax.set_xlabel('Number of points/10**6')
    ax.set_ylabel('(Runtime in milliseconds)/(number of points)')

    plt.suptitle('FIM run time with increasing grid size')
    plt.savefig(os.path.join(outdir, 'benchmark_fim_2d.png'))

