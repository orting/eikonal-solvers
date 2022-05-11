import os
import timeit
import numpy as np
from eikonal_solvers.fmm import fmm
from test_data import point_source
import matplotlib.pyplot as plt
    
def setup_fmm(n, ndim, force_first_order):
    X = point_source(n, ndim)
    F = np.full_like(X, 1)
    Delta = 1
    return {'X' : X, 'F' : F, 'Delta' : 1, 'force_first_order' : force_first_order}
    
if __name__ == '__main__':
    outdir = 'out/benchmarks'
    os.makedirs(outdir, exist_ok=True)
    ns = [50, 250, 500, 1000]
    min_fres = []
    min_sres = []
    n_points = []
    for n in ns:
        params = setup_fmm(n, 2, True)
        n_points.append(params['X'].size)
        fres = timeit.repeat('fmm(**params)',
                            repeat=3,
                            number=1,
                            globals=globals())
        print(sorted(fres))
        min_fres.append(min(fres))
        
        params = setup_fmm(n, 2, False)
        sres = timeit.repeat('fmm(**params)',
                            repeat=3,
                            number=1,
                            globals=globals())
        print(sorted(sres))
        min_sres.append(min(sres))

    n_points = np.array(n_points)
    min_fres = np.array(min_fres)
    min_sres = np.array(min_sres)
        
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1,2,1) 
    fline, = ax.plot(n_points/1e6, min_fres, 'k*--', )
    sline, = ax.plot(n_points/1e6, min_sres, 'ro-')
    fline.set_label('First order')
    sline.set_label('Second order')    
    ax.set_xlabel('(Number of points)/10**6')
    ax.set_ylabel('Runtime in seconds')
    ax.legend()
    
    ax = fig.add_subplot(1,2,2)
    fline, = ax.plot(n_points/1e6, 1e3*min_fres/n_points, 'k*--', )
    sline, = ax.plot(n_points/1e6, 1e3*min_sres/n_points, 'ro-')
    fline.set_label('First order')
    sline.set_label('Second order')    
    ax.set_xlabel('Number of points/10**6')
    ax.set_ylabel('(Runtime in milliseconds)/(number of points)')
    ax.legend()

    plt.suptitle('FMM run time with increasing grid size')
    plt.savefig(os.path.join(outdir, 'benchmark_fmm_2d.png'))

