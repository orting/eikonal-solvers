import os
import time
import numpy as np
import matplotlib.pyplot as plt
from eikonal_solvers.fim import fim
from eikonal_solvers.fmm import fmm

def example5():
    '''Example 5 in \cite{huang2021improved}.
    Shows distance calculations for spatially varying speed F
    '''
    outdir = 'out'    
    os.makedirs(outdir, exist_ok=True)
    figsize = (10,12)
    figsize_diff = (18,12)
    N = 201
    delta = 1
    I = np.ones((N, N))
    I[N//2,N//2] = 0
    F = np.empty_like(I)
    epsilon = 0.3
    domain_y = (-10,10)
    domain_x = (-10,10)
    ys = np.linspace(*domain_y,N)
    xs = np.linspace(*domain_x,N)
    for i,y in enumerate(ys):
        for j,x in enumerate(xs):
            F[i,j] = np.sin(x)**2 + np.sin(y)**2 + epsilon

    start = time.time()
    dI = fim(I, F, delta)
    elapsed = time.time() - start

    # extent is really weird...
    # https://matplotlib.org/3.1.0/tutorials/intermediate/imshow_extent.html
    extent = (*domain_x, domain_y[1], domain_y[0])
    
    plt.figure(figsize=figsize)
    plt.subplot(2,2,1)
    plt.imshow(I, cmap='gray', extent=extent)
    plt.title('Single pixel source')

    plt.subplot(2,2,2)
    plt.imshow(F, extent=extent)
    plt.title('Velocity image')

    plt.subplot(2,2,3)
    plt.imshow(dI, extent=extent)
    plt.title('Distance field using iFIM')

    plt.subplot(2,2,4)
    plt.imshow(I, cmap='gray', extent=extent)
    plt.contour(ys, xs, dI, levels=20)
    plt.title('Contours of distance field')

    plt.suptitle('Example 5 in: Yuhao Huang\n'
                 '"Improved Fast Iterative Algorithm for Eikonal Equation for GPU Computing"\n'
                 'Runtime %.3E seconds' % elapsed)
    plt.savefig(os.path.join(outdir, 'huang2021improved_example5_fim.png'))
    plt.show()

    plt.figure(figsize=(20,20))
    plt.imshow(I, cmap='gray', extent=extent)
    plt.contour(ys, xs, dI, levels=20)
    plt.title('Contours of FIM distance field')
    plt.savefig(os.path.join(outdir, 'huang2021improved_example5_contour_fim.png'))

    plt.show()

    
    # Use FMM second order
    start = time.time()
    dI = fmm(I, F, delta)
    elapsed = time.time() - start
    
    plt.figure(figsize=figsize)
    plt.subplot(2,2,1)
    plt.imshow(I, cmap='gray', extent=extent)
    plt.title('Single pixel source')

    plt.subplot(2,2,2)
    plt.imshow(F, extent=extent)
    plt.title('Velocity image')

    plt.subplot(2,2,3)
    plt.imshow(dI, extent=extent)
    plt.title('Distance field using FMM')

    plt.subplot(2,2,4)
    plt.imshow(I, cmap='gray', extent=extent)
    plt.contour(ys, xs, dI, levels=20)
    plt.title('Contours of distance field')

    plt.suptitle('Example 5 in: Yuhao Huang\n'
                 '"Improved Fast Iterative Algorithm for Eikonal Equation for GPU Computing"\n'
                 'Runtime %.3E seconds' % elapsed)
    plt.savefig(os.path.join(outdir, 'huang2021improved_example5_fmm.png'))
    plt.show()

    plt.figure(figsize=(20,20))
    plt.imshow(I, cmap='gray', extent=extent)
    plt.contour(ys, xs, dI, levels=20)
    plt.title('Contours of FMM distance field')
    plt.savefig(os.path.join(outdir, 'huang2021improved_example5_contour_fmm.png'))
    plt.show()


    
if __name__ == '__main__':
    example5()

    
