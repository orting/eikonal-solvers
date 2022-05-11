import os
import time
import numpy as np
import matplotlib.pyplot as plt
from eikonal_solvers.fim import fim, fim_sequential

def jeong2008fast_fig5_example2():
    '''Example 2 in Fig. 5 in 
    Won-Ki Jeong and Ross T. Whitaker
    A Fast Iterative Method for Eikonal Equations
    SIAM Vol. 30, No. 5, pp. 2512–2534 (2008)

    Shows distance calculations for three layers with different speed
    '''
    outdir = 'out'    
    os.makedirs(outdir, exist_ok=True)
    figsize = (10,12)
    figsize_diff = (18,12)
    N = 201
    delta = 1
    idx0,idx1 = N-N//10, N//2
    I = np.ones((N, N))
    I[idx0,idx1] = 0
    f = np.empty_like(I)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if 4*i-j < N//3:
                f[i,j] = 3
            elif 4*i+j < 3*N:
                f[i,j] = 2
            else:
                f[i,j] = 1

    # Sequential implementation
    start_seq = time.time()
    dI_fim = fim_sequential(I, f, (delta, delta))
    elapsed_seq = time.time() - start_seq

    plt.figure(figsize=figsize)
    plt.subplot(2,2,1)
    plt.imshow(I, cmap='gray')
    plt.title('Single pixel source')

    plt.subplot(2,2,2)
    plt.imshow(f)
    plt.title('Velocity image ($\Phi$)')

    plt.subplot(2,2,3)
    plt.imshow(dI_fim)
    plt.title('Distance field using sequential FIM ($F_s$)')

    plt.subplot(2,2,4)
    plt.imshow(f)
    plt.contour(dI_fim, levels=range(N//20,idx0,N//20))
    plt.title('Contours of $F_s$ on $\Phi$')

    plt.suptitle('Fig.5 Example 2 in: Jeong and Whitaker\n'
                 '"A Fast Iterative Method for Eikonal Equations"\n'
                 'using a sequential implementation. Runtime %.3E seconds' % elapsed_seq)
    plt.savefig(os.path.join(outdir, 'jeong2008fast_fig5_example2_sequential.png'))
    plt.show()

    # Parallel implementation
    start_par = time.time()
    dI_fim_parallel = fim(I, f, delta)
    elapsed_par = time.time() - start_par

    plt.figure(figsize=figsize)
    plt.subplot(2,2,1)
    plt.imshow(I, cmap='gray')
    plt.title('Single pixel source')

    plt.subplot(2,2,2)
    plt.imshow(f)
    plt.title('Velocity image ($\Phi$)')

    plt.subplot(2,2,3)
    plt.imshow(dI_fim_parallel)
    plt.title('Distance field using parallel FIM ($F_p$)')

    plt.subplot(2,2,4)
    plt.imshow(f)
    plt.contour(dI_fim_parallel, levels=range(N//20,idx0,N//20))
    plt.title('Contours of $F_p$ on $\Phi$')

    plt.suptitle('Fig.5 Example 2 in: Jeong and Whitaker\n'
                 '"A Fast Iterative Method for Eikonal Equations"\n'
                 'using a parallel implementation. Runtime %.3E seconds' % elapsed_par)
    plt.savefig(os.path.join(outdir, 'jeong2008fast_fig5_example2_parallel.png'))
    plt.show()

    # Difference between sequential and parallel
    plt.figure(figsize=figsize_diff)
    plt.subplot(1,3,1)
    plt.imshow(dI_fim)
    plt.title('Distance field using sequential FIM ($F_s$)')

    plt.subplot(1,3,2)
    plt.imshow(dI_fim_parallel)
    plt.title('Distance field using parallel FIM ($F_p$)')

    plt.subplot(1,3,3)
    diff = dI_fim - dI_fim_parallel
    plt.imshow(diff)
    plt.title('max(|Fs - Fp|) = %.3E' % np.max(np.abs(diff)))

    plt.suptitle('Fig.5 Example 2 in\n'
                 'Jeong and Whitaker\n'
                 '"A Fast Iterative Method for Eikonal Equations"\n'
                 'difference between sequential and parallel implementation')
    plt.savefig(
        os.path.join(outdir,'jeong2008fast_fig5_example2_difference_sequential_parallel.png')
    )
    plt.show()
    
if __name__ == '__main__':
    jeong2008fast_fig5_example2()
