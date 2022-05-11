import time
import numpy as np
from eikonal_solvers.fim import fim, fim_sequential
from test_data import point_source
import matplotlib.pyplot as plt

def Jeong2008_fig5_example2():
    '''Example 2 in Fig. 5 in 
    Won-Ki Jeong and Ross T. Whitaker
    A Fast Iterative Method for Eikonal Equations
    SIAM Vol. 30, No. 5, pp. 2512–2534 (2008)

    Shows distance calculations for three layers with different speed
    '''
    
    N = 201
    delta = 1
    idx0,idx1 = N-N//10, N//2
    I = np.zeros((N, N))
    I[idx0,idx1] = 1
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
    dI_fim = distance_transform_fim(I, f, (delta, delta))
    elapsed_seq = time.time() - start_seq

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

    plt.suptitle('Fig.5 Example 2 in\n'
                 'Jeong and Whitaker\n'
                 '"A Fast Iterative Method for Eikonal Equations"\n'
                 'using a sequential implementation. Runtime %.3E seconds' % elapsed_seq)
    plt.show()

    # Parallel implementation
    start_par = time.time()
    dI_fim_parallel = distance_transform_fim_parallel(I, f, delta)
    elapsed_par = time.time() - start_par
    
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

    plt.suptitle('Fig.5 Example 2 in\n'
                 'Jeong and Whitaker\n'
                 '"A Fast Iterative Method for Eikonal Equations"\n'
                 'using a parallel implementation. Runtime %.3E seconds' % elapsed_par)
    plt.show()

    # Difference between sequential and parallel
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
    plt.show()
    
