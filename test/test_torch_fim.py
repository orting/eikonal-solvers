'''Test eikonal_solvers.torch.fim'''
# pylint: disable=invalid-name, missing-function-docstring, too-many-locals
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from test_data import point_source
from eikonal_solvers.torch.fim import fim

def test_accuracy_with_gridspacing(outdir):
    outpath = 'accuracy_fim_gridspacing_2d_torch.png'
    if len(outdir) > 0:
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, outpath)
    ndim = 2
    n = 25
    I = point_source(n, ndim)
    X = torch.tensor(I) # pylint: disable=not-callable
    F = torch.full_like(X, 1)
    parallel_errors = []
    Deltas = [1,10,20,30,40,50]
    for Delta in Deltas:
        dIedt = distance_transform_edt(I, sampling=Delta)
        M = dIedt <= Delta*20
        dIparallel = fim(X, F, Delta).numpy()
        parallel_errors.append(np.mean(np.abs(dIedt[M] - dIparallel[M])))

    fig = plt.figure(figsize=(12,8.5))
    ax = fig.add_subplot(1,2,1) 
    sline, = ax.plot(Deltas, parallel_errors, 'ro-')
    sline.set_label('Parallel')    
    ax.set_xlabel('Spacing')
    ax.set_ylabel('Mean absolute error')
    ax.legend()
    
    ax = fig.add_subplot(1,2,2)
    sline, = ax.plot(np.log(Deltas), np.log(parallel_errors), 'ro-')
    sline.set_label('Parallel')
    ax.legend()
    ax.set_xlabel('log(Spacing)')
    ax.set_ylabel('log(Mean absolute error)')

    plt.suptitle('FIM (torch) Approximation error with increasing grid spacing')
    plt.savefig(outpath)

    for i in range(1, len(Deltas)):
        assert parallel_errors[i] > parallel_errors[i-1]        
    


def test_accuracy_2d(outdir):
    outpath='accuracy_fim_isotropic_2d_torch.png'
    if len(outdir) > 0:
        outpath = os.path.join(outdir, outpath)
    mean_abs_error, max_abs_error = _test_accuracy(2, outpath=outpath)
    assert mean_abs_error < 0.495
    assert max_abs_error  < 0.919

def test_accuracy_3d(outdir):
    outpath = 'accuracy_fim_isotropic_3d_torch.png'
    if len(outdir) > 0:
        outpath = os.path.join(outdir, outpath)
    mean_abs_error, max_abs_error = _test_accuracy(3, outpath=outpath)
    assert mean_abs_error < 0.92
    assert max_abs_error  < 1.51

def test_accuracy_anisotropic_2d(outdir):
    outpath = 'accuracy_fim_anisotropic_2d_torch.png'
    if len(outdir) > 0:
        outpath = os.path.join(outdir, outpath)
    Delta = (2,0.5)
    mean_abs_error, max_abs_error = _test_accuracy(2, Delta, outpath=outpath)
    assert mean_abs_error < 0.537
    assert max_abs_error  < 1.015    

def test_accuracy_anisotropic_3d(outdir):
    outpath = 'accuracy_fim_anisotropic_3d_torch.png'
    if len(outdir) > 0:
        outpath = os.path.join(outdir, outpath)
    Delta = (1,2,0.5)
    mean_abs_error, max_abs_error = _test_accuracy(3, Delta, outpath=outpath)
    assert mean_abs_error < 0.991
    assert max_abs_error  < 1.645

    
def _test_accuracy(ndim=2, Delta=1, outpath='out_fim_test_torch.png'):
    outdir = os.path.dirname(outpath)
    if len(outdir) > 0:
        os.makedirs(outdir, exist_ok=True)
    try:
        Delta = (float(Delta), *ndim)
    except TypeError:
        pass
    n = 25
    accuracy_radius = 20
    I = point_source(n, ndim)
    X = torch.tensor(I) # pylint: disable=not-callable
    F = torch.full_like(X, 1)

    dI = fim(X, F, Delta, dtype=torch.float64, max_iter=100).numpy()
    dIedt = distance_transform_edt(I, sampling=Delta)

    diff = dI-dIedt
    abs_error = np.abs(diff[dIedt <= accuracy_radius])
    mean_abs_error = np.mean(abs_error)
    max_abs_error = np.max(abs_error)

    if ndim == 3:
        I = I[n]
        dI = dI[n]
        dIedt = dIedt[n]
        diff = diff[n]
    
    fig = plt.figure(figsize=(12,8.5))
    ax = fig.add_subplot(2,3,1) 
    ax.imshow(I, cmap='gray')
    ax.set_title('I')

    ax = fig.add_subplot(2,3,2)
    ax.imshow(dI)
    ax.set_title('FIM')

    ax = fig.add_subplot(2,3,4)
    ax.imshow(dIedt)
    ax.set_title('Exact Euclidean')

    ax = fig.add_subplot(2,3,5)
    ax.imshow(diff)
    plt.title('FIM - Exact')

    ax = fig.add_subplot(2,3,6)
    extent = (0,I.shape[0], 0, I.shape[1])
    ax.contour(dI, origin='image', levels=range(0, n, 2), extent=extent, colors='red',
               linestyles='dashed')
    ax.contour(dIedt, origin='image', levels=range(0, n, 2), extent=extent, colors='black',
               linestyles='solid')
    ax.axis('image')
    ax.set_title(f'Mean abs error {mean_abs_error:.3f}\nMax abs error {max_abs_error:.3f}')    

    fig.savefig(outpath)    
    return mean_abs_error, max_abs_error
