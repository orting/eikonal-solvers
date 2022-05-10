'''Test eikonal_solvers.fim
The tests generate a bunch of images showing the output for various settings,
these should be inspected to understand the approximation errors.
'''
# pylint: disable=invalid-name, missing-function-docstring, too-many-locals
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from test_data import point_source
from eikonal_solvers.fim import fim_sequential, fim

def test_accuracy_with_gridspacing(outdir):
    outpath = 'accuracy_fim_gridspacing_2d.png'
    if len(outdir) > 0:
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, outpath)
    ndim = 2
    n = 25
    I = point_source(n, ndim)
    F = np.full_like(I, 1)
    sequential_errors = []
    parallel_errors = []
    Deltas = [1,10,20,30,40,50]
    for Delta in Deltas:
        dIedt = distance_transform_edt(I, sampling=Delta)
        M = dIedt <= Delta*20

        dIsequential = fim_sequential(I, F, Delta)
        sequential_errors.append(np.mean(np.abs(dIedt[M] - dIsequential[M])))
        
        dIparallel = fim(I, F, Delta)
        parallel_errors.append(np.mean(np.abs(dIedt[M] - dIparallel[M])))


    fig = plt.figure(figsize=(12,8.5))
    ax = fig.add_subplot(1,2,1) 
    sline, = ax.plot(Deltas, parallel_errors, 'ro-')
    fline, = ax.plot(Deltas, sequential_errors, 'k*--', )
    fline.set_label('Sequential')
    sline.set_label('Parallel')    
    ax.set_xlabel('Spacing')
    ax.set_ylabel('Mean absolute error')
    ax.legend()
    
    ax = fig.add_subplot(1,2,2)
    sline, = ax.plot(np.log(Deltas), np.log(parallel_errors), 'ro-')
    fline, = ax.plot(np.log(Deltas), np.log(sequential_errors), 'k*--')
    fline.set_label('Sequential')
    sline.set_label('Parallel')
    ax.legend()
    ax.set_xlabel('log(Spacing)')
    ax.set_ylabel('log(Mean absolute error)')

    plt.suptitle('FIM Approximation error with increasing grid spacing')
    plt.savefig(outpath)

    for se, pe in zip(sequential_errors, parallel_errors):
        assert abs(se - pe) < 1e12

    for i in range(1, len(Deltas)):
        assert sequential_errors[i] > sequential_errors[i-1]
        assert parallel_errors[i] > parallel_errors[i-1]        
    


def test_accuracy_sequential_2d(outdir):
    outpath = 'accuracy_fim_isotropic_2d_sequential.png'
    if len(outdir) > 0:
        outpath = os.path.join(outdir, outpath)
    mean_abs_error, max_abs_error = _test_accuracy(True, 2, outpath=outpath)
    assert mean_abs_error < 0.495
    assert max_abs_error  < 0.919

def test_accuracy_parallel_2d(outdir):
    outpath='accuracy_fim_isotropic_2d_parallel.png'
    if len(outdir) > 0:
        outpath = os.path.join(outdir, outpath)
    mean_abs_error, max_abs_error = _test_accuracy(False, 2, outpath=outpath)
    assert mean_abs_error < 0.495
    assert max_abs_error  < 0.919


@pytest.mark.slow    
def test_accuracy_sequential_3d(outdir):
    outpath = 'accuracy_fim_isotropic_3d_sequential.png'
    if len(outdir) > 0:
        outpath = os.path.join(outdir, outpath)
    mean_abs_error, max_abs_error = _test_accuracy(True, 3, outpath=outpath)
    assert mean_abs_error < 0.92
    assert max_abs_error  < 1.51

def test_accuracy_parallel_3d(outdir):
    outpath = 'accuracy_fim_isotropic_3d_parallel.png'
    if len(outdir) > 0:
        outpath = os.path.join(outdir, outpath)
    mean_abs_error, max_abs_error = _test_accuracy(False, 3, outpath=outpath)
    assert mean_abs_error < 0.92
    assert max_abs_error  < 1.51

    
def test_accuracy_anisotropic_sequential_2d(outdir):
    outpath = 'accuracy_fim_anisotropic_2d_sequential.png'
    if len(outdir) > 0:
        outpath = os.path.join(outdir, outpath)
    Delta = (2,0.5)
    mean_abs_error, max_abs_error = _test_accuracy(True, 2, Delta, outpath=outpath)
    assert mean_abs_error < 0.537
    assert max_abs_error  < 1.015

def test_accuracy_anisotropic_parallel_2d(outdir):
    outpath = 'accuracy_fim_anisotropic_2d_parallel.png'
    if len(outdir) > 0:
        outpath = os.path.join(outdir, outpath)
    Delta = (2,0.5)
    mean_abs_error, max_abs_error = _test_accuracy(False, 2, Delta, outpath=outpath)
    assert mean_abs_error < 0.537
    assert max_abs_error  < 1.015    


@pytest.mark.slow
def test_accuracy_anisotropic_sequential_3d(outdir):
    outpath = 'accuracy_fim_anisotropic_3d_sequential.png'
    if len(outdir) > 0:
        outpath = os.path.join(outdir, outpath)
    Delta = (1,2,0.5)
    mean_abs_error, max_abs_error = _test_accuracy(True, 3, Delta, outpath=outpath)
    assert mean_abs_error < 0.991
    assert max_abs_error  < 1.645    

def test_accuracy_anisotropic_parallel_3d(outdir):
    outpath = 'accuracy_fim_anisotropic_3d_parallel.png'
    if len(outdir) > 0:
        outpath = os.path.join(outdir, outpath)
    Delta = (1,2,0.5)
    mean_abs_error, max_abs_error = _test_accuracy(False, 3, Delta, outpath=outpath)
    assert mean_abs_error < 0.991
    assert max_abs_error  < 1.645

    
def _test_accuracy(use_sequential=False, ndim=2, Delta=1, outpath='out_fim_test.png'):
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
    F = np.full_like(I, 1)

    if use_sequential:
        dI = fim_sequential(I, F, Delta)
    else:
        dI = fim(I, F, Delta)
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
