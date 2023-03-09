import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy import signal
from scipy.interpolate import splprep, splev
from sklearn.utils.extmath import cartesian
from skimage.segmentation import active_contour
from skimage.io import imread
from eikonal_solvers.fmm import fmm
from scipy.spatial import KDTree


def get_boundary(im, outer=True):
    stencil = np.zeros((3,3))
    stencil[0,1] = stencil[2,1] = stencil[1,0] = stencil[1,2] = 1
    if outer:
        boundary = (signal.convolve2d(im, stencil, mode='same'))
        return ((boundary * np.logical_not(im))>0).astype(float)
    else:
        boundary = (signal.convolve2d(im==0, stencil, mode='same'))
        return ((boundary * im)>0).astype(float)

def get_no_init(im):
    X = np.full_like(im, np.inf)
    X[im==0] = 0
    return X

def get_spline_init(im, inner_boundary, outer_boundary):
    # Find an ordering of the pixels
    marked = set()
    idxs = np.nonzero(outer_boundary)
    y = idxs[0][0]
    x = idxs[1][0]
    ys = [y]
    xs = [x]
    marked.add((y,x))
    added = True
    while added:
        added = False
        y0,x0 = y,x
        for dy,dx in [(-1,0), (-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1)]:
            y = y0 + dy
            x = x0 + dx
            if outer_boundary[y,x] and not (y,x) in marked:
                ys.append(y)
                xs.append(x)
                marked.add((y,x))
                added = True
                break

    # Fit splines to
    # f(t) = xs
    # f(t) = ys
    # and make small step predictions.
    xs.append(xs[0])
    ys.append(ys[0])
    xs = np.array(xs)
    ys = np.array(ys)
    N = len(xs)
    knots = np.linspace(0, 1, N//10)

    tck, u = splprep([xs, ys], k=3, t=knots, task=-1)
    xsp,ysp = splev(np.linspace(0, 1, 100*N), tck)

    # Use kdtree to find nearest point for each pixel on the inner boundary
    data = np.stack([xsp, ysp], axis=-1)
    kdtree = KDTree(data)
    query_ys, query_xs = np.nonzero(inner_boundary)
    points = np.stack([query_xs, query_ys], axis=-1)

    distances, neighbors = kdtree.query(points, 1)    
    spline_init = np.full_like(inner_boundary, np.inf)
    spline_init[query_ys, query_xs] = distances
    
    return spline_init

def get_smooth_mask_init(im, inner_boundary, outer_boundary):
    smooth_mask = gaussian_filter((im != 0).astype(float), order=0, sigma=5)
    smooth_mask_init = np.zeros_like(im)
    smooth_mask_init[inner_boundary==1] = smooth_mask[inner_boundary==1]
    smooth_mask_init[inner_boundary != 1] = np.inf
    low = smooth_mask_init[inner_boundary == 1].min()
    high = smooth_mask_init[inner_boundary == 1].max()
    smooth_mask_init[inner_boundary == 1] = (smooth_mask_init[inner_boundary == 1] - low)/(high - low)
    return smooth_mask_init
    
    
def circle():
    n = 101
    im = np.zeros((n,n))
    r2 = (n//2 - 5)**2
    for i in range(n):
        for j in range(n):
            if (i - n//2)**2 + (j - n//2)**2 < r2:
                im[i,j] = np.inf
                
    # print('Exact distance field')
    r = np.sqrt(r2)
    exact = np.zeros_like(im)
    for i in range(n):
        for j in range(n):
            d2 = (i - n//2)**2 + (j - n//2)**2
            if d2 <= r2: 
                exact[i,j] = r - np.sqrt(d2)                
    return im, exact

def s_shape():
    im = (imread('S.png')[...,0]==0).astype(float)
    exact = distance_transform_edt(im)
    return im, exact

def fd_grad(im):
    fd_y = im[1:,1:] - im[:-1,1:]
    fd_x = im[1:,1:] - im[1:,:-1]
    fd_mag = np.sqrt(fd_y**2 + fd_x**2)
    zero_mag = fd_mag == 0
    fd_mag[zero_mag] = 1
    fd = np.stack([(1+fd_y/fd_mag)/2,
                   (1+fd_x/fd_mag)/2,
                   np.zeros_like(fd_y)], axis=-1)
    fd[zero_mag,:] = 0
    return fd

def example(name):
    if name == 'circle':
        im, exact = circle()
    elif name == 's-shape':
        im, exact = s_shape()
    else:
        return
    exact_fd = fd_grad(exact)

    # Calculate distance transforms and gradients
    inner_boundary = get_boundary(im != 0, False)
    outer_boundary = get_boundary(im != 0, True)
    phi = (im > 0).astype(float)
    
    exact_init = np.full_like(im, np.inf)
    exact_init[inner_boundary==1] = exact[inner_boundary==1]
    d_exact_init = fmm(exact_init, phi, X_is_distance=True)
    d_exact_init[im==0] = 0
    d_exact_init_fd = fd_grad(d_exact_init)

    no_init = get_no_init(im)
    d_no_init = fmm(no_init, phi, X_is_distance=True)
    d_no_init[im==0] = 0
    d_no_init_fd = fd_grad(d_no_init)

    smooth_mask_init = get_smooth_mask_init(im, inner_boundary, outer_boundary)
    d_smooth_mask_init = fmm(smooth_mask_init, phi, X_is_distance=True)
    d_smooth_mask_init[im==0] = 0
    d_smooth_mask_init_fd = fd_grad(d_smooth_mask_init)

    spline_init = get_spline_init(im, inner_boundary, outer_boundary)
    d_spline_init = fmm(spline_init, phi, X_is_distance=True)
    d_spline_init[im==0] = 0
    d_spline_init_fd = fd_grad(d_spline_init)

    # Plot it    
    disp_smooth_mask_init = np.zeros_like(smooth_mask_init)
    disp_smooth_mask_init[inner_boundary == 1] = smooth_mask_init[inner_boundary == 1]
    disp_exact_init = np.zeros_like(exact_init)
    disp_exact_init[inner_boundary == 1] = exact_init[inner_boundary == 1]
    disp_spline_init = np.zeros_like(spline_init)
    disp_spline_init[inner_boundary == 1] = spline_init[inner_boundary == 1]
    
    fig,ax = plt.subplots(2,3,figsize=(20,15))
    ax[0,0].imshow(disp_exact_init)
    ax[0,0].set_title('Exact initialization of boundary pixels distance')

    ax[0,1].imshow(disp_smooth_mask_init)
    ax[0,1].set_title('Smooth mask initialization of boundary pixels distance')

    ax[0,2].imshow(disp_spline_init)
    ax[0,2].set_title('Spline initialization of boundary pixels distance')
    
    diff = disp_exact_init - disp_smooth_mask_init
    ax[1,1].imshow(diff)
    ax[1,1].set_title(f'max(|exact - smooth|) = {np.max(np.abs(diff)):02f}')

    diff = disp_exact_init - disp_spline_init
    ax[1,2].imshow(diff)
    ax[1,2].set_title(f'max(|exact - spline|) = {np.max(np.abs(diff)):02f}')
    plt.tight_layout()
    plt.savefig(f'out/dist-field-grad-{name}-boundary-initialization.png')

    
    fig, ax = plt.subplots(3,5, figsize=(25,20))
    ax[0][0].imshow(im != 0, cmap='gray')

    ax[0][1].imshow(d_exact_init, cmap='gray')
    ax[0][1].set_title('FMM from exact init')

    ax[0][2].imshow(d_no_init, cmap='gray')
    ax[0][2].set_title('FMM from no init')

    ax[0][3].imshow(d_smooth_mask_init, cmap='gray')
    ax[0][3].set_title('FMM from smooth mask init')

    ax[0][4].imshow(d_spline_init, cmap='gray')
    ax[0][4].set_title('FMM from spline init')
    
    ax[1][0].imshow(exact, cmap='gray')
    ax[1][0].set_title('Exact')
    
    ax[1][1].imshow(np.abs(d_exact_init-exact), cmap='gray')
    ax[1][1].set_title(f'max abs error {np.max(np.abs(d_exact_init-exact)):.2f}')

    ax[1][2].imshow(np.abs(d_no_init-exact), cmap='gray')
    ax[1][2].set_title(f'max abs error {np.max(np.abs(d_no_init-exact)):.2f}')

    ax[1][3].imshow(np.abs(d_smooth_mask_init-exact), cmap='gray')
    ax[1][3].set_title(f'max abs error {np.max(np.abs(d_smooth_mask_init-exact)):.2f}')

    ax[1][4].imshow(np.abs(d_spline_init-exact), cmap='gray')
    ax[1][4].set_title(f'max abs error {np.max(np.abs(d_spline_init-exact)):.2f}')

    ax[2][0].imshow(exact_fd)
    ax[2][0].set_title('Distance field gradients')
    ax[2][1].imshow(d_exact_init_fd)
    ax[2][2].imshow(d_no_init_fd)
    ax[2][3].imshow(d_smooth_mask_init_fd)
    ax[2][4].imshow(d_spline_init_fd)
    plt.suptitle('Influence of initialization on distance field gradients')
    plt.tight_layout()
    plt.savefig(f'out/dist-field-grad-{name}.png')
    
if __name__ == '__main__':
    example('circle')
    example('s-shape')
