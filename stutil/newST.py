
import numpy as np
from scipy.ndimage import filters
from scipy.ndimage import convolve as conv
from scipy import signal


def getGaussNoNorm(t, kSize=5):

    s = np.sqrt(t)
    x = np.arange(int(-np.ceil(s*kSize)), int(np.ceil(s*kSize))+1)
    x = np.reshape(x,(-1,1))
    g = np.exp(-x**2/(2*t))
    return g


def structure_tensor_2d_new(image, sigma, rho, out=None, truncate=4.0):
    """Structure tensor for 2D image data.
    Arguments:
        image: array_like
            A 2D array. Pass ndarray to avoid copying image.
        sigma: scalar
            A noise scale, structures smaller than sigma will be removed by smoothing.
        rho: scalar
            An integration scale giving the size over the neighborhood in which the
            orientation is to be analysed.
        out: ndarray, optinal
            A Numpy array with the shape (3, volume.shape) in which to place the output.
        truncate: float
            Truncate the filter at this many standard deviations. Default is 4.0.
    Returns:
        S: ndarray
            An array with shape (3, image.shape) containing elements of structure tensor
            (s_xx, s_yy, s_xy).
    Authors:
        vand@dtu.dk, 2019; niejep@dtu.dk, 2020
    """

    # Make sure it's a Numpy array.
    image = np.asarray(image)

    # Check data type. Must be floating point.
    if not np.issubdtype(image.dtype, np.floating):
        logging.warning('image is not floating type array. This may result in a loss of precision and unexpected behavior.') 

    # Compute derivatives (Scipy implementation truncates filter at 4 sigma).
    Ix = filters.gaussian_filter(image, sigma, order=[1, 0], mode='nearest', truncate=truncate)
    Iy = filters.gaussian_filter(image, sigma, order=[0, 1], mode='nearest', truncate=truncate)
    
    if out is None:
        # Allocate S.
        S = np.empty((3, ) + image.shape, dtype=image.dtype)
    else:
        # S is already allocated. We assume the size is correct.
        S = out

    # sigma2 = 0.707461 * (sigma* 2.75)
    sigma2 = 1.0613 * (sigma)

    g1 = getGaussNoNorm(sigma2**2, kSize=4)
    g2 = getGaussNoNorm((sigma2*0.999)**2,kSize=4/0.999) 

    norm = np.sum(signal.convolve(g1,g1.T)- signal.convolve(g2,g2.T))

    # Integrate elements of structure tensor (Scipy uses sequence of 1D).
    tmp = np.empty(image.shape, dtype=image.dtype)
    np.multiply(Ix, Ix, out=tmp)
    S[0] = (conv(conv(tmp,g1),g1.T) - conv(conv(tmp,g2),g2.T))/norm
    S[0] = filters.gaussian_filter(S[0], rho, mode='nearest', truncate=truncate)
    np.multiply(Iy, Iy, out=tmp)
    S[1] = (conv(conv(tmp,g1),g1.T) - conv(conv(tmp,g2),g2.T))/norm
    S[1] = filters.gaussian_filter(S[1], rho, mode='nearest', truncate=truncate)
    np.multiply(Ix, Iy, out=tmp)
    S[2] = (conv(conv(tmp,g1),g1.T) - conv(conv(tmp,g2),g2.T))/norm
    S[2] = filters.gaussian_filter(S[2], rho, mode='nearest', truncate=truncate)

    # S = S*(sigma**(2*1.1))
    S = S*(sigma**(2*1.2)+rho**2.5)

    return S


def structure_tensor_3d_new(volume, sigma, rho, out=None, truncate=4.0):
    """Structure tensor for 3D image data.
    Arguments:
        volume: array_like
            A 3D array. Pass ndarray to avoid copying volume.
        sigma: scalar
            A noise scale, structures smaller than sigma will be removed by smoothing.
        rho: scalar
            An integration scale giving the size over the neighborhood in which the
            orientation is to be analysed.
        out: ndarray, optional
            A Numpy array with the shape (6, volume.shape) in which to place the output.
        truncate: float
            Truncate the filter at this many standard deviations. Default is 4.0.
    Returns:
        S: ndarray
            An array with shape (6, volume.shape) containing elements of structure tensor
            (s_xx, s_yy, s_zz, s_xy, s_xz, s_yz).
    Authors: vand@dtu.dk, 2019; niejep@dtu.dk, 2019-2020
    """

    # Make sure it's a Numpy array.
    volume = np.asarray(volume)

    # Check data type. Must be floating point.
    if not np.issubdtype(volume.dtype, np.floating):
        logging.warning('volume is not floating type array. This may result in a loss of precision and unexpected behavior.')  

    # Computing derivatives (scipy implementation truncates filter at 4 sigma).
    Vx = filters.gaussian_filter(volume, sigma, order=[0, 0, 1], mode='nearest', truncate=truncate)
    Vy = filters.gaussian_filter(volume, sigma, order=[0, 1, 0], mode='nearest', truncate=truncate)
    Vz = filters.gaussian_filter(volume, sigma, order=[1, 0, 0], mode='nearest', truncate=truncate)

    if out is None:
        # Allocate S.
        S = np.empty((6, ) + volume.shape, dtype=volume.dtype)
    else:
        # S is already allocated. We assume the size is correct.
        S = out

    # scaling = 1.77
    # sigma2 = sigma*scaling
    # sigma2 = 0.707461 * (sigma* 1.5)
    sigma2 = 1.0613 * (sigma)

    g1 = getGaussNoNorm(sigma2**2, kSize=4)
    g2 = getGaussNoNorm((sigma2*0.999)**2,kSize=4/0.999)

    # norm = 4*np.pi*(sigma*1.77)**2
    # norm = 4*np.pi*(sigma2)**2
    norm = np.sum(signal.convolve(signal.convolve(g1,g1.T)[:,:,None],g1.T[None,:]) - (signal.convolve(signal.convolve(g2,g2.T)[:,:,None],g2.T[None,:])))

    # Integrating elements of structure tensor (scipy uses sequence of 1D).
    tmp = np.empty(volume.shape, dtype=volume.dtype)
    np.multiply(Vx, Vx, out=tmp)
    S[0] = (conv(conv(conv(tmp,g1[:,:,None]),g1.T[:,:,None]),g1.T[None,:,:]) - conv(conv(conv(tmp,g2[:,:,None]),g2.T[:,:,None]),g2.T[None,:,:]))/norm
    S[0] = filters.gaussian_filter(S[0], rho, mode='nearest', truncate=truncate)
    # filters.gaussian_filter(tmp, rho, mode='nearest', output=S[0], truncate=truncate)
    np.multiply(Vy, Vy, out=tmp)
    S[1] = (conv(conv(conv(tmp,g1[:,:,None]),g1.T[:,:,None]),g1.T[None,:,:]) - conv(conv(conv(tmp,g2[:,:,None]),g2.T[:,:,None]),g2.T[None,:,:]))/norm
    S[1] = filters.gaussian_filter(S[1], rho, mode='nearest', truncate=truncate)
    # filters.gaussian_filter(tmp, rho, mode='nearest', output=S[1], truncate=truncate)
    np.multiply(Vz, Vz, out=tmp)
    S[2] = (conv(conv(conv(tmp,g1[:,:,None]),g1.T[:,:,None]),g1.T[None,:,:]) - conv(conv(conv(tmp,g2[:,:,None]),g2.T[:,:,None]),g2.T[None,:,:]))/norm
    S[2] = filters.gaussian_filter(S[2], rho, mode='nearest', truncate=truncate)
    # filters.gaussian_filter(tmp, rho, mode='nearest', output=S[2], truncate=truncate)
    np.multiply(Vx, Vy, out=tmp)
    S[3] = (conv(conv(conv(tmp,g1[:,:,None]),g1.T[:,:,None]),g1.T[None,:,:]) - conv(conv(conv(tmp,g2[:,:,None]),g2.T[:,:,None]),g2.T[None,:,:]))/norm
    S[3] = filters.gaussian_filter(S[3], rho, mode='nearest', truncate=truncate)
    # filters.gaussian_filter(tmp, rho, mode='nearest', output=S[3], truncate=truncate)
    np.multiply(Vx, Vz, out=tmp)
    S[4] = (conv(conv(conv(tmp,g1[:,:,None]),g1.T[:,:,None]),g1.T[None,:,:]) - conv(conv(conv(tmp,g2[:,:,None]),g2.T[:,:,None]),g2.T[None,:,:]))/norm
    S[4] = filters.gaussian_filter(S[4], rho, mode='nearest', truncate=truncate)
    # filters.gaussian_filter(tmp, rho, mode='nearest', output=S[4], truncate=truncate)
    np.multiply(Vy, Vz, out=tmp)
    S[5] = (conv(conv(conv(tmp,g1[:,:,None]),g1.T[:,:,None]),g1.T[None,:,:]) - conv(conv(conv(tmp,g2[:,:,None]),g2.T[:,:,None]),g2.T[None,:,:]))/norm
    S[5] = filters.gaussian_filter(S[5], rho, mode='nearest', truncate=truncate)
    # filters.gaussian_filter(tmp, rho, mode='nearest', output=S[5], truncate=truncate)

    # S = S*(sigma**(2*1.1))
    S = S*(sigma**(2*1.2)+rho**2.5)

    return S