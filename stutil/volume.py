import numpy as np
import os
import tifffile
import scmap
import scipy.ndimage
import maxflow

from stutil import vtk_write_lite

#TODO: update documentation

def hsv2rgb3d(vol):
    """Converts a volume from a hsv to rgb definition, extension of the 2D version\n
    Source:https://www.had2know.org/technology/hsv-rgb-conversion-formula-calculator.html\n
    Params:\n
    vol - (3,x,y,z) color (hsv) volume\n
    Returns: (3,x,y,z) rgb volume"""
    #
    M = vol[2,:,:,:]
    m = M*(1-vol[1,:,:,:])
    mod_res = ((vol[0,:,:,:]*180/(np.pi*60)))%2
    z = (M-m)*(1-np.abs(mod_res-1))
    
    # Assuming H is in 0-180 range (Not true anymore?)
    R = np.zeros_like(vol[0,:,:,:])
    G = np.zeros_like(vol[0,:,:,:])
    B = np.zeros_like(vol[0,:,:,:])
    
    Hrange1 = vol[0,:,:,:] < np.pi/3
    Hrange2 = np.bitwise_and(vol[0,:,:,:] >= np.pi/3, vol[0,:,:,:] < np.pi*2/3)
    Hrange3 = np.bitwise_and(vol[0,:,:,:] >= np.pi*2/3, vol[0,:,:,:] < np.pi)
    Hrange4 = np.bitwise_and(vol[0,:,:,:] >= np.pi, vol[0,:,:,:] < np.pi*4/3)
    Hrange5 = np.bitwise_and(vol[0,:,:,:] >= np.pi*4/3, vol[0,:,:,:] < np.pi*5/3)
    Hrange6 = vol[0,:,:,:] >= np.pi*5/3
    
    R[Hrange1] = M[Hrange1]
    G[Hrange1] = (z+m)[Hrange1]
    B[Hrange1] = m[Hrange1]

    R[Hrange2] = (z+m)[Hrange2]
    G[Hrange2] = M[Hrange2]
    B[Hrange2] = m[Hrange2]

    R[Hrange3] = m[Hrange3]
    G[Hrange3] = M[Hrange3]
    B[Hrange3] = (z+m)[Hrange3]
    
    R[Hrange4] = m[Hrange4]
    G[Hrange4] = (z+m)[Hrange4]
    B[Hrange4] = M[Hrange4]
    
    R[Hrange5] = (z+m)[Hrange5]
    G[Hrange5] = m[Hrange5]
    B[Hrange5] = M[Hrange5]
    
    R[Hrange6] = M[Hrange6]
    G[Hrange6] = m[Hrange6]
    B[Hrange6] = (z+m)[Hrange6]

    return np.array([R,G,B])

def convertToFan(vec, halfSphere=False, weights=None, mask=None):
    """Converts a volume of vectors to a volume of rgba values representing vector directions using a Fan color scheme.\n
    Useful for vectors that don't point up or down.\n
    Params:\n
    vec - (3,x,y,z) volume of vectors\n
    halfSphere - if True, the color range is squished to half a sphere\n
    weights - weights used for the alpha channel\n
    mask - binary mask of areas of interest\n
    Returns: (4,x,y,z) rgba volume\n
    """

    if halfSphere:
        # Stretch artificially the x values to use all colors
        fake_hsv = np.arctan2(vec[1,:,:],(vec[0,:,:]*2)-1)+np.pi
    else:
        fake_hsv = np.arctan2(vec[1,:,:],vec[0,:,:])+np.pi
    fake_hsv = np.array([fake_hsv,np.ones_like(fake_hsv),np.ones_like(fake_hsv)])

    fake_rgb = hsv2rgb3d(fake_hsv)
    colormap_vol = (1-vec[2,:,:]**2)*fake_rgb + 0.5*(vec[2,:,:]**2)

    colormap_vol = np.vstack((colormap_vol,np.ones_like(colormap_vol[0,:,:,:])[None,:]))

    if weights is not None:
        colormap_vol[3,:] = colormap_vol[3,:]*weights

    if mask is not None:
        mask_rgba = np.invert(np.array((mask,mask,mask,mask)))
        colormap_vol[mask_rgba] = 0

    return colormap_vol

def convertToIco(vec,  weights=None, mask=None):
    """Converts a volume of vectors to a volume of rgba values representing vector directions using an Icosahedron color scheme.\n
    Useful for vectors that don't have any particular distribution.\n
    Params:\n
    vec - (3,x,y,z) volume of vectors\n
    weights - weights used for the alpha channel\n
    mask - binary mask of areas of interest\n
    Returns: (4,x,y,z) rgba volume\n
    """

    coloring = scmap.Ico() 
    vec_flip = np.moveaxis(vec.reshape(3,-1), 0,-1)
    colormap_vol = coloring(vec_flip)
    colormap_vol = np.moveaxis(colormap_vol,0,-1).reshape(vec.shape)

    colormap_vol = np.vstack((colormap_vol,np.ones_like(colormap_vol[0,:,:,:])[None,:]))

    if weights is not None:
        colormap_vol[3,:] = colormap_vol[3,:]*weights

    if mask is not None:
        mask_rgba = np.invert(np.array((mask,mask,mask,mask)))
        colormap_vol[mask_rgba] = 0

    return colormap_vol

def saveRgbaVolume(rgba,savePath):
    """Prepares and saves RGBA .tiff volume to be viewed in ParaView\n
    Due to ParaView behaviour that this function corrects for, it may not perfrom as well in other programs\n
    Params:\n
    rgba - (4,x,y,z) rgba volume \n
    savePath - path where the volume should be saved\n
    """

    rgba = np.moveaxis(rgba,0,-1)
    #  ParaView prefers to work with 8 bit uints
    rgba = (rgba*255).astype(np.uint8)
    # # For some reason alpha channel is in the opposite order in ParaView
    rgba[:,:,:,3] = 255-rgba[:,:,:,3]
    # ParaView needs a full range of values to work correctly
    rgba[0,0,0,3] = 255
    rgba[1,0,0,3] = 0
    #Save as tiff
    assert os.path.dirname(savePath) == '' or os.path.exists(os.path.dirname(savePath)), f"Given path does not exist {savePath}"
    _, ext = os.path.splitext(savePath)
    assert ext == '.tiff', f"File extension ({ext}) is not .tiff"

    tifffile.imwrite(savePath, rgba)

def holeFillGauss(vol, thresh=5, minBlobSize=50, dilSize=5):
    """Finds and fills dark holes in the cheese volume using gaussian noise\n
    Gaussian distribution tries to follow the data distribution\n
    Params:\n
    vol - (x,y,z) grayscale volume \n
    thresh - binarization thershold for finding the holes\n
    minBlobSize - minimum hole size that will won't be filtered out\n
    dilSize - dilation kernel size\n
    Returns: volume with filled holes, mask with the found holes set to 0
    """

    #Threshold volume and closing
    holeMask = vol < thresh
    holeMask = scipy.ndimage.binary_closing(holeMask, np.ones((3,3,3)))
    #Connected components detection
    labels, num_labels = scipy.ndimage.label(holeMask)
    blobSize = scipy.ndimage.sum(holeMask,labels,np.arange(0,np.max(labels)).tolist())
    holeMask = np.isin(labels,np.where(blobSize > minBlobSize))
    #Small dilation to be safe
    holeMask = scipy.ndimage.binary_dilation(holeMask, np.ones((dilSize,dilSize,dilSize)))

    volMask = np.invert(holeMask)
    # Prepeare gaussian param
    volFilled = np.copy(vol)
    std = np.std(vol[volMask])
    mean = np.mean(vol[volMask])
    num = np.sum(holeMask)
    # Fill holes
    volFilled[holeMask] = np.random.normal(mean,std,num)
    
    return volFilled, volMask

def holeFillMrfGauss(vol, meanHole=100, meanObj=120, beta=5, minBlobSize=3000):
    """Finds holes using 3D MRF egmentation and fills them using gaussian noise\n
    Gaussian distribution tries to follow the data distribution\n
    Params:\n
    vol - (x,y,z) grayscale volume (float)\n
    meanObj - mean pixel intensity of the segmented object\n
    meanHole - mean pixel intensity of the holes\n
    beta - MRF 2-clique potential parameter\n
    minBlobSize - minimum hole size that will won't be filtered out\n
    Returns: volume with filled holes, mask with the found holes set to 0
    """

    # Prepare cost volumes
    mu = np.array([meanHole, meanObj])
    U = np.stack([(vol-mu[i])**2 for i in range(len(mu))],axis=3)

    #Prepare graph
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(vol.shape)
    g.add_grid_edges(nodeids, beta)
    g.add_grid_tedges(nodeids, U[:,:,:,1], U[:,:,:,0])

    # Solve
    g.maxflow()
    S = g.get_grid_segments(nodeids)
    holeMask = 1-S

    # 1st filter bobs by size
    labels, num_labels = scipy.ndimage.label(holeMask)
    blobSize = scipy.ndimage.sum(holeMask,labels,np.arange(0,np.max(labels)).tolist())
    holeMask = np.isin(labels,np.where(blobSize > minBlobSize))

    # Binary filtering 
    holeMask = scipy.ndimage.binary_closing(holeMask, np.ones((3,3,3)))
    holeMask = scipy.ndimage.binary_erosion(holeMask, np.ones((3,3,3)))

    # Another filter by size
    labels, num_labels = scipy.ndimage.label(holeMask)
    blobSize = scipy.ndimage.sum(holeMask,labels,np.arange(0,np.max(labels)).tolist())
    holeMask = np.isin(labels,np.where(blobSize > minBlobSize))

    # Dilation to cover small neighbourhood of holes
    holeMask = scipy.ndimage.binary_dilation(holeMask, np.ones((5,5,5)))

    volMask = np.invert(holeMask)
    # Prepeare gaussian param
    volFilled = np.copy(vol)
    std = np.std(vol[volMask])
    mean = np.mean(vol[volMask])
    num = np.sum(holeMask)
    # Fill holes
    volFilled[holeMask] = np.random.normal(mean,std,num)
    
    return volFilled, volMask