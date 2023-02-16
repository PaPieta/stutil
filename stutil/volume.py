import numpy as np
import os
import tifffile
import scmap
import scipy.ndimage

from stutil import vtk_write_lite

#TODO: update documentation

def hsv2rgb3d(vol):
    #https://www.had2know.org/technology/hsv-rgb-conversion-formula-calculator.html
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
    """Converts a volume of vectors to a volume of rgba values representing vector directions."""

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
    """Converts a volume of vectors to a volume of rgba values representing vector directions."""

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

def saveRgbaVolume(rgba,savePath=None):
    
    rgba = np.moveaxis(rgba,0,-1)
    #  ParaView prefers to work with 8 bit uints
    rgba = (rgba*255).astype(np.uint8)
    # # For some reason alpha channel is in the opposite order in ParaView
    # rgba = 255-rgba
    rgba[:,:,:,3] = 255-rgba[:,:,:,3]
    # ParaView needs a full range of values to work correctly
    rgba[0,0,0,3] = 255
    rgba[1,0,0,3] = 0
    #Save as tiff
    if savePath is not None:
        assert os.path.dirname(savePath) == '' or os.path.exists(os.path.dirname(savePath)), f"Given path does not exist {savePath}"
        _, ext = os.path.splitext(savePath)
        assert ext == '.tiff', f"File extension ({ext}) is not .tiff"

        #Switched to tiff save from vtk for faster visualization (vtk small for big volumes)
        # vtk_write_lite.save_rgba2vtk(colormap_vol, savePath)
        tifffile.imwrite(savePath, rgba)

    # temp = np.moveaxis(colormap_vol.astype('float32'),0,-1)
    # temp[:,:,:,:-1] = temp[:,:,:,:-1]*255
    # temp[0,0,0,3] = 255
    # temp[0,0,0,1] = 0
    # tifffile.imwrite('temp.tiff', (temp).astype(np.uint8))

def holeFillGauss(vol, thresh=5, minBlobSize=50):

    #Threshold volume and closing
    holeMask = vol < thresh
    holeMask = scipy.ndimage.binary_closing(holeMask, np.ones((3,3,3)))
    #Connected components detection
    labels, num_labels = scipy.ndimage.label(holeMask)
    blobSize = scipy.ndimage.sum(holeMask,labels,np.arange(0,np.max(labels)).tolist())
    holeMask = np.isin(labels,np.where(blobSize > minBlobSize))
    #Small dilation to be safe
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