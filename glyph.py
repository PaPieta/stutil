import numpy as np
# import vtk
# from mayavi import mlab
import os
import vtk_write_lite

def cart2sph(cart):
    """Performs a conversion from carthesian to sperical coordinates, mimics matlab function with the same name\n
    Source:https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion\n
    Params:\n
    cart - (3,n) np.array with cartesian coordinates\n
    returns: (radius, elevation,azimuth)"""
    sph = np.zeros(cart.shape)
    xy = cart[0,:]**2 + cart[1,:]**2
    sph[0,:] = np.sqrt(xy + cart[2,:]**2)
    # sph[:,1] = np.arctan2(np.sqrt(xy), cart[2,:]) # for elevation angle defined from Z-axis down
    sph[1,:] = np.arctan2(cart[2,:], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    sph[2,:] = np.arctan2(cart[1,:], cart[0,:])
    return sph

def orientationVec(vec,poleOrder=[0,1,2],fullSphere=True, weights=None):
    """Get spherical orientation (azimuth, elevation) from collection of unit direction
    vectors mapped either onto a sphere or half sphere with a given poleOrder.\n
    Params:\n
    vec - (3,n) np.array with unit vectors\n
    poleOrder - order of the poles in the vec parameter (eg [2,1,0] = [Z,Y,X])\n
    fullSphere - if True, returns complete sphere (redundant but nice looking), if False, returns half sphere\n
    weights - additional weights related to the vectors, carried over in case they need to be doubled when transitioning to fullSphere
    """
    
    vec = vec[poleOrder,:]
    
    sph = cart2sph(vec)
    
    if fullSphere == False:
        # Flip directions onto half-sphere:
        idxN = np.where(sph[2,:] < -np.pi/2);
        idxP = np.where(sph[2,:] > np.pi/2);
        
        sph[2,idxN] = sph[2,idxN] + np.pi
        sph[1,idxN] = -sph[1,idxN] # I think this is needed, HansMartin doesn't have that
        
        sph[2,idxP] = sph[2,idxP] - np.pi
        sph[1,idxP] = -sph[1,idxP]
    if fullSphere == True:
        sph2 = np.copy(sph)    
        
        sph2[1,:] = -sph2[1,:]
        
        idxN = np.where(sph2[2,:] < 0);
        idxP = np.where(sph2[2,:] > 0);
        
        sph2[2,idxN] = sph2[2,idxN] + np.pi
        sph2[2,idxP] = sph2[2,idxP] - np.pi
        
        sph = np.hstack((sph,sph2))
        if weights is not None:
            weights = np.hstack((weights,weights))
        
    if weights is None:
        return sph
    else:
        return sph, weights
    
def histogram2d(sph:np.array, bins=[100,200], norm=None, weights=None):
    """Wrapper to the numpy histogram 2d function with some extra steps.\n
    Params:\n
    sph - (3,n) spherical coordinates\n
    bins - nuber of histogram bins in each direction\n
    norm - normalization type:\n
        'prob' - normalizes to 0-1 range\n
        'binArea' - corrects for the variable bin area\n
        'prob_binArea' - combines both
    weights - additional weights for the histogram calculation (e.g. linearity score)
    """

    H,el,az = np.histogram2d(sph[1,:],sph[2,:],bins=bins, weights=weights) 

    az_diff = az[1:]-az[:-1]
    el_diff = np.sin(el[1:])-np.sin(el[:-1]) # elevation is in -pi/2 to p/2 so sign(cos(el)) will always be 1
    el_dif_grid,az_dif_grid = np.meshgrid(el_diff,az_diff)
    binArea = (el_dif_grid*az_dif_grid).transpose()
    
    if norm=='prob':
        H = H/np.sum(H)
    elif norm=='binArea':
        H = H/binArea
    elif norm=='prob_binArea':
        H = H/binArea
        H = H/np.sum(H)
    else:
        assert norm is None, f"Wrong norm type: {norm}"

    return H, el, az, binArea

def save_glyph(H,el,az,savePath):
    """Creates glyph-like mesh visualization from the 2d spherical coordinate eigenvector histogram\n
    Params\n
    H - histogram values\n
    el, az - binning limits in elevation and azimuth direction\n
    savePath - path where the .vtk file of the glyph surface should be saved.
    """

    el_center = (el[1:]+el[:-1])/2
    az_center = (az[1:]+az[:-1])/2
    el_center_grid,az_center_grid = np.meshgrid(el_center,az_center)
    x1 = H.transpose()*np.cos(el_center_grid)*np.cos(az_center_grid)
    y1 = H.transpose()*np.cos(el_center_grid)*np.sin(az_center_grid)
    z1 = H.transpose()*np.sin(el_center_grid)

    # Save as vtk surf
    assert os.path.dirname(savePath) == '' or os.path.exists(os.path.dirname(savePath)), f"Given path does not exist {savePath}"
    _, ext = os.path.splitext(savePath)
    assert ext == '.vtk', f"File extension ({ext}) is not .vtk"

    vtk_write_lite.save_surf2vtk(savePath, x1, y1, z1)
    
    
    ### DISABLED MAYAVI VERSION AS IT REQUIRES OPENGL AND A DISPLAY - PROBLEMATIC WITH HPC
    # if not plot:
    #     mlab.options.offscreen = True
    # surf = mlab.mesh(x1,y1,z1,scalars=H.transpose())
    # if plot:
    #     mlab.show()


    # if savePath is not None:
    #     assert os.path.dirname(savePath) == '' or os.path.exists(os.path.dirname(savePath)), f"Given path does not exist {savePath}"
    #     _, ext = os.path.splitext(savePath)
    #     assert ext == '.stl', f"File extension ({ext}) is not .stl"


    #     surface_vtk = mlab.pipeline.get_vtk_src(surf)
    #     stlWriter = vtk.vtkSTLWriter()
    #     # Set the file name
    #     stlWriter.SetFileName(savePath)
    #     # Set the input for the stl writer. surface.output[0]._vtk_obj is a polydata object
    #     stlWriter.SetInputData(surface_vtk[0]._vtk_obj)
    #     stlWriter.Write()

    # return surf