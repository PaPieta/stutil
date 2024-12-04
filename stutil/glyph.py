import numpy as np

# import vtk
# from mayavi import mlab
import os
import scipy

from stutil import vtk_write_lite
from stutil import volume


def cart2sph(cart):
    """Performs a conversion from carthesian to sperical coordinates, mimics matlab function with the same name
    Source:https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    Params:
    cart - (3,n) np.array with cartesian coordinates
    returns: (radius, elevation,azimuth)"""
    sph = np.zeros(cart.shape)
    xy = cart[0, :] ** 2 + cart[1, :] ** 2
    sph[0, :] = np.sqrt(xy + cart[2, :] ** 2)
    # sph[:,1] = np.arctan2(np.sqrt(xy), cart[2,:]) # for elevation angle defined from Z-axis down
    sph[1, :] = np.arctan2(
        cart[2, :], np.sqrt(xy)
    )  # for elevation angle defined from XY-plane up
    sph[2, :] = np.arctan2(cart[1, :], cart[0, :])
    return sph


def sph2cart(sph):
    """Performs a conversion from sperical to cartesian coordinates, mimics matlab function with the same name
    Source:https://se.mathworks.com/help/matlab/ref/sph2cart.html
    Params:
    sph - (radius, elevation,azimuth)\n
    returns: (3,n) np.array with cartesian coordinates (x,y,z)"""
    cart = np.zeros(sph.shape)
    cart[0, :] = sph[0, :] * np.cos(sph[1, :]) * np.cos(sph[2, :])
    cart[1, :] = sph[0, :] * np.cos(sph[1, :]) * np.sin(sph[2, :])
    cart[2, :] = sph[0, :] * np.sin(sph[1, :])
    return cart


def orientationVec(vec, flipOrder=True, fullSphere=True, weights=None):
    """Get spherical orientation (azimuth, elevation) from collection of unit direction
    vectors mapped either onto a sphere or half sphere.
    Params:
    vec - (3,n) np.array with unit vectors
    flipOrder - if True, vec is flipped to (ZYX) order instead of (XYZ) to follow most visualization software
    fullSphere - if True, returns complete sphere (redundant but nice looking), if False, returns half sphere
    weights - additional weights related to the vectors, carried over in case they need to be doubled when transitioning to fullSphere
    """

    if flipOrder:
        vec = vec[[2, 1, 0], :]

    sph = cart2sph(vec)

    if not fullSphere:
        # Flip directions onto half-sphere:
        idxN = np.where(sph[2, :] < -np.pi / 2)
        idxP = np.where(sph[2, :] > np.pi / 2)

        sph[2, idxN] = sph[2, idxN] + np.pi
        sph[1, idxN] = -sph[1, idxN]  # I think this is needed, HansMartin doesn't have that

        sph[2, idxP] = sph[2, idxP] - np.pi
        sph[1, idxP] = -sph[1, idxP]
    if fullSphere:
        sph2 = np.copy(sph)

        sph2[1, :] = -sph2[1, :]

        idxN = np.where(sph2[2, :] < 0)
        idxP = np.where(sph2[2, :] > 0)

        sph2[2, idxN] = sph2[2, idxN] + np.pi
        sph2[2, idxP] = sph2[2, idxP] - np.pi

        sph = np.hstack((sph, sph2))
        if weights is not None:
            weights = np.hstack((weights, weights))

    if weights is None:
        return sph
    else:
        return sph, weights


def histogram2d(sph: np.array, bins=[100, 200], norm=None, weights=None):
    """Wrapper to the numpy histogram 2d function with some extra steps.
    Params:
    sph - (3,n) spherical coordinates
    bins - nuber of histogram bins in each direction
    norm - normalization type:
        'prob' - normalizes to 0-1 range
        'binArea' - corrects for the variable bin area
        'prob_binArea' - combines both
    weights - additional weights for the histogram calculation (e.g. linearity score)
    """

    H, el, az = np.histogram2d(sph[1, :], sph[2, :], bins=bins, weights=weights)

    az_diff = az[1:] - az[:-1]
    el_diff = np.sin(el[1:]) - np.sin(
        el[:-1]
    )  # elevation is in -pi/2 to p/2 so sign(cos(el)) will always be 1
    el_dif_grid, az_dif_grid = np.meshgrid(el_diff, az_diff)
    binArea = (el_dif_grid * az_dif_grid).transpose()

    if norm == "prob":
        H = H / np.sum(H)
    elif norm == "binArea":
        H = H / binArea
    elif norm == "prob_binArea":
        # With this order we don't sum to 1 but we are invariate to the volume size
        # Opposite order would sum it up to 1, but binArea calculation is prone
        # to numerical errors that add up fast when summed
        H = H / np.sum(H)
        H = H / binArea
    else:
        assert norm is None, f"Wrong norm type: {norm}"

    return H, el, az, binArea


def generate_glyph(H, el, az, rotAngle, generateColor=True, flipColor=True):
    """Creates glyph-like mesh visualization from the 2d spherical coordinate eigenvector histogram
    Params
    H - histogram values
    el, az - binning limits in elevation and azimuth direction
    normDiv - normalization value, what to divide the histogram values by
    generateColor - if True, adds color information to glyph directions
    Returns:
    XYZ - (3,n) np.array with cartesian coordinates
    RGB - (3,n) np.array with RGB color values
    """
    el_center = (el[1:] + el[:-1]) / 2
    az_center = (az[1:] + az[:-1]) / 2

    # If elevation and azimuth bins have to be rotated
    if np.any(np.array(rotAngle) != 0):
        el_az_grid = np.array(np.meshgrid(el_center, az_center))
        el_az_grid = np.vstack((np.ones(el_az_grid.shape[1:])[None, :], el_az_grid))
        el_az_cart = sph2cart(el_az_grid)

        # Reverse order of rotation
        rot_ang_flip = rotAngle * [-1, 1]
        rot = scipy.spatial.transform.Rotation.from_euler("yz", rot_ang_flip)
        # rot = R.from_euler('yz',rot_ang_flip)
        el_az_cart_rot = rot.as_matrix() @ el_az_cart.reshape(3, -1)

        # el_az_cart_rot = rot([rotAngle[0],-rotAngle[1]],el_az_cart.reshape(3,-1),flipOpposites=False)

        el_az_rot = cart2sph(el_az_cart_rot)

        el_az_rot = el_az_rot.reshape(el_az_grid.shape)
        el_center_grid = el_az_rot[1, :]
        az_center_grid = el_az_rot[2, :]
    else:
        el_center_grid, az_center_grid = np.meshgrid(el_center, az_center)

    x = np.cos(el_center_grid) * np.cos(az_center_grid)
    y = np.cos(el_center_grid) * np.sin(az_center_grid)
    z = np.sin(el_center_grid)
    XYZ = np.array([x.transpose(), y.transpose(), z.transpose()])
    if generateColor:
        XYZ_copy = np.copy(XYZ)
        if flipColor:
            # Flip to one half of the sphere, to have same colors on both sides
            flipMask = XYZ_copy[0, :] < 0
            flipMask = np.array([flipMask, flipMask, flipMask])
            XYZ_copy[flipMask] = -XYZ_copy[flipMask]

        # RGB = volume.convertToFan(np.expand_dims(XYZ_copy,-1),halfSphere=flipColor)[:3,:,:,0]
        RGB = volume.convertToIco(np.expand_dims(XYZ_copy, -1))[:3, :, :, 0]
        # RGB = 1-RGB
    else:
        RGB = None
    # XYZ = XYZ*H.transpose()
    # XYZ = np.moveaxis(XYZ,1,-1)
    XYZ = XYZ * H

    return XYZ, RGB


def save_glyph(XYZ, RGB, savePath):
    """Saves glyph in a VTK format based on the given mesh and optional color information
    Params:
    XYZ - (3,n) np.array with cartesian coordinates
    RGB - (3,n) np.array with RGB color values
    savePath - path to save the glyph
    """
    # Save as vtk surf
    assert os.path.dirname(savePath) == "" or os.path.exists(
        os.path.dirname(savePath)
    ), f"Given path does not exist {savePath}"
    _, ext = os.path.splitext(savePath)
    assert ext == ".vtk", f"File extension ({ext}) is not .vtk"

    vtk_write_lite.save_surf2vtk(savePath, XYZ, RGB)