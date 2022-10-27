import os

import numpy as np
import skimage.io 
import matplotlib.pyplot as plt

from scale_space import ScaleSpace
import glyph


if __name__ == "__main__":

    I = skimage.io.imread('/work3/papi/Cheese/cheese_10X-40kV-air-45s_recon_cut.tif')
    I = (I.astype('float')/np.max(I))*255
    # I = I[:200,:200,:200].astype(float)

    print(f"Image size: {I.shape}")

    # tensorScaleSpace = ScaleSpace(I,sigma_scales=[1,2,4,8,16],rho_scales=[1,2,4,8,16])
    tensorScaleSpace = ScaleSpace(I,sigma_scales=[1,2,4,8],rho_scales=[1,2,4,8])

    val,vec,lin,scale = tensorScaleSpace.calcFast()

    sph, sph_lin = glyph.orientationVec(vec.reshape(3,-1), [2,1,0], fullSphere=True, weights=lin.ravel())

    H, el, az, binArea = glyph.histogram2d(sph,bins=[100,200],norm='prob_binArea', weights=sph_lin)

    glyph.save_glyph(H,el,az,savePath="/work3/papi/Cheese/test.vtk")

    print("Done")
