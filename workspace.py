import numpy as np
import skimage.io 
import matplotlib.pyplot as plt


from scale_space import ScaleSpace
import glyph
import volume


if __name__ == "__main__":

    I = skimage.io.imread('../Cheese/Scans/Anders_first_scans/rockwool/HU0507G_4X-80kV-Ai-10W-5s_recon_cut-300-500.tif')
    I = (I.astype('float')/np.max(I))*255
    I = I[:,200:400:,200:400].astype(float)

    tensorScaleSpace = ScaleSpace(I,sigma_scales=[2,3,4],rho_scales=[1,2,3])

    val,vec,lin,scale = tensorScaleSpace.calcFast()

    rgba = volume.convertToColormap(vec)
    volume.saveRgbaVolume(rgba,savePath="testVolume.tiff")

    print("RGBA direction volume saved.")

    sph, sph_lin = glyph.orientationVec(vec.reshape(3,-1), [2,1,0], fullSphere=True, weights=lin.ravel())

    H, el, az, binArea = glyph.histogram2d(sph,bins=[100,200],norm='prob_binArea', weights=sph_lin)

    glyph.save_glyph(H,el,az,savePath="testGlyph.vtk")

    print("Glyph generated and saved.")

