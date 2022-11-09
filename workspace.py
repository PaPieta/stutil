import numpy as np
import skimage.io 
import matplotlib.pyplot as plt
import os
from datetime import datetime

from scale_space import ScaleSpace
import glyph
import volume

if __name__ == "__main__":

    file_path = '/work3/papi/Cheese/cheese_10X-40kV-air-45s_recon_cut.tif'
    run_name = '6_scales'
    rho_scales = [1,2,4,8,12,16]
    sigma_scales = [1,2,4,8,12,16]

    I = skimage.io.imread(file_path)
    I = (I.astype('float')/np.max(I))*255
    # I = I[:,200:400:,200:400].astype(float)

    print(f"Image size: {I.shape}")

    # Create general result folder
    img_result_path = os.path.splitext(file_path)[0]
    if not os.path.isdir(img_result_path):
        os.makedirs(img_result_path)
    # Create results for this run
    run_result_path = os.path.join(img_result_path,run_name)
    if not os.path.isdir(run_result_path):
        os.makedirs(run_result_path)
    # Save setup file
    setup_path = os.path.join(run_result_path,'setup.txt')
    time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    with open(setup_path, 'w') as f:
        f.write(f'Date: {time}')
        f.write('\nRho scales: ')
        f.write("".join([f'{str(i)} 'for i in rho_scales]))
        f.write('\nSigma scales: ')
        f.write("".join([f'{str(i)} 'for i in sigma_scales]))


    print("Folder setup complete")

    tensorScaleSpace = ScaleSpace(I,sigma_scales=[1,2,4,8,12,16],rho_scales=[1,2,4,8,12,16])

    # Structure tensor scale space
    val,vec,lin,scale,scaleHist = tensorScaleSpace.calcFast()
    # Save results
    np.save(os.path.join(run_result_path,'eigenvalues.npy'), val)
    np.save(os.path.join(run_result_path,'eigenvectors.npy'), vec)
    np.save(os.path.join(run_result_path,'scales.npy'), scale)
    np.save(os.path.join(run_result_path,'scaleHist.npy'), scaleHist)

    # Save rgba volume
    rgba = volume.convertToColormap(vec,weights=lin)
    volume.saveRgbaVolume(rgba,savePath=os.path.join(run_result_path,'rgbaWeighted.tiff'))

    print("RGBA direction volume saved.")

    # Create and save glyph
    sph, sph_lin = glyph.orientationVec(vec.reshape(3,-1), [2,1,0], fullSphere=True, weights=lin.ravel())

    H, el, az, binArea = glyph.histogram2d(sph,bins=[100,200],norm='prob_binArea', weights=sph_lin)

    glyph.save_glyph(H,el,az,savePath=os.path.join(run_result_path,'glyph.vtk'))

    print("Glyph generated and saved.")

