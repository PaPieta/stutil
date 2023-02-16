import numpy as np
import skimage.io 
import matplotlib.pyplot as plt
import os
from datetime import datetime

from stutil.scale_space import ScaleSpace
from stutil import glyph
from stutil import volume

if __name__ == "__main__":

    file_path = '/zhome/5a/4/153708/2022_DANFIX_31_EXCHEQUER/analysis/processed_data/14-01-23_7012716-lfov/uint_8/vol_500cube.tiff'
    run_name = '6_58_scales_sigma_x0_25_filledHole'
    rho_scales = np.array([6,10,14,18,22,26,30,34,38,42,46,50,54,58])
    sigma_scales = np.array([6,10,14,18,22,26,30,34,38,42,46,50,54,58])/4
    # run_name = 'hole_fill_test'
    # rho_scales = np.array([1,2])
    # sigma_scales = np.array([1,2])/4
    glyph_full_sphere = True
    flipOpposites = True # If opposite directions should be flipped to the same direction
    detectAndFillHoles = True #If holes in the cheese should be detected and filled 
    cpu_num = 16
    block_size = 100
    holeThresh = 5
    minHoleSize = 50

    I = skimage.io.imread(file_path)
    I = (I.astype('float')/np.max(I))*255
    # I = I[:,200:400:,200:400].astype(float)
    print(f"Loaded image from: {file_path}")
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
    
    if detectAndFillHoles:
        I, I_mask = volume.holeFillGauss(I,thresh=holeThresh,minBlobSize=minHoleSize)
    else:
        I_mask = np.ones(shape(I))

    tensorScaleSpace = ScaleSpace(I,sigma_scales=sigma_scales,rho_scales=rho_scales,cpu_num=cpu_num,block_size=block_size)

    # Structure tensor scale space
    val,vec,lin,scale,scaleHist = tensorScaleSpace.calcFast()

    if flipOpposites:
        flipMask = vec[0,:] < 0
        flipMask = np.array([flipMask,flipMask,flipMask])
        vec[flipMask] = -vec[flipMask]
    # Save results
    np.save(os.path.join(run_result_path,'eigenvalues.npy'), val.astype(np.float16))
    np.save(os.path.join(run_result_path,'eigenvectors.npy'), vec.astype(np.float16))
    np.save(os.path.join(run_result_path,'scales.npy'), scale)
    np.save(os.path.join(run_result_path,'scaleHist.npy'), scaleHist)
    if detectAndFillHoles:
        np.save(os.path.join(run_result_path,'holeMask.npy'), I_mask)
    # Save rgba volume
    # rgba = volume.convertToColormap(vec, halfSphere=flipOpposites, weights=lin)
    rgba = volume.convertToIco(vec,  weights=lin, mask=I_mask)
    volume.saveRgbaVolume(rgba,savePath=os.path.join(run_result_path,'rgbaWeighted.tiff'))

    print("RGBA direction volume saved.")

    # Remove hole results from vec and lin
    I_mask3ch = np.array([I_mask, I_mask, I_mask])
    vec = vec[I_mask3ch].reshape(3,-1)
    lin = lin[I_mask].ravel()

    # Create and save glyph
    sph, sph_lin = glyph.orientationVec(vec, fullSphere=glyph_full_sphere, weights=lin)

    H, el, az, binArea = glyph.histogram2d(sph,bins=[100,200],norm='prob_binArea', weights=sph_lin)

    glyph.save_glyph(H,el,az,np.array([0,0]),savePath=os.path.join(run_result_path,'glyph.vtk'), flipColor=flipOpposites)

    print("Glyph generated and saved.")

