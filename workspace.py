import numpy as np
import skimage.io 
import matplotlib.pyplot as plt
import os
from datetime import datetime

from stutil.scale_space import ScaleSpace
from stutil import glyph
from stutil import volume

if __name__ == "__main__":

    file_path = '/zhome/5a/4/153708/2022_DANFIX_31_EXCHEQUER/analysis/processed_scans/autumnBatch/36-2-22-3/uint_8/vol_500cube_smooth2sigma.tiff'
    run_name = '2_14_scales_rho_x001_eig_newST_01_step'
    # rho_scales = np.array([12])*0.001
    # sigma_scales = np.array([12])
    sigma_scales = np.arange(2,14.1,0.1)
    rho_scales = sigma_scales*0.01
    # run_name = 'hole_fill_test'
    # rho_scales = np.array([6])*0.01
    # sigma_scales = np.array([6])
    correctScale = False
    glyph_full_sphere = True
    flipOpposites = True # If opposite directions should be flipped to the same direction
    cpu_num = 16
    block_size = 100

    detectAndFillHoles = True #If holes in the cheese should be detected and filled 
    holeThresh = 75
    # holeThresh = 100
    minHoleSize = 50
    # minHoleSize = 20
    dilSize=5
    # dilSize=3
    fatMean = 100
    # fatMean = 50
    cheeseMean = 120
    mrfBeta = 5
    # minFatSize = 10000
    minFatSize = 3000

    I = skimage.io.imread(file_path).astype('float')
    I = (I/np.max(I))
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

        f.write(f'\n\nScale correction applied: {correctScale}')
        f.write(f'\nGlyph full sphere: {glyph_full_sphere}')
        f.write(f'\nFlip opposites: {flipOpposites}')
        f.write(f'\nCpu num: {cpu_num}')
        f.write(f'\nBlock size: {block_size}')
        f.write(f'\nDetect and fill holes: {detectAndFillHoles}')
        f.write(f'\nHole intensity threshold: {holeThresh}')
        f.write(f'\nHole size threshold: {minHoleSize}')
        f.write(f'\nHole dilation filter size: {dilSize}')
        f.write(f'\nMean fat pixel intensity: {fatMean}')
        f.write(f'\nMean cheese pixel intensity: {cheeseMean}')
        f.write(f'\nMRF 2-clique beta: {mrfBeta}')
        f.write(f'\nFat size threshold: {minFatSize}')
        f.write('\n\nSaved scale value, not scale index, no scaleHist.')

    print("Folder setup complete")
    
    if detectAndFillHoles:
        # Fat and big air bubbles
        I, I_mask2 = volume.holeFillMrfGauss(I,meanHole=fatMean,meanObj=cheeseMean,beta=mrfBeta,minBlobSize=minFatSize)
        # Small air bubbles
        I, I_mask = volume.holeFillGauss(I,thresh=holeThresh,minBlobSize=minHoleSize, dilSize=dilSize)
        # Combine
        I_mask = np.bitwise_and(I_mask, I_mask2)
    else:
        # I_mask = np.ones(np.shape(I)).astype(np.bool_)
        # I_mask = I>91
        I_mask = I>140

    print("Mask prepared")

    tensorScaleSpace = ScaleSpace(I,sigma_scales=sigma_scales,rho_scales=rho_scales,correctScale=correctScale,cpu_num=cpu_num,block_size=block_size)

    # Structure tensor scale space
    # val,vec,lin,scale,scaleHist = tensorScaleSpace.calcFast()
    S,val,vec,discr,scale = tensorScaleSpace.calcFast()

    meanVal = np.mean(val, axis=0)
    anis = np.sqrt(3/2) * np.sqrt((val[0]-meanVal)**2+(val[1]-meanVal)**2+(val[2]-meanVal)**2)/np.sqrt(val[0]**2+val[1]**2+val[2]**2)

    if flipOpposites:
        flipMask = vec[0,:] < 0
        flipMask = np.array([flipMask,flipMask,flipMask])
        vec[flipMask] = -vec[flipMask]
    # Save results
    np.save(os.path.join(run_result_path,'S.npy'), S.astype(np.float16))
    np.save(os.path.join(run_result_path,'discr.npy'), S.astype(np.float16))
    np.save(os.path.join(run_result_path,'eigenvalues.npy'), val.astype(np.float16))
    np.save(os.path.join(run_result_path,'eigenvectors.npy'), vec.astype(np.float16))
    np.save(os.path.join(run_result_path,'scales.npy'), scale.astype(np.float16))
    # np.save(os.path.join(run_result_path,'scaleHist.npy'), scaleHist)
    if detectAndFillHoles:
        np.save(os.path.join(run_result_path,'holeMask.npy'), I_mask)
    # Save rgba volume
    # rgba = volume.convertToColormap(vec, halfSphere=flipOpposites, weights=anis)
    rgba = volume.convertToIco(vec,  weights=anis, mask=I_mask)
    volume.saveRgbaVolume(rgba,savePath=os.path.join(run_result_path,'rgbaWeighted.tiff'))

    print("RGBA direction volume saved.")

    # Remove hole results from vec and anis
    I_mask3ch = np.array([I_mask, I_mask, I_mask])
    vec = vec[I_mask3ch].reshape(3,-1)
    anis = anis[I_mask].ravel()

    # Create and save glyph
    sph, sph_anis = glyph.orientationVec(vec, fullSphere=glyph_full_sphere, weights=anis)

    H, el, az, binArea = glyph.histogram2d(sph,bins=[100,200],norm='prob_binArea', weights=sph_anis)

    glyph.save_glyph(H,el,az,np.array([0,0]),savePath=os.path.join(run_result_path,'glyph.vtk'), flipColor=flipOpposites)

    print("Glyph generated and saved.")

