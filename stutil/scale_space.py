from structure_tensor import eig_special_3d, structure_tensor_3d, parallel_structure_tensor_analysis

from stutil import newST

import numpy as np
import time

class ScaleSpace:

    def __init__(self, volume: np.array, sigma_scales: list, rho_scales: list, correctScale: bool, cpu_num: int, block_size: int):
        """Class initialization.\n
        Params:\n
        volume - volume on which the structure tensor scale space should be calculated\n
        sigma_scales - values of the sigma parameter (noise scale)\n
        rho_scales - values of the rho parameter (integration scale)\n
        correctScale - if True, the scale map is normalized to correct for filter alignment inaccuracies\n
        cpu_num - number of cpu cores used in st calculation\n
        block_size - size of a block that the image is divided into for parallel processing
        """
        self.volume = np.asarray(volume)
        self.scales_num = len(sigma_scales)

        assert self.scales_num == len(rho_scales), "Rho and sigma list should have the same length"
        self.sigma_scales = sigma_scales
        self.rho_scales = rho_scales

        self.correctScale = correctScale
        # assert self.discr == "lin" or self.discr == 'negSph' or self.discr == 'fAnis' or self.discr == 'normTrS', "Allowed values for discr are \"lin\" \"negSph\" \"fAnis\" and \"normTrS\""

        self.cpu_num = cpu_num
        self.block_size = block_size

        #Allocate structure tensor output S array
        self.S = np.empty((6, ) + self.volume.shape, dtype=self.volume.dtype)

    def calcFast(self):
        """Fast and low memory scale space calculation by merging scales at each step. Returns only the most optimal solution.
        """

        #initilaize arrays: eignevectors, eigenvalues, linearity score and scale histograms
        SFin = np.empty((6, ) + self.volume.shape, dtype=self.volume.dtype)
        # valScale,vecScale, discrScale = [np.empty((3,)+self.volume.shape, dtype=self.volume.dtype) for _ in range(3)]
        valFin,vecFin, discrFin = [np.empty((3,)+self.volume.shape, dtype=self.volume.dtype) for _ in range(3)]
        #array with original scale index used
        scaleFin = np.ones(self.volume.shape, dtype=float) * self.sigma_scales[0]
        #array with boolean swap indices
        swapIdx = np.empty((3,)+self.volume.shape, dtype=bool)
        print(f"Initialization finished, starting scale space structure tensor calculation.")
        
        for i in range(self.scales_num):
            t0 = time.time()

            # self.S = structure_tensor_3d(self.volume, self.sigma_scales[i], self.rho_scales[i])
            self.S = newST.structure_tensor_3d_new(self.volume, self.sigma_scales[i], self.rho_scales[i])
            # valScale, vecScale = eig_special_3d(self.S, full=False)


            # For parallel the order of eigenvalues is flipped, so that eig0 > eig1 > eig2 
            # S, vecScale, valScale = parallel_structure_tensor_analysis(self.volume, self.sigma_scales[i], self.rho_scales[i], structure_tensor=True, devices=self.cpu_num*['cpu'], block_size=self.block_size, truncate=3.0, include_all_eigenvalues=False)
         
            discrScale = self.S[0] + self.S[1] + self.S[2]

            if i == 0:
                # valFin = np.copy(valScale)
                # vecFin = np.copy(vecScale)
                discrFin = np.copy(discrScale)
                SFin = np.copy(self.S)
            else:
                swapIdx = np.repeat(discrScale[None,:]>discrFin[None,:],6,axis=0)
                SFin[swapIdx] = self.S[swapIdx]
                # swapIdx = np.repeat(discrScale[None,:]>discrFin[None,:],3,axis=0)
                # valFin[swapIdx] = valScale[swapIdx]
                # vecFin[swapIdx] = vecScale[swapIdx]
                scaleFin[swapIdx[0]] = self.sigma_scales[i]
                discrFin[swapIdx[0]] = discrScale[swapIdx[0]]

            t1 = time.time()
            print(f"Scale {i} finished in {t1-t0}.")

        # del valScale, vecScale, discrScale, swapIdx
        del discrScale

        valFin, vecFin = eig_special_3d(SFin, full=False)

        # Fix pole order from ZYX to XYZ TODO: check that
        # valFin = valFin[[2,1,0],:]
        vecFin = vecFin[[2,1,0],:]


        # Use fractional anisotorpy as a final measure, despite using Trace of S for scale selection
        if self.correctScale == True:

            #New normalization for 3D
            lin = (valFin[1]-valFin[0])/valFin[2]
            plan = (valFin[2]-valFin[1])/valFin[2]
            sph = valFin[0]/valFin[2]
            m = 1.07; l = 0.65; p = 1; s = 0.5
            scaleFin = scaleFin/(m*(l*lin+p*plan+s*sph))

        print("Scale space calculation finished")

        # return valFin,vecFin,discrFin,scaleIdx,scaleHist
        return SFin, valFin,vecFin,discrFin,scaleFin