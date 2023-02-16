from structure_tensor import eig_special_3d, structure_tensor_3d, parallel_structure_tensor_analysis

import numpy as np
import time

class ScaleSpace:

    def __init__(self, volume: np.array, sigma_scales: list, rho_scales: list, cpu_num: int, block_size: int):
        """Class initialization.\n
        Params:\n
        volume - volume on which the structure tensor scale space should be calculated\n
        sigma_scales - values of the sigma parameter (noise scale)\n
        rho_scales - values of the rho parameter (integration scale)\n
        cpu_num - number of cpu cores used in st calculation\n
        block_size - size of a block that the image is divided into for parallel processing
        """
        self.volume = np.asarray(volume)
        self.scales_num = len(sigma_scales)

        assert self.scales_num == len(rho_scales), "Rho and sigma list should have the same length"
        self.sigma_scales = sigma_scales
        self.rho_scales = rho_scales

        self.cpu_num = cpu_num
        self.block_size = block_size

        #Allocate structure tensor output S array
        # self.S = np.empty((6, ) + self.volume.shape, dtype=self.volume.dtype)

    def calcFast(self):
        """Fast and low memory scale space calculation by merging scales at each step. Returns only the most optimal solution.
        """

        #initilaize arrays: eignevectors, eigenvalues, linearity score and scale histograms
        valScale,vecScale, linScale = [np.empty((3,)+self.volume.shape, dtype=self.volume.dtype) for _ in range(3)]
        valFin,vecFin, linFin = [np.empty((3,)+self.volume.shape, dtype=self.volume.dtype) for _ in range(3)]
        scaleHist = np.zeros((len(self.rho_scales)+1,len(self.rho_scales)), dtype=np.int32)
        scaleHist[0,:] = np.array(self.rho_scales)
        #array with original scale index used
        scaleIdx = np.zeros(self.volume.shape, dtype=np.int8)
        #array with boolean swap indices
        swapIdx = np.empty((3,)+self.volume.shape, dtype=bool)
        print(f"Initialization finished, starting scale space structure tensor calculation.")
        
        for i in range(self.scales_num):
            t0 = time.time()

            # self.S = structure_tensor_3d(self.volume, self.sigma_scales[i], self.rho_scales[i], out = self.S)
            # valScale, vecScale = eig_special_3d(self.S, full=False)

            # linScale = (valScale[1]-valScale[0])/valScale[2]

            # For parallel the order of eigenvalues is flipped, so that eig0 > eig1 > eig2
            vecScale, valScale = parallel_structure_tensor_analysis(self.volume, self.sigma_scales[i], self.rho_scales[i], devices=self.cpu_num*['cpu'], block_size=self.block_size, truncate=3.0, include_all_eigenvalues=False)
            
            linScale = (valScale[1]-valScale[2])/valScale[0]

            if i == 0:
                valFin = np.copy(valScale)
                vecFin = np.copy(vecScale)
                linFin = np.copy(linScale)
            else:
                swapIdx = np.repeat(linScale[None,:]>linFin[None,:],3,axis=0)
                valFin[swapIdx] = valScale[swapIdx]
                vecFin[swapIdx] = vecScale[swapIdx]
                linFin[swapIdx[0]] = linScale[swapIdx[0]]
                scaleIdx[swapIdx[0]] = i

            _,counts = np.unique(scaleIdx,return_counts=True)
            scaleHist[i+1,:i+1] = counts

            t1 = time.time()
            print(f"Scale {i} finished in {t1-t0}.")

        del valScale, vecScale, linScale, swapIdx

        # Fix pole order from ZYX to XYZ TODO: check that
        # valFin = valFin[[2,1,0],:]
        vecFin = vecFin[[2,1,0],:]

        print("Scale space calculation finished")

        return valFin,vecFin,linFin,scaleIdx,scaleHist