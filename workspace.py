import numpy as np
import skimage.io
import os
from datetime import datetime
import shutil

import hydra

from stutil import glyph
from stutil import volume

from stss import st

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s : %(levelname)s : %(module)s : %(message)s",
    datefmt="%I:%M:%S",
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="default_config.yaml", version_base=None)
def run_st(config):
    data_hparams = config.data
    st_hparams = config.st
    pre_hparams = config.pre
    post_hparams = config.post
    hydra_output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    file_path = os.path.join(data_hparams.base_path, data_hparams.data_path)
    run_name = f"s{st_hparams.sigma_start}-{st_hparams.sigma_step}-{st_hparams.sigma_end}_{data_hparams.output_name_app}"
    logger.info(f"Running {run_name} on {file_path}")
    logger.info(f"Logs temporarily saved to {hydra_output_path}")

    img = skimage.io.imread(file_path).astype("float")
    img = img / np.max(img)
    # img = img[:,200:400:,200:400].astype(float)
    logger.info(f"Loaded image from: {file_path}")
    logger.info(f"Image size: {img.shape}")

    # Create general result folder for data instance
    img_result_path = os.path.splitext(file_path)[0]
    if not os.path.isdir(img_result_path):
        os.makedirs(img_result_path)
    # Create result folder for this setup
    setup_result_path = os.path.join(img_result_path, run_name)
    if not os.path.isdir(setup_result_path):
        os.makedirs(setup_result_path)
    # Create result folder for this instance
    run_result_path = os.path.join(
        setup_result_path, datetime.today().strftime("%Y-%m-%d %H:%M")
    )
    if not os.path.isdir(run_result_path):
        os.makedirs(run_result_path)
    logger.debug("Folder setup complete")

    if pre_hparams.detectAndFillHoles:
        # Fat and big air bubbles
        img, img_mask2 = volume.holeFillMrfGauss(
            img,
            meanHole=pre_hparams.fatMean,
            meanObj=pre_hparams.cheeseMean,
            beta=pre_hparams.mrfBeta,
            minBlobSize=pre_hparams.minFatSize,
        )
        # Small air bubbles
        img, img_mask = volume.holeFillGauss(
            img,
            thresh=pre_hparams.holeThresh,
            minBlobSize=pre_hparams.minHoleSize,
            dilSize=pre_hparams.dilSize,
        )
        # Combine
        img_mask = np.bitwise_and(img_mask, img_mask2)
    else:
        img_mask = np.ones(np.shape(img)).astype(np.bool_)
        # img_mask = img>91
        # img_mask = img>140

    logger.debug("Mask prepared")

    # Structure tensor scale space
    sigma_scales = np.arange(
        st_hparams.sigma_start, st_hparams.sigma_end, st_hparams.sigma_step
    )
    if st_hparams.rho_enabled:
        rho_scales = sigma_scales * 2
    else:
        rho_scales = None
    # Old way
    # tensorScaleSpace = ScaleSpace(img,sigma_scales=sigma_scales,rho_scales=rho_scales,correctScale=correctScale,cpu_num=cpu_num,block_size=block_size)
    # S,val,vec,discr,scale = tensorScaleSpace.calcFast()
    # New way
    S, val, vec, scale = st.scale_space(
        img,
        sigma_scales,
        ring_filter=st_hparams.ring_filter,
        rho_list=rho_scales,
        correctScale=st_hparams.correctScale,
        gamma=st_hparams.gamma,
    )

    meanVal = np.mean(val, axis=0)
    anis = (
        np.sqrt(3 / 2)
        * np.sqrt(
            (val[0] - meanVal) ** 2 + (val[1] - meanVal) ** 2 + (val[2] - meanVal) ** 2
        )
        / np.sqrt(val[0] ** 2 + val[1] ** 2 + val[2] ** 2)
    )

    discr = S[0] + S[1] + S[2]

    if post_hparams.flip_opposites:
        flipMask = vec[0, :] < 0
        flipMask = np.array([flipMask, flipMask, flipMask])
        vec[flipMask] = -vec[flipMask]
    # Save results
    np.save(os.path.join(run_result_path, "S.npy"), S.astype(np.float16))
    np.save(os.path.join(run_result_path, "discr.npy"), discr.astype(np.float16))
    np.save(os.path.join(run_result_path, "eigenvalues.npy"), val.astype(np.float16))
    np.save(os.path.join(run_result_path, "eigenvectors.npy"), vec.astype(np.float16))
    np.save(os.path.join(run_result_path, "scales.npy"), scale.astype(np.float16))
    # np.save(os.path.join(run_result_path,'scaleHist.npy'), scaleHist)
    if pre_hparams.detectAndFillHoles:
        np.save(os.path.join(run_result_path, "holeMask.npy"), img_mask)
    # Save rgba volume
    # rgba = volume.convertToColormap(vec, halfSphere=post_hparams.flip_opposites, weights=anis)
    rgba = volume.convertToIco(vec, weights=anis, mask=img_mask)
    volume.saveRgbaVolume(
        rgba, savePath=os.path.join(run_result_path, "rgbaWeighted.tiff")
    )

    logger.info("RGBA direction volume saved.")

    # Remove hole results from vec and anis
    img_mask3ch = np.array([img_mask, img_mask, img_mask])
    vec = vec[img_mask3ch].reshape(3, -1)
    anis = anis[img_mask].ravel()

    # Create and save glyph
    sph, sph_anis = glyph.orientationVec(
        vec, fullSphere=post_hparams.glyph_full_sphere, weights=anis
    )

    H, el, az, binArea = glyph.histogram2d(
        sph, bins=[100, 200], norm="prob_binArea", weights=sph_anis
    )

    glyph.save_glyph(
        H,
        el,
        az,
        np.array([0, 0]),
        savePath=os.path.join(run_result_path, "glyph.vtk"),
        flipColor=post_hparams.flip_opposites,
    )

    logger.info("Glyph generated and saved.")

    # return hydra workspace and data save path
    logger.info(f"Results saved to {run_result_path}")
    logger.info(f"Hydra output saved to {hydra_output_path}")

    # Create log folder inside run_result_path
    log_result_path = os.path.join(run_result_path, "log")
    # Move all files from hydra_output_path to log_result_path
    shutil.copytree(hydra_output_path, log_result_path)


if __name__ == "__main__":
    run_st()
