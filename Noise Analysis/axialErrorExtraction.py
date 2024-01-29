"""
Analysis of lateral and axial noise

Developed for Bachelor's thesis
"""
__author__ = "Katarina Osvaldova"

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
from scipy.stats import norm
from scipy.stats import kstest, ttest_1samp
from sklearn.linear_model import LinearRegression
from misc import readDepthMap, saveDictToJSON, applyFilter
from typing import Optional


def extractErrors(imgs: np.ndarray, method: str, kernel: np.ndarray = None, visualise: bool = False,
                  showHists: bool = False, testNormality=False, scene="", testMean0=True):
    averageImage = ma.masked_equal(imgs, 0).mean(axis=0)
    averageImage.mask = np.full_like(averageImage, False)
    averageImage = healZero(np.array(averageImage))
    if method == "plane":
        y, x = np.indices(averageImage.shape)
        X = np.array((x.flatten(), y.flatten())).T
        Y = averageImage.flatten()

        modelLSM = LinearRegression().fit(X[Y != 0], Y[Y != 0])
        groundTruth = modelLSM.predict(X).reshape(averageImage.shape)
    elif method == "fourier":
        fshifted = np.fft.fftshift(np.fft.fft2(averageImage))  # compute transform and shift low frequencies to center
        ytemp, xtemp = np.indices((averageImage.shape[0], averageImage.shape[1]))
        dists = np.sqrt((((xtemp - (averageImage.shape[1] / 2)) / (averageImage.shape[1] / 2))**2) +
                        ((ytemp - (averageImage.shape[0] / 2)) / (averageImage.shape[0] / 2))**2)  # euclidean distance adjusted to rectangular shape
        fshifted[dists > 1.1] = 0  # apply mask to filter frequencies (low-pass)
        groundTruth = np.fft.ifft2(np.fft.ifftshift(fshifted))  # shift back centering and decode
        groundTruth = np.abs(groundTruth)  # for conversion from complex numbers

    elif method == "filter" and kernel is not None:
        groundTruth = applyFilter(averageImage, kernel)
    else:
        raise Exception("No kernel was supplied for filtering" if (method == "filter" and kernel is None)
                        else f"{method} is not a valid method for ground truth image creation")

    if visualise:
        plt.figure(f'{scene} - {method}')
        plt.subplot(131), plt.imshow(ma.masked_equal(averageImage, 0), cmap='viridis')
        plt.colorbar(label="Depth [mm]")
        plt.title('Average depth map'), plt.xticks([]), plt.yticks([])

        plt.subplot(132), plt.imshow(ma.masked_equal(groundTruth, 0), cmap='viridis')
        plt.colorbar(label="Depth [mm]")
        plt.title('Fit for ground truth'), plt.xticks([]), plt.yticks([])

        errorImage = ma.array(imgs[0] - ma.masked_equal(groundTruth, 0), mask=(imgs[0] == 0))
        cmax = np.maximum(errorImage.max(), np.abs(errorImage.min()))
        plt.subplot(133), plt.imshow(errorImage, cmap='seismic', vmin=-cmax, vmax=cmax)
        plt.colorbar(label="Depth error [mm]")
        plt.title('Example noise'), plt.xticks([]), plt.yticks([])

        plt.show()

    errs = ma.masked_equal(imgs, 0) - ma.masked_equal(groundTruth, 0)
    if isinstance(errs.mask, np.bool_):
        errs = errs.flatten()
    else:
        errs = errs[~errs.mask].data

    if showHists:

        b1, b2 = errs.min(), errs.max()
        plt.figure(f'{scene} - {method} (mean: {errs.mean()}, std: {errs.std()})', figsize=(4, 3))
        plt.subplots_adjust(bottom=0.25)
        plt.hist(errs, bins='fd', density=True)
        xx = np.linspace(b1, b2, 500)
        plt.plot(xx, norm.pdf(xx, errs.mean(), errs.std()), 'k--', linewidth=1, alpha=0.8)
        plt.yticks([])
        plt.xlabel("Depth error [mm]")
        plt.show()

    if testNormality:
        pValue = kstest(errs, norm.cdf, args=(errs.mean(), errs.std())).pvalue
        print(f'\t\tNormality {"rejected" if pValue < 0.05 else "accepted"} with p-value {pValue}')
    if testMean0:
        pValue = ttest_1samp(errs, 0).pvalue
        print(f'\t\tMean equal to 0 {"rejected" if pValue < 0.05 else "accepted"} with p-value {pValue} '
              f'(mean calculated as {errs.mean()})')

    return errs, groundTruth


def healZero(img: np.ndarray, neighbourhood=1) -> np.ndarray:
    toReturn = img.copy()
    source = ma.masked_equal(img, 0)
    rs, cs = np.where(img == 0)
    for i, r in enumerate(rs):
        c = cs[i]
        r1 = np.maximum(0, r-neighbourhood)
        r2 = np.minimum(img.shape[0], r+neighbourhood+1)
        c1 = np.maximum(0, c - neighbourhood)
        c2 = np.minimum(img.shape[1], c + neighbourhood + 1)
        replaceWith = source[r1:r2, c1:c2].mean()
        if isinstance(replaceWith, ma.core.MaskedConstant):
            return healZero(img, neighbourhood+1)
        toReturn[r, c] = replaceWith
    return toReturn


def runErrorExtraction(inputDict: dict = None, inputDictFilename: str = "annotatedScenes.json",
                       saveResult: str = "errorsAxial.json", showHists: bool = False,
                       visualise: bool = False,
                       testNormality: bool = False) -> Optional[dict]:
    matplotlib.use('TkAgg')
    if inputDict is None:
        try:
            with open(inputDictFilename) as file:
                inputDict = json.load(file)
        except FileNotFoundError:
            print("No annotation data supplied for error extraction...")
            return None
    #   inputDict[f"{angle}_{distance[m]}_{uniqueScene}"] = {"dir": str,
    #                                                        "filenames": list[str],
    #                                                        "maskClick": (x, y),
    #                                                        "fillClick": (x, y),
    #                                                        "binCoords": (A.x, A.y, B.x, B.y, C.x, C.y, D.x, D.y)
    #                                                        "innerCoords": (E.x, E.y, F.x, F.y)}
    #         .______C
    #  .______B      |
    #  |  E___|___.  |
    #  |  |   |   |  |
    #  |  .___|___F  |
    #  A______|      |
    #         D______.

    dataList = []
    for i, scene in enumerate(sorted(inputDict.keys(), reverse=True)):
        print(f"scene {i + 1}/{len(inputDict)}: {scene}")

        innerCoords = inputDict[scene]["innerCoords"]
        if len(innerCoords) == 0:
            continue
        x1, y1, x2, y2 = innerCoords

        imgCutOuts = np.array([readDepthMap(f"{inputDict[scene]['dir']}/{filename}")[y1:y2, x1:x2]
                               for filename in inputDict[scene]["filenames"]])
        a, d, _, _, cam = scene.split('_')
        am = a
        dm = d
        sceneData = []
        for method, kernelDescription, kernel in (
                                                  ("plane", "", None),
                                                  ("fourier", "", None),
                                                  ("filter", "3x3", np.ones((3, 3)) / 9),
                                                  ("filter", "5x5", np.ones((5, 5)) / 25),
                                                  ("filter", "3x3-2", np.array([[1, 1, 1],
                                                                                [1, 2, 1],
                                                                                [1, 1, 1]]) / 10),
                                                  ("filter", "3x3-4", np.array([[1, 2, 1],
                                                                                [2, 4, 2],
                                                                                [1, 2, 1]]) / 16)
                                                  ):
            errs, fit = extractErrors(imgCutOuts, method, kernel, scene=scene, showHists=showHists, visualise=visualise,
                                      testNormality=testNormality)
            sceneData.append({"method": method + kernelDescription,
                              "std": errs.std()})
            print(f"\t- {method}{kernelDescription}")
            if method == "plane":
                dm = fit.mean()
                r, c = fit.shape
                r //= 2

                if cam[-1] == '1':
                    factor = 285 / 500
                elif cam[-1] == '2':
                    factor = 185 / 500
                else:
                    factor = 560 / 500
                factor *= 1000
                c *= (dm / factor)
                am = np.degrees(np.arctan2(np.abs(fit[r, 0] - fit[r, -1]), c))

        dataList.append({"dist": float(d),
                         "angle": int(a),
                         "camera": cam,
                         "dist measured": dm,
                         "angle measured": am,
                         "stds": sceneData})

    saveDictToJSON({"data": dataList}, saveResult)
    return {"data": dataList}
