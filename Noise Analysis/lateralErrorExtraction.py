"""
Analysis of lateral and axial noise

Developed for Bachelor's thesis
"""
__author__ = "Katarina Osvaldova"

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.odr import Model, Data, ODR
from misc import saveDictToJSON
from typing import Optional


def extractErrors(xs, ys, visualise=False):
    def orthogonalFit(xs: np.array, ys: np.array, a0: int, b0: int) -> tuple[int, int, bool]:
        """
        Orthogonal regression linear fit
            starting fit y = a0 * x + b0
        """

        def f(B, u):
            """Linear function y = B[0]*u + B[1]"""
            return B[0] * u + B[1]

        odr = ODR(Data(xs, ys), Model(f), beta0=[1, 1], maxit=5000)
        odr.set_job(fit_type=0)

        output = odr.run()
        FAIL = 'Sum of squares convergence' not in output.stopreason
        # if FAIL:
        #     output.pprint()

        return output.beta[0], output.beta[1], FAIL

    def orthogonalSignedDistance(x, y, a, b, c):
        """
        Distance of (x, y) point from line ax + by + c = 0
        """
        return (a * x + b * y + c) / (np.sqrt(a ** 2 + b ** 2))

    modelLSM = LinearRegression().fit(xs.reshape((-1, 1)), ys)

    fitA, fitB, notConverged = orthogonalFit(xs, ys, modelLSM.coef_[0], modelLSM.intercept_)

    if notConverged:
        # if ODR didn't get good results, take OLS
        fitA, fitB = modelLSM.coef_[0], modelLSM.intercept_
        if visualise:
            fig, ax = plt.subplots(1, 1)
            ax.scatter(xs, ys)
            xv = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(xv, xv * modelLSM.coef_[0] + modelLSM.intercept_, 'g', label="Least squares")
            ax.plot(xv, xv * fitA + fitB, 'r', label="Orthogonal")
            ax.legend()
            plt.show()

    return [orthogonalSignedDistance(xs[k], ys[k], fitA, -1, fitB) for k in range(len(xs))]


def runErrorExtraction(inputDict: dict = None, inputDictFilename: str = "sceneEdges.json",
                       saveResult: str = "errorsLateral.json", showHists: bool = False) -> Optional[dict]:
    if inputDict is None:
        try:
            with open(inputDictFilename) as file:
                inputDict = json.load(file)
        except FileNotFoundError:
            print("No edge data supplied for error extraction...")
            return None
    #  inputDict[f"{angle}_{distance[m]}_{uniqueScene}"] =
    #      {"left data": [[[int, int],],],   (for each file, a list of pairs (x, y) of border pixels)
    #       "right data": [[[int, int],],],  (for each file, a list of pairs (x, y) of border pixels)
    #       "left depths": [int],            (for each file one mean depth)
    #       "right depths": [int]}           (for each file one mean depth)

    dataList = []
    for i, scene in enumerate(inputDict):
        print(f"scene {i + 1}/{len(inputDict)}: {scene}")
        AL = np.array(inputDict[scene]["left data"])
        AL = AL.reshape(-1, AL.shape[-1])
        XL = AL[:, 1]
        YL = AL[:, 0]
        errorsL = extractErrors(XL, YL)

        AR = np.array(inputDict[scene]["right data"])
        AR = AR.reshape(-1, AR.shape[-1])
        XR = AR[:, 1]
        YR = AR[:, 0]
        errorsR = extractErrors(XR, YR)
        if showHists:
            plt.figure(scene)
            plt.hist(errorsR)
            plt.show()

        a, d, _, _, c = scene.split('_')
        dataList.append({"dist": float(d),
                         "angle": int(a),
                         "camera": c,
                         "left": {"dist": np.array(inputDict[scene]["left depths"]).mean(),
                                  "sd": np.std(errorsL),
                                  "errors": errorsL},
                         "right": {"dist": np.array(inputDict[scene]["right depths"]).mean(),
                                   "sd": np.std(errorsR),
                                   "errors": errorsR}})

    saveDictToJSON({"data": dataList}, saveResult)
    return {"data": dataList}


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    runErrorExtraction()







