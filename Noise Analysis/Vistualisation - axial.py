"""
Analysis of lateral and axial noise

Developed for Bachelor's thesis
"""
__author__ = "Katarina Osvaldova"

import json
import numpy as np
from matplotlib import pyplot as plt
from annotation import runAnnotation
from axialErrorExtraction import runErrorExtraction


readErrorData = True
startWithAnnotation = False

if readErrorData:
    with open("Data/errorsAxial.json") as file:
        data = json.load(file)["data"]
else:
    errorData = runErrorExtraction(runAnnotation() if startWithAnnotation else None)

    data = errorData["data"]
    # data = [{"dist": float,
    #          "angle": int,
    #          "camera": str,
    #          "dist measured": float,
    #          "angle measured": float,
    #          "stds": [{"method": str,
    #                    "std": float }]}]

for method in sorted(set(map(lambda sceneStds: sceneStds['method'], data[0]['stds']))):
    cams = sorted(set(map(lambda sceneDict: sceneDict['camera'], data)))
    fig, ax = plt.subplots(len(cams), 2, sharey=True, sharex='col', figsize=(9, 7), )

    for i, cameraStr in enumerate(cams):

        selectedData = list(filter(lambda x: (x["camera"] == cameraStr), data))
        angles = np.array(list(map(lambda x: x["angle measured"], selectedData)))
        dists = np.array(list(map(lambda x: x["dist measured"], selectedData)))
        sds = np.array(list(map(lambda x: list(filter(lambda y: y["method"] == method, x["stds"]))[0]["std"],
                                selectedData)))

        for j, x, c in ((0, angles, dists), (1, dists, angles)):
            ax[i][j].set_title(cameraStr)
            if j == 0:
                ax[i][j].set_xticks(range(0, int(np.ceil(angles.max()/10) * 10) + 1, 10))
            cax = ax[i][j].scatter(x, sds, c=c, cmap='cividis' if j == 0 else 'plasma', alpha=0.6,
                                   vmin=500 if j == 0 else 0, vmax=2250 if j == 0 else np.ceil(angles.max()/10) * 10)
            if i == 0:
                fig.colorbar(cax, ax=ax[:, j], shrink=0.5, label="Angle [degree]" if j == 1 else "Distance [mm]")

    fig.canvas.manager.set_window_title(f'{method}')
    fig.supylabel('Standard deviation of axial noise [mm]')
    ax[len(cams)-1][0].set_xlabel("Angle [degree]")
    ax[len(cams)-1][1].set_xlabel("Distance [mm]")
    # plt.savefig(f'axial {method}.png')
    plt.show()
