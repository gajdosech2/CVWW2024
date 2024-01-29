"""
Analysis of lateral and axial noise

Developed for Bachelor's thesis
"""
__author__ = "Katarina Osvaldova"

import json
import numpy as np
from matplotlib import pyplot as plt
from annotation import runAnnotation
from borderDataExtraction import runEdgeExtraction
from lateralErrorExtraction import runErrorExtraction


readErrorData = True
startWithAnnotation = False
startWithEdgeExtraction = True

if readErrorData:
    with open("Data/errorsLateral.json") as file:
        data = json.load(file)["data"]
else:
    errorData = runErrorExtraction(runEdgeExtraction(runAnnotation()
                                                     if startWithAnnotation else None)
                                   if (startWithEdgeExtraction or startWithAnnotation) else None)
    data = errorData["data"]
    # data = [{"dist": float,
    #          "angle": int,
    #          "camera": str,
    #          "left": {"dist": float,
    #                   "sd": float,
    #                   "errors": [float]},
    #          "right": {"dist": float,
    #                    "sd": float,
    #                    "errors": [float]}}]

cams = sorted(set(map(lambda sceneDict: sceneDict['camera'], data)))
fig, ax = plt.subplots(len(cams), 2, sharey=True, sharex='col', figsize=(9, 7))
for i, cameraStr in enumerate(cams):
    selectedData = list(filter(lambda x: (x["camera"] == cameraStr), data))
    angles = np.array(list(map(lambda x: (x["angle"], x["angle"]), selectedData))).flatten()
    dists = np.array(list(map(lambda x: (x["left"]["dist"], x["right"]["dist"]), selectedData))).flatten()
    sds = np.array(list(map(lambda x: (x["left"]["sd"], x["right"]["sd"]), selectedData))).flatten()

    for j, x, c in ((0, (angles + np.random.normal(0, 1, size=angles.shape)), dists), (1, dists, angles)):
        ax[i][j].set_title(cameraStr)
        if j == 0:
            ax[i][j].set_xticks(range(0, 81, 10))
            ax[i][j].set_xticks(range(5, 85, 10), minor=True)
            ax[i][j].grid(which='minor', axis='x')
        cax = ax[i][j].scatter(x, sds, c=c, cmap='cividis' if j == 0 else 'plasma', alpha=0.6,
                               vmin=500 if j == 0 else 0, vmax=2250 if j == 0 else 80)
        if i == 0:
            fig.colorbar(cax, ax=ax[:, j], shrink=0.5, label="Angle [degree]" if j == 1 else "Distance [mm]")


fig.supylabel('Standard deviation of lateral noise [pixel]')
ax[len(cams)-1][0].set_xlabel("Angle [degree]")
ax[len(cams)-1][1].set_xlabel("Distance [mm]")
# plt.savefig('lateral comparison.png')
plt.show()
