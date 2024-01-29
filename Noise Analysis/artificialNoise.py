"""
Analysis of lateral and axial noise

Developed for Bachelor's thesis
"""
__author__ = "Katarina Osvaldova"

import numpy as np
from model import modelAxial, modelLateral
from misc import displayMatrixMasked
from model import NoiseModel


def addAxialNoise(noiseModel: NoiseModel, depthMap: np.ndarray, angleMap: np.ndarray):
    print("Adding axial noise")
    if depthMap.ndim != 2 or angleMap.ndim != 2 or depthMap.shape != angleMap.shape:
        raise Exception(f"Invalid dimensions ({depthMap.shape}, {angleMap.shape})")

    addNoise = np.vectorize(noiseModel.generateNoise)
    return addNoise(depthMap, angleMap) + depthMap


def addLateralNoise(noiseModel: NoiseModel, depthMap: np.ndarray, angleMap: np.ndarray):
    print("Adding lateral noise")
    if depthMap.ndim != 2 or angleMap.ndim != 2 or depthMap.shape != angleMap.shape:
        raise Exception(f"Invalid dimensions ({depthMap.shape}, {angleMap.shape})")

    noisy = depthMap.copy()

    for r in range(noisy.shape[0]):
        for c in range(noisy.shape[1]):
            dr = noiseModel.generateNoise(depthMap[r, c], angleMap[r, c])
            dc = noiseModel.generateNoise(depthMap[r, c], angleMap[r, c])
            newR = int(np.round(r + dr)[0])
            newC = int(np.round(c + dc)[0])
            if 0 <= newR < noisy.shape[0] and 0 <= newC < noisy.shape[1]:
                noisy[newR, newC] = depthMap[r, c]

    return noisy


def quantise(depthMap: np.ndarray, step: float):
    print("Quantising")

    def q(v, s):
        return step * np.round(v / s)

    runQuantisation = np.vectorize(q)
    return runQuantisation(depthMap, step)


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def exampleDepthMapAndAngleMap(h, w):
    result = np.full((h, w), 2000.0)

    mask = create_circular_mask(h, w, center=(h//4, w//4), radius=h//6)
    result[mask] = 1200.0

    angles = np.zeros_like(result)
    a = 50
    for i, c in enumerate(range(w//2, w//2 + w//3)):
        result[h//4:3*h//4, c] = 1500.0 + np.tan(np.radians(a)) * i * 2
        angles[h//4:3*h//4, c] = a

    return result, angles


if __name__ == '__main__':
    h, w = 200, 200
    syntheticDepthMap, synthAngles = exampleDepthMapAndAngleMap(1000, 1000)
    print(syntheticDepthMap.shape)
    synthAngles = np.zeros((1000, 1000))

    # displayMatrixMasked(syntheticDepthMap, figsize=(10, 3), units="Depth [mm]")

    axialModels = modelAxial()
    lateralModels = modelLateral()

    for q, camera in [(0.001, "PhoXif")]:

        axNoisy = quantise(addAxialNoise(axialModels[(camera, "fourier")], syntheticDepthMap, synthAngles), q)

        axlNoisy = addLateralNoise(lateralModels[camera], axNoisy, synthAngles)
        displayMatrixMasked(axlNoisy, title=camera, figsize=(10, 3), units="Depth [mm]", saveTo=f"noisy {camera}")

        # displayMatrixMasked(axlNoisy[h//4 + 5:3*h//4 - 5, w//2 + 5:w//2 + w//3 - 5],
        #                     title=camera, figsize=(10, 3), units="Depth [mm]", saveTo=f"noisy cut {camera}")
        #
        # displayMatrixMasked(axlNoisy[h//4 + 5:3*h//4 - 5, w//2 - 17:w//2 + 17],
        #                     title=camera, figsize=(10, 3), units="Depth [mm]", saveTo=f"noisy edge {camera}")
