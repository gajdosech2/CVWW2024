"""
Analysis of lateral and axial noise

Developed for Bachelor's thesis
"""
__author__ = "Katarina Osvaldova"

import json
import numpy as np
import matplotlib
from scipy import stats
from scipy.stats import norm
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from annotation import runAnnotation
import axialErrorExtraction
import lateralErrorExtraction
from borderDataExtraction import runEdgeExtraction
from copy import deepcopy
import plotly.graph_objs as go


np.random.seed(42)


class NoiseModel:
    def __init__(self, model: LinearRegression, poly: PolynomialFeatures):
        self.model = deepcopy(model)
        self.poly = deepcopy(poly)  # distance, angle
        self.interpretation = f"({model.intercept_:.2e}) + " + \
                              " + ".join(map(lambda p: f"({p[0]:.2e}*{p[1]})",
                                             zip(model.coef_, poly.get_feature_names_out(('d', 'a')))))

    def __str__(self):
        return self.interpretation

    __repr__ = __str__

    def predictStd(self, distance: float, angle: float):
        return self.model.predict(self.poly.fit_transform(np.array([[distance, angle]])))

    def generateNoise(self, distance: float, angle: float):
        std = self.predictStd(distance, angle)
        if distance == 0:
            return 0
        if std < 0:
            raise Exception("predicted standard deviation was negative")
        return np.random.normal(0, std)


def runModelling(distances: np.array, angles: np.array, stds: np.array, upToOrder: int, visualise: bool = False,
                 title: str = "", maskZero: bool = True, units: str = "mm"):
    if maskZero:
        distances = distances.copy()[stds != 0]
        angles = angles.copy()[stds != 0]
        stds = stds.copy()[stds != 0]

    portionToTrainWith = 0.75

    if distances.shape != angles.shape or distances.shape != angles.shape:
        raise Exception("The dimensions of input arrays does not match")

    counts = {"training": int(distances.shape[0] * portionToTrainWith),
              "validation": distances.shape[0] - int(distances.shape[0] * portionToTrainWith)}
    randomAssignmentToTraining = np.random.permutation([True] * counts["training"] + [False] * counts["validation"])
    partition = {"training": {"distances": distances[randomAssignmentToTraining],
                              "angles": angles[randomAssignmentToTraining],
                              "stds": stds[randomAssignmentToTraining]},
                 "validation": {"distances": distances[~randomAssignmentToTraining],
                                "angles": angles[~randomAssignmentToTraining],
                                "stds": stds[~randomAssignmentToTraining]}}

    lastValidationScore, model, poly = None, None, None

    for order in range(1, upToOrder+1):
        polyNew = PolynomialFeatures(degree=order, include_bias=False)

        X = polyNew.fit_transform(np.array([partition["training"]["distances"], partition["training"]["angles"]]).T)
        XValidation = polyNew.fit_transform(np.array([partition["validation"]["distances"],
                                                      partition["validation"]["angles"]]).T)
        modelNew = LinearRegression().fit(X, partition["training"]["stds"])

        validationScore = modelNew.score(XValidation, partition['validation']['stds'])

        if lastValidationScore is not None and (validationScore - lastValidationScore) < 0:
            break
        poly, model = polyNew, modelNew
        lastValidationScore = validationScore

    if visualise:
        plot3D(distances, angles, stds, model, poly, title + f"({model.intercept_:.2e}) + " + " + ".join(map(lambda p: f"({p[0]:.2e}*{p[1]})", zip(model.coef_, poly.get_feature_names_out(('d', 'a'))))), units=units)

    return NoiseModel(model, poly)


def plot3D(distances, angles, stds, model, poly, title, units="mm"):
    precision = 50

    x, y = np.linspace(500, 2500, precision), np.linspace(0, 80, precision)
    xx, yy = np.meshgrid(x, y)
    z = model.predict(poly.fit_transform(np.array([xx.flatten(), yy.flatten()]).T))

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z.reshape(precision, precision), opacity=0.5, colorscale='matter')])
    fig.layout.coloraxis.colorbar.x = -1.5
    fig.update_layout(title=title, autosize=False,
                      width=1200, height=550,
                      template="plotly_white")
    fig.update_layout(scene=dict(
        xaxis_title="Distance [mm]",
        yaxis_title="Angle [degree]",
        zaxis_title=f"Standard deviation [{units}]"))

    fig.update_traces(colorbar=dict(orientation='v', x=1.1, thickness=20,
                                    title=dict(text=f"Standard deviation [{units}]", side="right")))
    fig.add_scatter3d(x=distances, y=angles, z=stds, mode='markers',
                      marker=dict(size=3.5, color=stds,
                                  colorscale='viridis',
                                  colorbar_thickness=20))
    fig.show()


def modelAxial(visualise: bool = False, maxDegree: int = 2):

    readErrorData = True
    startWithAnnotation = False

    if readErrorData:
        with open("Data/errorsAxial.json") as file:
            data = json.load(file)["data"]
    else:
        errorData = mainAnalysisAxialErrorExtraction.runErrorExtraction(runAnnotation()
                                                                        if startWithAnnotation
                                                                        else None)

        data = errorData["data"]
        # data = [{"dist": float,
        #          "angle": int,
        #          "camera": str,
        #          "dist measured": float,
        #          "angle measured": float,
        #          "stds": [{"method": str,
        #                    "std": float }]}]

    chosenModels = {}

    for i, cameraStr in enumerate(sorted(set(map(lambda sceneDict: sceneDict['camera'], data)), reverse=True)):
        for method in sorted(set(map(lambda sceneStds: sceneStds['method'], data[0]['stds']))):
            selectedData = list(filter(lambda x: (x["camera"] == cameraStr), data))
            angles = np.array(list(map(lambda x: x["angle measured"], selectedData)))
            dists = np.array(list(map(lambda x: x["dist measured"], selectedData)))
            stds = np.array(list(map(lambda x: list(filter(lambda y: y["method"] == method, x["stds"]))[0]["std"],
                                     selectedData)))

            chosenModels[(cameraStr, method)] = runModelling(dists, angles, stds, maxDegree,
                                                             visualise=visualise, title=" - ".join((cameraStr, method)))
    return chosenModels


def modelLateral(visualise: bool = False, maxDegree: int = 1):
    readErrorData = True
    startWithAnnotation = False
    startWithEdgeExtraction = True

    if readErrorData:
        with open("Data/errorsLateral.json") as file:
            data = json.load(file)["data"]
    else:
        errorData = (mainAnalysisLateralErrorExtraction
                     .runErrorExtraction(runEdgeExtraction(runAnnotation()
                                                           if startWithAnnotation else None)
                                         if (startWithEdgeExtraction or startWithAnnotation) else None))

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

    chosenModels = {}

    for i, cameraStr in enumerate(sorted(set(map(lambda sceneDict: sceneDict['camera'], data)), reverse=True)):
        selectedData = list(filter(lambda x: (x["camera"] == cameraStr), data))
        angles = np.array(list(map(lambda x: (x["angle"], x["angle"]), selectedData))).flatten()
        dists = np.array(list(map(lambda x: (x["left"]["dist"], x["right"]["dist"]), selectedData))).flatten()
        stds = np.array(list(map(lambda x: (x["left"]["sd"], x["right"]["sd"]), selectedData))).flatten()

        chosenModels[cameraStr] = runModelling(dists, angles, stds, maxDegree, visualise=visualise, title=cameraStr,
                                               units="pixel")
    return chosenModels


if __name__ == '__main__':
    lateralModels = modelLateral(False, maxDegree=2)
    print("Lateral models")
    print("\n".join(map(lambda x: f"{x[0]}: {x[1]}", lateralModels.items())))

    print()
    #
    axialModels = modelAxial(False, maxDegree=2)
    print("Axial models")
    print("\n".join(map(lambda x: f"{x[0]}: {x[1]}", axialModels.items())))
