"""
Analysis of lateral and axial noise

Developed for Bachelor's thesis
"""
__author__ = "Katarina Osvaldova"

import typing
import json
import matplotlib
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from annotation import uint16ToUint8
from misc import readDepthMap, saveDictToJSON
from typing import Optional


def findMask(img: np.ndarray, maskClick: typing.Tuple[int, int], fillClick: typing.Tuple[int, int],
             value=np.iinfo('uint8').max) -> np.ndarray:
    img = img.copy()
    tempMask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
    tempMask[maskClick[1] - 3] = 1
    tempMask[maskClick[1] + 3] = 1
    dummyImage = img.copy()
    dummyImage[dummyImage == 0] = 1
    dummyImage = cv.floodFill(dummyImage, tempMask, maskClick, 0, 1, 1)[1]
    dummyImageBoolean = np.full((img.shape[0] + 2, img.shape[1] + 2), False)
    dummyImageBoolean[1:-1, 1:-1] = dummyImage == 0

    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
    mask[dummyImageBoolean] = 1

    img = cv.floodFill(img, mask, fillClick, value, 2, 2)[1]
    img[img != value] = 0
    return img


def borderPixels(img: np.ndarray, fullValue: int, emptyValue: int = 0) -> np.ndarray:
    cornerPositions = [[(X + x, Y + y)
                        for x in range(5) for y in range(5)]
                       for X in (0, img.shape[0] - 5)
                       for Y in (0, img.shape[1] - 5)]

    corners = [img[:5, :5],  img[:5, -5:],
               img[-5:, :5], img[-5:, -5:]]
    cornerFullness = list(map(lambda corner: np.count_nonzero(corner), corners))
    positions = cornerPositions[np.argmin(cornerFullness)]
    for y, x in positions:
        if img[y, x] == 0:
            break
    floodFull = 10
    imgForFlooding = img.copy()
    imgForFlooding[img == emptyValue] = 0
    imgForFlooding[img == fullValue] = floodFull
    # fill in holes
    flooded = cv.floodFill(imgForFlooding, np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8), (x, y), floodFull,
                           1, 1)[1]
    imgC = img.copy()
    imgC[flooded == 0] = fullValue

    filtered = cv.filter2D(imgC, -1, np.ones((3, 3)))
    filtered[img != fullValue] = 0
    filtered[filtered == 9 * fullValue] = 0
    filtered[filtered != 0] = fullValue
    return filtered


def runEdgeExtraction(inputDict: dict = None, inputDictFilename: str = "annotatedScenes.json",
                      saveResult: str = "sceneEdges.json", showScenes: bool = False) -> Optional[dict]:
    value = 1

    if inputDict is None:
        try:
            with open(inputDictFilename) as file:
                inputDict = json.load(file)
        except FileNotFoundError:
            print("No annotation data supplied for edge extraction...")
            return None

    #  inputDict[f"{angle}_{distance[m]}_{uniqueScene}"] = {"dir": str,
    #                                                         "filenames": list[str],
    #                                                         "maskClick": (x, y),
    #                                                         "fillClick": (x, y),
    #                                                         "binCoords": (A.x, A.y, B.x, B.y, C.x, C.y, D.x, D.y)
    #                                                         "innerCoords": (E.x, E.y, F.x, F.y)}
    #         .______C
    #  .______B      |
    #  |  E___|___.  |
    #  |  |   |   |  |
    #  |  .___|___F  |
    #  A______|      |
    #         D______.

    scenesDataDic = {}
    willAskLater = []
    for i, sceneString in enumerate(inputDict):
        print(f"scene {i+1}/{len(inputDict)}: {sceneString}")
        directory = inputDict[sceneString]["dir"]
        filenames = inputDict[sceneString]["filenames"]
        maskClick = inputDict[sceneString]["maskClick"]
        fillClick = inputDict[sceneString]["fillClick"]
        binCoords = inputDict[sceneString]["binCoords"]
        LXB, LYB, LXT, LYT, RXB, RYB, RXT, RYT = binCoords
        initialMask = findMask(uint16ToUint8(readDepthMap(f"{directory}/{filenames[0]}")), maskClick, fillClick,
                               value=value)
        height, width = initialMask.shape

        LXB, LYB, LYT = np.maximum(LXB, 0), np.minimum(LYB, height - 1), np.maximum(LYT, 0)
        RYB, RXT, RYT = np.minimum(RYB, height - 1), np.minimum(RXT, width - 1), np.maximum(RYT, 0)

        initialMaskSize = (initialMask == np.iinfo('uint8').max).sum()
        dataDic = {"left data": [],
                   "right data": [],
                   "left depths": [],
                   "right depths": []}
        scenesDataDic[sceneString] = dataDic
        for filename in filenames:
            depthMap = readDepthMap(f"{directory}/{filenames[0]}")
            mask = findMask(uint16ToUint8(depthMap), maskClick, fillClick, value=value)
            maskDifference = (initialMask != mask).sum()
            if maskDifference > 0.5 * initialMaskSize:
                willAskLater.append((sceneString, directory, filename, maskClick, fillClick, binCoords))
                print(f"Skipping {filename} in {sceneString} for now ------------------------------------------------")
                continue
            for rt, rb, cl, cr, appendData, appendDepth in ((LYT, LYB, LXB, LXT,
                                                             dataDic["left data"], dataDic["left depths"]),
                                                            (RYT, RYB, RXB, RXT,
                                                             dataDic["right data"], dataDic["right depths"])):
                cutout = mask[rt:rb, cl:cr]
                borderImage = borderPixels(cutout, value)

                appendDepth.append(int(depthMap[rt:rb, cl:cr][borderImage == value].mean()))
                appendData.append(np.flip(np.transpose(np.nonzero(borderImage)), 1).tolist())

        if showScenes:
            fig, axs = plt.subplots(1, 3)
            for j, img in enumerate((depthMap, cutout, borderImage)):
                axs[j].imshow(img)
            plt.show()

    if len(willAskLater) != 0:
        print(f"""{len(willAskLater)} files were left out, as while flooding the area, overspill was detected
        for now, they will be ignored
        
        feature of manual flooding was envisioned, but as it was not needed during the analysis of our data, 
        it was not implemented""")

    saveDictToJSON(scenesDataDic, saveResult)
    return scenesDataDic


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    
    runEdgeExtraction()
