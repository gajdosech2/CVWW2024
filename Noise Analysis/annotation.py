"""
Analysis of lateral and axial noise

Developed for Bachelor's thesis
"""
__author__ = "Katarina Osvaldova"

import matplotlib
import os
import re
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from misc import readDepthMap, askForDirectory, saveDictToJSON


class Memory:
    def __init__(self, path: str):
        self.img = uint16ToUint8(readDepthMap(path))

        self.firstDepthMapClick = True
        self.depthMapClickable = True
        self.NOimg = cv.imread('NO.png')
        self.OKimg = cv.imread('OK.png')
        self.AGAINimg = cv.imread('AGAIN.png')

        self.clickX = None
        self.clickY = None
        self.mask = None
        self.axesDepthMap = None
        self.axesOK = None
        self.axesNO = None
        self.axesAGAIN = None
        self.verticalAngleLimit = 15
        self.eImg = None
        self.binCoords = None
        self.innerCoords = None
        self.maskClick = None
        self.fillClick = None
        self.OK = None
        self.AGAIN = None

    def edgeImg(self, boxes: bool = False):
        def horizontalAngle(x1, y1, x2, y2):
            return np.degrees(np.arctan(abs(x2 - x1) / abs(y2 - y1))) if y1 != y2 else 90

        eImgGray = cv.Canny(self.img, 0, 255)
        self.eImg = cv.cvtColor(eImgGray, cv.COLOR_GRAY2BGR)
        if boxes:
            lines = cv.HoughLinesP(eImgGray, 1, np.pi / 180, 30, maxLineGap=25)
            linesArray = np.array([line[0] for line
                                   in filter(lambda line: horizontalAngle(*line[0]) < self.verticalAngleLimit, lines)])
            self.binCoords = tuple(map(int, boxCoords(linesArray)))
            LX1, LY1, LX2, LY2, RX1, RY1, RX2, RY2 = self.binCoords
            cv.rectangle(self.eImg, (LX1, LY1), (LX2, LY2), (255, 0, 0), 2)
            cv.rectangle(self.eImg, (RX1, RY1), (RX2, RY2), (255, 0, 0), 2)

            self.innerCoords = tuple(map(int, getInnerCoords(linesArray)))
            if len(self.innerCoords) != 0:
                x1, y1, x2, y2 = self.innerCoords
                cv.rectangle(self.eImg, (x1, y1), (x2, y2), (0, 255, 0), 2)


def uint16ToUint8(matrix: np.ndarray):
    c = ((np.iinfo('uint8').max - 1 - np.iinfo('uint8').min) /
         (matrix.max() - matrix.min()))
    toReturn = ((matrix - matrix.min()) * c).astype('uint8')
    toReturn[toReturn == np.iinfo('uint8').max] = np.iinfo('uint8').max-1
    return toReturn


def onclick(event, memory: Memory):
    if event.inaxes is not None:
        memory.clickX = int(event.xdata)
        memory.clickY = int(event.ydata)
    if event.inaxes == memory.axesDepthMap and memory.depthMapClickable:
        handleClickDepthMap(memory)
        return
    if event.inaxes == memory.axesOK:
        handleClickAccept(memory, True)
        return
    if event.inaxes == memory.axesNO:
        handleClickAccept(memory, False)
        return
    if event.inaxes == memory.axesAGAIN:
        handleClickAccept(memory, False, True)


def handleClickAccept(memory: Memory, accept: bool, again: bool = False):
    memory.OK = accept
    memory.AGAIN = again
    plt.close()


def handleClickDepthMap(memory):
    if memory.firstDepthMapClick:
        memory.firstDepthMapClick = False
        memory.maskClick = (memory.clickX, memory.clickY)
        tempMask = np.zeros((memory.img.shape[0] + 2, memory.img.shape[1] + 2), np.uint8)
        tempMask[memory.maskClick[1] - 3] = 1
        tempMask[memory.maskClick[1] + 3] = 1
        dummyImage = memory.img.copy()
        dummyImage[dummyImage == 0] = 1
        dummyImage = cv.floodFill(dummyImage, tempMask, memory.maskClick, 0, 1, 1)[1]
        dummyImageBoolean = np.full((memory.img.shape[0] + 2, memory.img.shape[1] + 2), False)
        dummyImageBoolean[1:-1, 1:-1] = dummyImage == 0
        memory.mask = np.zeros((memory.img.shape[0] + 2, memory.img.shape[1] + 2), np.uint8)
        memory.mask[dummyImageBoolean] = 1
        return

    memory.fillClick = (memory.clickX, memory.clickY)
    memory.depthMapClickable = False
    memory.img = cv.floodFill(memory.img, memory.mask, memory.fillClick, np.iinfo('uint8').max, 2, 2)[1]
    memory.img[memory.img != np.iinfo('uint8').max] = 0
    plt.close()


def boxCoords(lines):
    interpolation = 0.75
    heightRatio = 0.7
    leftX = np.min(lines.min(axis=0)[[0, 2]])
    rightX = np.max(lines.max(axis=0)[[0, 2]])
    midpoint = np.mean([leftX, rightX])
    rightQuarter = np.mean([midpoint, rightX])
    leftQuarter = np.mean([midpoint, leftX])
    leftCutoff = interpolation * leftX + (1 - interpolation) * midpoint
    rightCutoff = interpolation * rightX + (1 - interpolation) * midpoint
    meansOfLines = lines[:, [0, 2]].mean(axis=1)
    rightLines = lines[meansOfLines > rightCutoff]
    leftLines = lines[meansOfLines < leftCutoff]
    topRightY = rightLines[:, [1, 3]].min(axis=1).min()
    bottomRightY = rightLines[:, [1, 3]].max(axis=1).max()
    topLeftY = leftLines[:, [1, 3]].min(axis=1).min()
    bottomLeftY = leftLines[:, [1, 3]].max(axis=1).max()
    closeInterpol = (1 - heightRatio) / 2
    farInterpol = (1 + heightRatio) / 2
    return ((3 * leftQuarter - 2 * midpoint),
            closeInterpol * topLeftY + farInterpol * bottomLeftY,
            midpoint,
            closeInterpol * bottomLeftY + farInterpol * topLeftY,

            midpoint,
            closeInterpol * topRightY + farInterpol * bottomRightY,
            (3 * rightQuarter - 2 * midpoint),
            closeInterpol * bottomRightY + farInterpol * topRightY)


def getInnerCoords(lines):
    padding = 10
    cutoff = 20
    midpointX = np.mean([np.min(lines.min(axis=0)[[0, 2]]), np.max(lines.max(axis=0)[[0, 2]])])
    xMeansOfLines = lines[:, [0, 2]].mean(axis=1)
    rightLines = lines[xMeansOfLines > midpointX]
    leftLines = lines[xMeansOfLines < midpointX]
    rightLines = rightLines[np.absolute(rightLines[:, 0] - np.median(rightLines[:, 0])) < cutoff]
    leftLines = leftLines[np.absolute(leftLines[:, 0] - np.median(leftLines[:, 0])) < cutoff]
    if len(leftLines) == 0 or len(rightLines) == 0:
        return tuple()
    x1 = leftLines[:, [0, 2]].max(axis=1).max() + padding
    x2 = rightLines[:, [0, 2]].min(axis=1).min() - padding
    y1 = np.max([leftLines[:, [1, 3]].min(axis=1).min(), rightLines[:, [1, 3]].min(axis=1).min()]) + padding
    y2 = np.min([leftLines[:, [1, 3]].max(axis=1).max(), rightLines[:, [1, 3]].max(axis=1).max()]) - padding

    return x1, y1, x2, y2


def uniqueSceneFilename(flNm: str) -> str:
    match = re.match('(\\d\\d-\\d\\d-\\d\\d\\d\\d_\\d\\d-\\d\\d-\\d\\d_)(T-?\\d+\\.?\\d*)*([^_]+)_.*\.dpthx?(f?)', flNm)
    return match.group(1) + match.group(3) + match.group(4)


def display(memory: Memory, OK: bool = False, boxes: bool = False, windowTitle: str = ""):
    fig, axs = plt.subplots(3, 2, figsize=(15, 7), gridspec_kw={'height_ratios': [10, 1, 1]})
    fig.canvas.manager.set_window_title(windowTitle)
    # main image
    # ax = plt.subplot(221)
    memory.axesDepthMap = axs[0][0]
    axs[0][0].imshow(memory.img, cmap='gray')
    axs[0][0].set_title('Binary image' if boxes else 'Depth Map (quantised to uint8)')
    axs[0][0].set_xticks([]), axs[0][0].set_yticks([])

    # cancel
    # ax = plt.subplot(223)
    axs[1][0].imshow(memory.NOimg)
    axs[1][0].set_xticks([]), axs[1][0].set_yticks([])
    memory.axesNO = axs[1][0]

    # OK
    if OK:
        # ax = plt.subplot(224)
        axs[1][1].imshow(memory.OKimg)
        memory.axesOK = axs[1][1]
    else:
        axs[1][1].axis('off')
    axs[1][1].set_xticks([]), axs[1][1].set_yticks([])

    # AGAIN
    axs[2][0].imshow(memory.AGAINimg)
    memory.axesAGAIN = axs[2][0]
    axs[2][0].set_xticks([]), axs[2][0].set_yticks([])

    # nothing here
    axs[2][1].axis('off')
    axs[2][1].set_xticks([]), axs[2][1].set_yticks([])

    # edge image
    memory.edgeImg(boxes)
    # ax = plt.subplot(222)
    axs[0][1].imshow(memory.eImg)
    axs[0][1].set_title('Edge Image'), axs[0][1].set_xticks([]), axs[0][1].set_yticks([])

    fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, memory))
    # fig.add_gridspec(2, 2, height_ratios=[3, 1])
    mng = plt.get_current_fig_manager()
    mng.window.wm_geometry("+%d+%d" % (0, 0))
    plt.show()


def getInfoAboutScene(path: str, windowTitle: str):
    memory = Memory(path)
    display(memory, boxes=False, windowTitle=windowTitle)

    if memory.OK is not None or memory.depthMapClickable:
        return None, memory.AGAIN

    display(memory, boxes=True, OK=True, windowTitle=windowTitle)

    if memory.OK:
        return memory.maskClick, memory.fillClick, memory.binCoords, memory.innerCoords
    return None, memory.AGAIN


def runAnnotation(saveResult: str = "annotatedScenes.json") -> dict:
    scenesDir = {}

    angles = range(int(input("Min angle [deg]: ")),
                   int(input("Max angle [deg]: ")) + 1,
                   int(input("Angle step [deg]: ")))
    distances = np.arange(int(input("Min distance [cm]: ")) / 100,
                          (int(input("Max distance [cm]: ")) + 1) / 100,
                          int(input("Distance step [cm]: ")) / 100)
    print()
    dirPrefix = askForDirectory("Select location of subdirectories")

    for distance in distances:
        for angle in angles:
            directory = dirPrefix + f"\\{distance:.2f}m\\{angle}deg"
            if not os.path.isdir(directory):
                print(f"Skipping {angle}° at {distance:.2f} m (directory not-existent)", flush=True)
                continue
            filenames = os.listdir(directory)
            uniqueScenes = {}
            for filename in filenames:
                if (filename is None
                        or not os.path.isfile(directory + '/' + filename)
                        or filename.split('.')[-1] not in ("dpth", "dpthf", "dpthx", "dpthxf")):
                    continue
                scene = uniqueSceneFilename(filename)
                if scene not in uniqueScenes:
                    uniqueScenes[scene] = []
                uniqueScenes[scene].append(filename)
            if len(uniqueScenes) == 0:
                print(f"No depthmap files found, skipping {angle}° at {distance:.2f} m", flush=True)
            for uniqueScene in uniqueScenes:
                info = getInfoAboutScene(directory + '/' + uniqueScenes[uniqueScene][0],
                                         f"{angle}° at {distance:.2f} m {uniqueScene}")
                while info[0] is None:
                    if not info[1]:
                        break
                    info = getInfoAboutScene(directory + '/' + uniqueScenes[uniqueScene][0],
                                             f"{angle}° at {distance:.2f} m {uniqueScene} - TRY AGAIN!")
                if info[0] is None:
                    print(f"Ignoring scene {uniqueScene}", flush=True)
                    continue
                maskClick, fillClick, binCoords, innerCoords = info
                scenesDir[f"{angle}_{distance:.2f}_{uniqueScene}"] = {"dir": directory,
                                                                      "filenames": uniqueScenes[uniqueScene],
                                                                      "maskClick": maskClick,
                                                                      "fillClick": fillClick,
                                                                      "binCoords": binCoords,
                                                                      "innerCoords": innerCoords}

    saveDictToJSON(scenesDir, saveResult)
    return scenesDir


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    runAnnotation()
