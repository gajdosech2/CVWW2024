"""
Analysis of lateral and axial noise

Developed for Bachelor's thesis
"""
__author__ = "Katarina Osvaldova"

import os
import re
import json
from typing import Optional
import typing

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import sys
from tkinter import filedialog
from contextlib import ExitStack

matplotlib.use('TkAgg')


def readShort(file, endian, signed=True) -> int:
    return int.from_bytes(file.read(2), endian, signed=signed)


def imageShape(filename) -> typing.Tuple[int, int]:  # (width, height)
    endian = sys.byteorder
    with open(filename, 'rb') as file:
        endianCheck = readShort(file, endian, signed=False)
        if endianCheck == 256:
            endian = "big" if endian == "little" else "little"
        elif endianCheck != 1:
            raise Exception("The file you are reading seems to be corrupted")

        return readShort(file, endian, signed=False), readShort(file, endian, signed=False)


def readDepthMap(filename: str) -> np.ndarray:
    endian = sys.byteorder
    needsSwap = False
    with open(filename, 'rb') as file:
        endianCheck = readShort(file, endian, signed=False)
        if endianCheck == 256:
            endian = "big" if endian == "little" else "little"
            needsSwap = True
        elif endianCheck != 1:
            raise Exception("The file you are reading seems to be corrupted")

        width = readShort(file, endian, signed=False)
        height = readShort(file, endian, signed=False)
        result = np.ndarray(shape=(height, width),
                            dtype=(np.single if filename[-1] == 'f' else np.ushort),
                            buffer=file.read())
        if needsSwap:
            result.byteswap(inplace=True)
        return result


def readDepthMapNoEndian(filename: str) -> np.ndarray:
    """
    function for reading files created before endian check introduced
    """
    endian = 'little'
    with open(filename, 'rb') as file:
        width = readShort(file, endian, signed=False)
        height = readShort(file, endian, signed=False)
        result = np.ndarray(shape=(height, width), dtype=np.ushort, buffer=file.read())

        return result


def manualFileSelection() -> typing.Tuple[str]:
    return filedialog.askopenfilenames(title="Select captures",
                                         initialdir="C:\\Users\\osvka\\Documents\\School\\Predmety\\Bakalárska práca"
                                                    "\\Dataset",
                                         filetypes=[("Depth maps", ".dpth .dpthf")])


def allFilesFromADirectory(filterWord="", title="Select directory", manualOverride=None) -> typing.Tuple[str, typing.Tuple[str]]:
    r = re.compile(".*" + filterWord + ".*\\.dpthx?f?")
    directory = askForDirectory(title) if manualOverride is None else manualOverride
    return tuple() if directory == "" else (directory, tuple(filter(r.fullmatch, os.listdir(directory))))


def askForDirectory(title: str) -> str:
    try:
        return filedialog.askdirectory(mustexist=True,
                                       title=title,
                                       initialdir="C:\\Users\\osvka\\Documents\\School"
                                                  "\\Predmety\\Bakalárska práca\\Dataset")
    except FileNotFoundError:
        return ""


def applyFunctionPixelByPixel(pathPrefix: str, filenames: typing.Tuple[str], function, batchSize=10**8,
                              maskZero=True) -> np.ndarray:
    if len(filenames) == 0:
        raise Exception("No filenames were supplied")

    prefilter = len(filenames)
    filenames = list(filter(lambda filename: filename[-1] == filenames[0][-1], filenames))
    if len(filenames) != prefilter:
        print(prefilter - len(filenames), "files were omitted, as they were of different format (.dpth/.dpthf)")

    endian = sys.byteorder
    needsSwap = False
    chunkSize = batchSize // len(filenames)
    if chunkSize % 4 != 0:
        chunkSize -= chunkSize % 4
    pathPrefix += '/' if pathPrefix != '' else ''
    with ExitStack() as stack:
        files = [stack.enter_context(open(pathPrefix + filename, 'rb')) for filename in filenames]

        endianChecks = [readShort(file, endian, signed=False) for file in files]
        if all(endianCheck == 256 for endianCheck in endianChecks):
            endian = "big" if endian == "little" else "little"
            needsSwap = True
        elif any(endianCheck != 1 for endianCheck in endianChecks):
            raise Exception("At least one of the files you are reading seems to be corrupted")

        width = readShort(files[0], endian, signed=False)
        height = readShort(files[0], endian, signed=False)
        files[0].seek(2, 0)

        prefilter = len(files)
        files = list(filter(lambda file: (readShort(file, endian, signed=False) == width) and
                                         (readShort(file, endian, signed=False) == height),
                            files))
        if len(files) != prefilter:
            print(prefilter-len(files), "files were omitted, as they were of different size")

        resultArray = np.array([])
        ultimateMask = np.array([])
        floatFile = filenames[0][-1] == 'f'
        bytesInNumber = 4 if floatFile else 2

        while True:
            chunks = [file.read(chunkSize) for file in files]
            numbersInChunk = len(chunks[0]) // bytesInNumber
            if not chunks[0]:
                break
            results = [np.ndarray(shape=(numbersInChunk,),
                                  dtype=(np.single if floatFile else np.ushort),
                                  buffer=chunk) for chunk in chunks]
            if needsSwap:
                [result.byteswap(inplace=True) for result in results]

            maskedResults = (ma.masked_equal(results, 0) if maskZero else results)
            if maskZero:
                ultimateMask = np.hstack((ultimateMask, np.all(maskedResults.mask, axis=0)))
            # resultArray = np.hstack((resultArray, function(maskedResults, axis=0)))
            resultArray = np.hstack((resultArray, np.apply_along_axis(function, 0, maskedResults)))

    resultArray = resultArray.reshape((height, width))
    if maskZero:
        ultimateMask = ultimateMask.reshape((height, width))
        resultArray.mask = ultimateMask
    return resultArray


def readPixels(pathPrefix: str, filenames: typing.Tuple[str], pixels: typing.Tuple[typing.Tuple[int, int]],
               temps: bool = False) -> typing.Tuple[np.array, Optional[np.array]]:  # pixels is tuple of (x, y) pairs
    if len(filenames) == 0:
        raise Exception("No filenames were supplied")

    prefilter = len(filenames)
    filenames = list(filter(lambda filename: filename[-1] == filenames[0][-1], filenames))
    if len(filenames) != prefilter:
        print(prefilter - len(filenames), "files were omitted, as they were of different format (.dpth/.dpthf)")

    resultArray = None
    rows = list(map(lambda xy: xy[1], pixels))
    columns = list(map(lambda xy: xy[0], pixels))
    pathPrefix += '/' if pathPrefix != '' else ''
    tempArray = None

    for filename in filenames:
        pixelValues = readDepthMap(pathPrefix + filename)[rows, columns]
        resultArray = pixelValues if resultArray is None else np.vstack((resultArray, pixelValues))
        if temps:
            tempValues = np.array(list(map(lambda x: float(x[1:]), re.findall("T-?\d*\.\d*", filename))))
            tempValues = tempValues[tempValues != -273.15]
            tempArray = tempValues if tempArray is None else np.vstack((tempArray, tempValues))

    return resultArray, tempArray


def applyFilter(img: np.ndarray, f: np.ndarray):
    result = np.zeros(img.shape)
    ri, ci = img.shape
    rf, cf = f.shape
    # result = np.zeros((ri - rf + 1, ci - cf + 1))
    for r in range(ri - rf + 1):
        for c in range(ci - cf + 1):
            result[r+np.floor_divide(rf, 2), c+np.floor_divide(cf, 2)] = (f * img[r:r+rf, c:c+cf]).sum()

    return result


def displayMatrix(matrix: np.ndarray, cmap='cividis', saveTo=None, units="", hideTicks=True, cmin=None, cmax=None,
                  figsize=None, title="", show=True):
    fig = plt.figure(title, figsize=figsize)
    c = plt.colorbar(fig.add_subplot(111).matshow(matrix, cmap=cmap, vmin=cmin, vmax=cmax))
    c.set_label(units)
    if hideTicks:
        plt.xticks([], [])
        plt.yticks([], [])
    # if cmin is not None and cmax is not None:
    #     plt.clim(cmin, cmax)
    if saveTo is not None:
        plt.savefig(saveTo)
        print(f'Plot saved to {saveTo}')
    plt.show(block=show)
    if not show:
        plt.close()


def displayMatrixMasked(matrix: np.ndarray, cmap='cividis', saveTo=None, units="", hideTicks=True, cmin=None, cmax=None,
                        figsize=None, title="", show=True, lessThanEq=0):
    displayMatrix(ma.masked_less_equal(matrix, lessThanEq), cmap, saveTo, units, hideTicks, cmin, cmax, figsize, title,
                  show)


def displayAsPointCloud(img: np.ndarray, saveTo=None, units="", hideTicks=True, cmin=None, cmax=None, elev=0, azim=0,
                        roll=0, show: bool = True):
    y, x = np.indices(img.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim, roll=roll)
    c = plt.colorbar(ax.scatter(x.flatten(), y.flatten(), img.flatten(),
                                c=img.flatten(), cmap='cividis', edgecolors='none', s=8),
                     shrink=0.5, label=units)
    c.set_label(units)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(units, rotation=90)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (0, 0, 0, 0.1)
    ax.yaxis._axinfo["grid"]['color'] = (0, 0, 0, 0.1)
    ax.zaxis._axinfo["grid"]['color'] = (0, 0, 0, 0.1)

    if hideTicks:
        plt.xticks(color='w')
        plt.yticks(color='w')
    if cmin is not None and cmax is not None:
        plt.clim(cmin, cmax)
    if saveTo is not None:
        plt.savefig(saveTo, dpi=90)

    plt.show(block=show)
    if not show:
        plt.close()


def changeInTime(pathPrefix: str, filenames: typing.Tuple[str], lowerBound=0, pixels=None, pixelCount=1, saveTo=None,
                 temps: bool = False, figsize=None, show: bool = True):
    if len(filenames) == 0:
        raise Exception("No filenames were supplied")
    timeTuples = enumerate([(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
                           for m in [re.match("[^_]*_[^_]*_[^_]*_(\\d\\d)-(\\d\\d)-(\\d\\d)\\.(\\d\\d\\d)", filename)
                                     for filename in filenames]])
    order = sorted(timeTuples, key=lambda x: x[1])
    sortedFilenames = tuple(map(lambda x: str(filenames[x[0]]), order))
    h, m, s, ms = order[0][1]
    start = (ms / (60 * 1000) +
             s / 60 +
             m +
             h * 60)
    times = tuple(map(lambda ohmsms: (ohmsms[1][3] / (60 * 1000) +
                                      ohmsms[1][2] / 60 +
                                      ohmsms[1][1] +
                                      ohmsms[1][0] * 60 -
                                      start),
                      order))

    pathPrefix += '/' if pathPrefix != '' else ''
    width, height = imageShape(pathPrefix + sortedFilenames[0])
    if pixels is None:
        pixels = tuple(zip(np.random.randint(0, width, pixelCount), np.random.randint(100, height, pixelCount)))
    pixelValues, tempValues = readPixels(pathPrefix, sortedFilenames, pixels, temps=temps)

    dividers = [0] + list(np.where((np.array(times)[1:] - np.array(times)[:-1]) > 0.2)[0])

    xs, ds, stds, ts = [], [], [], []
    for start, end in zip(dividers[:-1], dividers[1:]):
        end += 1
        xs.append(np.mean(times[start:end]))
        ds.append(np.mean(ma.masked_equal(pixelValues[start:end], 0), axis=0))
        stds.append(np.std(ma.masked_equal(pixelValues[start:end], 0), axis=0))
        if temps:
            ts.append(np.mean(tempValues[start:end], axis=0))

    # plotMatrixMultiline(times, ma.masked_equal(pixelValues, 0), lowerBound, saveTo)
    plotMatrixMultiline(xs, np.array(ds), np.array(ts) if temps else None, lowerBound, saveTo,
                        figsize=figsize, show=show)
    plotMatrixMultiline(xs, np.array(stds), np.array(ts) if temps else None, 0, saveTo + "std",
                        figsize=figsize, show=show,
                        yLabel="Standard deviation of depth [mm]")


def plotMatrixMultiline(x, ys, ts=None, lowerBound=0, saveTo=None, figsize=None, yLabel="Depth [mm]", tempIndex=2,
                        show: bool = True):
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.183, left=0.217, right=0.84)
    ax.set_ylabel(yLabel)
    ax.tick_params(axis='y', labelcolor='#00204c')
    ax.set_xlabel("Time [min]")
    for col in range(ys.shape[1]):
        if not np.all(ys[:, col] < lowerBound):
            ax.plot(x, ys[:, col], '#00204c')

    if ts is not None:
        ax2 = ax.twinx()
        ax2.set_ylabel("Temperature [°C]")
        ax2.tick_params(axis='y', labelcolor='#ffe945')
        ax2.plot(x, ts[:, tempIndex], '#ffe945')

    if saveTo is not None:
        plt.savefig(saveTo)
        print(f"Plot saved to {saveTo}")
    plt.show(block=show)
    if not show:
        plt.close()


def computeFailProbability(pathPrefix: str, filenames: typing.Tuple[str], areaToObserve: typing.Tuple[int, int, int, int]):
    c1, r1, c2, r2 = areaToObserve
    pathPrefix += '/' if pathPrefix != '' else ''
    # probabilityImage = np.zeros_like(imageShape(f'{pathPrefix}{filenames[0]}')[::-1])
    # for filename in filenames: in range(ys.shape[1]):

    counts = applyFunctionPixelByPixel(pathPrefix, filenames, np.count_nonzero, maskZero=False)[r1:r2, c1:c2]
    return counts / (-len(filenames)) + 1


def saveDictToJSON(dictionary: dict, filename: str, indent=None):
    if len(filename) == 0:
        return
    with open(filename, "w") as file:
        file.write(json.dumps(dictionary, indent=indent))
    print(f"Data successfully saved to {filename}")
