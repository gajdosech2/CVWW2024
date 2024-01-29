from misc import readDepthMap, manualFileSelection, displayMatrixMasked


def displayDepthMaps():
    while True:
        filenames = manualFileSelection()
        if len(filenames) == 0:
            return
        displayMatrixMasked(readDepthMap(filenames[0]), units="Depth [mm]", title=filenames[0])


if __name__ == '__main__':
    displayDepthMaps()
