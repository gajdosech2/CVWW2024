/*
 * Application for range image capture
 *  - Kinect v1 management
 *
 * Developed for Bachelor thesis by Katarina Osvaldova, 2023
 * 
 * Inspired by code accessible at https://ed.ilogues.com/Tutorials/kinect/kinect2.html
 */
 
/// includes
#include "KinectV1.h"
#include "dataManipulation.h"
#include "misc.h"

#include <Windows.h>

#include <NuiApi.h>
#include <NuiImageCamera.h>
#include <NuiSensor.h>

#include <iostream>


using namespace std;


// Kinect variables
const int nOfDevices = 3;
INuiSensor* sensors[nOfDevices]{ NULL, NULL, NULL };
HANDLE depthStreams[nOfDevices]{ NULL, NULL, NULL };
CameraState  states[nOfDevices]{ notInitialisedYet, notInitialisedYet, notInitialisedYet };

const int startUpWaitTimeMsV1 = 1000;

string modeFlag = "";


/// connect to the device
bool initKinectV1(const int index, const bool nearMode, const NUI_IMAGE_RESOLUTION resolution) {
    cout << "\nStarting Kinect v1 (index " << index << ") initialisation" << endl;
    // Get a working kinect sensor and try to initialise
    int numSensors;
    if (FAILED(NuiGetSensorCount(&numSensors) || numSensors < 1)) {

        cout << "No Kinect v1 could not be initialised (could not find any device)" << endl;
        for (auto& state : states) {
            state = unableToInitialise;
        }
        return false;
    }
    cout << "Number of devices found: " << numSensors << endl;
    DWORD flags = nearMode ? NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE : NULL;
    
    return initDevice(index, sensors[index], depthStreams[index], states[index], flags, resolution);
}

/// initiate the device (turn on IR projector) and open depth stream
bool initDevice(const int index, INuiSensor* &sensor, HANDLE &depthStream, CameraState &state, DWORD flags, NUI_IMAGE_RESOLUTION resolution) {
    // connect to the device
    if (FAILED(NuiCreateSensorByIndex(index, &sensor))) {
        cout << "Kinect v1 (index " << index << ") " << "could not be initialised (could not connect to the device)" << endl;
        state = unableToInitialise;
        return false;
    }

    // initialize sensor
    sensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_DEPTH);

    // open Image stream
    if (FAILED(sensor->NuiImageStreamOpen(NUI_IMAGE_TYPE_DEPTH,
                                          resolution, // Image resolution
                                          flags,      // Image stream flags, e.g. near mode
                                          2,          // Number of frames to buffer, 2 is the standard recommendation
                                          NULL,       // Event handle
                                          &depthStream))) {
        cout << "Kinect v1 (index " << index << ") " << "could not be initialised (could not open stream with given parameters. Most probably, the resolution demanded could not be achieved)" << endl;
        state = unableToInitialise;
        return false;
    }
    cout << "Waiting " << startUpWaitTimeMsV1 << "ms for the device to start up properly" << endl;
    Sleep(startUpWaitTimeMsV1);

    cout << "Kinect v1 (index " << index << ") " << "has been successfully initialised" << endl;
    state = initialised;
    return true;
}

/// turn of projector and close connection
void closeKinectV1() {
    cout << endl;
    int index = 0;
    for (auto &state: states) {
        if (state == initialised) {
            sensors[index]->NuiShutdown();
            cout << "Kinect v1 (index " << index << ") has been closed" << endl;
            state = notInitialisedYet;
        }
        index++;
    }
}

/// helper function for accesing depth map size from resolution settigns
int getWidthFromResolution(const NUI_IMAGE_RESOLUTION resolution) {
    switch (resolution)
    {
    case NUI_IMAGE_RESOLUTION_80x60:
        return 80;
    case NUI_IMAGE_RESOLUTION_320x240:
        return 320;
    case NUI_IMAGE_RESOLUTION_640x480:
        return 640;
    case NUI_IMAGE_RESOLUTION_1280x960:
        return 1280;
    default:
        return -1;
    }
}

/// trigger one shot and save to .dpth file
int getHeightFromResolution(const NUI_IMAGE_RESOLUTION resolution) {
    switch (resolution)
    {
    case NUI_IMAGE_RESOLUTION_80x60:
        return 60;
    case NUI_IMAGE_RESOLUTION_320x240:
        return 240;
    case NUI_IMAGE_RESOLUTION_640x480:
        return 480;
    case NUI_IMAGE_RESOLUTION_1280x960:
        return 960;
    default:
        return -1;
    }
}

/// check if any Kinect v1 device is initialised
bool noInitalisedCamera() {
    for (auto& state : states) {
        if (state == initialised) {
            return false;
        }
    }
    return true;
}

/// access captured depth map and save to file
bool getKinectV1Data() {
    if (noInitalisedCamera()) {
        cout << "\nStarting capture by Kinect v1 unsuccessful (no camera has not been initialised yet)" << endl;
        return false;
    }
    bool returnValue = false;
    for (int index = 0; index < nOfDevices; index++) {
        if (states[index] != initialised) {
            continue;
        }
        cout << "\nStarting capture by Kinect v1 (index " << index << ")" << flush;
        
        NUI_IMAGE_FRAME imageFrame;
        NUI_LOCKED_RECT LockedRect;

        if (FAILED(sensors[index]->NuiImageStreamGetNextFrame(depthStreams[index], 100, &imageFrame))) {
            cout << "  -  unsuccessful (couldn't get a frame from the camera)" << endl;
            continue;
        }

        INuiFrameTexture* texture = imageFrame.pFrameTexture;
        texture->LockRect(0, &LockedRect, NULL, 0);

        if (LockedRect.Pitch != 0) {

            int width = getWidthFromResolution(imageFrame.eResolution);
            int height = getHeightFromResolution(imageFrame.eResolution);

            const unsigned short* curr = (const unsigned short*)LockedRect.pBits;
            const unsigned short* dataEnd = curr + (width * height);

            if (openFile("KinectV1", width, height, modeFlag)) {
                while (curr < dataEnd) {
                    // get depth in millimeters
                    unsigned short depth = NuiDepthPixelToDepth(*curr++);

                    // save it info file
                    writeUnsignedShort(depth);
                }
                closeFile();
                returnValue = true;
                cout << "  -  successful (" << getLastFilename() << ")" << endl;
            } else {
                cout << "  -  unsuccessful (couldn't open file to write to)" << endl;
            }
        } else {
            cout << "  -  unsuccessful (captured frame is empty)" << endl;
        }

        texture->UnlockRect(0);
        sensors[index]->NuiImageStreamReleaseFrame(depthStreams[index], &imageFrame);
    }
    return returnValue;
}

/// trigger one shot and save to file
void KinectV1Shot(const bool nearMode, const NUI_IMAGE_RESOLUTION resolution, const int indexOverride) {
    KinectV1MultipleShots(1, nearMode, resolution, indexOverride);
}

/// trigger multiple shots and save to file
void KinectV1MultipleShots(const int numberOfShots, const bool nearMode, const NUI_IMAGE_RESOLUTION resolution, const int indexOverride) {
    KinectV1Periodic(true, 0, numberOfShots, 1, nearMode, resolution, indexOverride);
}

/// capture multiple shots periodically and save to file
void KinectV1Periodic(const bool keepOn, const int waitMS, const int numberOfShots, const int numberOfRounds, const bool nearMode, const NUI_IMAGE_RESOLUTION resolution, const int indexOverride) {
    
    int index;
    if (indexOverride != -1) {
        index = indexOverride;
    } else {
        getNumberFromConsole(index, "\nChoose Kinect v1 camera\nindex: ", "Choose a number from <0, 2>\nindex: ", 0, 2);
    }

    if (keepOn) {
        if (initKinectV1(index, nearMode, resolution)) {
            for (auto round = 0; round < numberOfRounds - 1; round++) {
                cout << "Captruing round " << (round + 1) << endl;
                for (size_t i = 0; i < numberOfShots; i++) {
                    getKinectV1Data();
                }
                cout << "Sleep commencing" << endl;
                Sleep(waitMS);
            }
            cout << "Captruing round " << numberOfRounds << endl;
            for (size_t i = 0; i < numberOfShots; i++) {
                getKinectV1Data();
            }
        }
        closeKinectV1();
    } else {
        for (auto round = 0; round < numberOfRounds - 1; round++) {
            if (initKinectV1(nearMode, resolution)) {
                cout << "Captruing round " << (round + 1) << endl;
                for (size_t i = 0; i < numberOfShots; i++) {
                    getKinectV1Data();
                }
            }
            closeKinectV1();
            cout << "Sleep commencing" << endl;
            Sleep(max(0, waitMS - startUpWaitTimeMsV1));
        }
        if (initKinectV1(index, nearMode, resolution)) {
            cout << "Captruing round " << numberOfRounds << endl;
            for (size_t i = 0; i < numberOfShots; i++) {
                getKinectV1Data();
            }
        }
        closeKinectV1();
    }
}
