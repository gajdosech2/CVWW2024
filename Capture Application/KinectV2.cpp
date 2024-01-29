/*
 * Application for range image capture
 *  - Kinect v2 management
 *
 * Developed for Bachelor thesis by Katarina Osvaldova, 2023
 *
 * Inspired by 
 * code accessible at https://ed.ilogues.com/Tutorials/kinect2/kinect2.html
 * and DepthBasics.cpp from examples attached with Kinect v2 SDK
 */

/// includes
#include "KinectV2.h"
#include "dataManipulation.h"
#include "misc.h"

#include <Windows.h>
#include <Ole2.h>

#include <iostream>

//Kinect 2.0 SDK
#include <Kinect.h>  


using namespace std;


// Kinect variables
IKinectSensor* sensor = NULL;         // Kinect sensor
IDepthFrameReader* reader = NULL;     // Kinect depth data source

CameraState cameraV2state = notInitialisedYet;

const int startUpWaitTimeMsV2 = 3000;


/// connect to the device
bool initKinectV2() { 
    cout << "\nStarting Kinect v2 initialisation" << endl;
    HRESULT result = GetDefaultKinectSensor(&sensor);
    if (FAILED(result)) {
        cout << "Kinect v2 could not be initialised" << endl;
        cameraV2state = unableToInitialise;
        return false;
    }
    

    if (sensor) {
        


        IDepthFrameSource* framesource = NULL;
        if (SUCCEEDED(sensor->Open())) {
            if (SUCCEEDED(sensor->get_DepthFrameSource(&framesource))) {
                result = framesource->OpenReader(&reader);
            }
        }
               
        if (framesource != NULL) {
            framesource->Release();
            framesource = NULL;
        }
    } 

    if (!sensor || FAILED(result)) {
        cout << "Kinect v2 could not be initialised" << endl;
        cameraV2state = unableToInitialise;
        return false;
    }

    cout << "Waiting " << startUpWaitTimeMsV2 << "ms for the device to start up properly" << endl;
    Sleep(startUpWaitTimeMsV2);

    boolean available;
    sensor->get_IsAvailable(&available);
    if (!available) {
        cout << "Kinect v2 could not be initialised - sensor not available (start up unsuccessful, or no camera plugged in)" << endl;
        cameraV2state = unableToInitialise;
        return false;
    }
   
    cout << "Kinect v2 has been successfully initialised" << endl;
    cameraV2state = initialised;
    return true;
}

void closeKinectV2() {
    if (cameraV2state == initialised) {

        sensor->Close();
        cout << "\nKinect v2 has been closed" << endl;
        cameraV2state = notInitialisedYet;
    }
}

bool getKinectV2Data() {
    cout << "\nStarting capture by Kinect v2" << flush;
    if (cameraV2state != initialised) {
        cout << "  -  unsuccessful (camera has not been initialised yet)" << endl;
        return false;
    }

    IDepthFrame* frame = NULL;

    if (FAILED(reader->AcquireLatestFrame(&frame))) {
        cout << "  -  unsuccessful (couldn't get a frame from the camera)" << endl;
        if (frame) frame->Release();
        return false;
    }
    
    bool returnValue = false;
    unsigned int sz;
    unsigned short* buf;
    frame->AccessUnderlyingBuffer(&sz, &buf);
    if (sz > 0) {
        const int width = 512;
        const int height = 424;

        const unsigned short* curr = (const unsigned short*)buf;
        const unsigned short* dataEnd = curr + (width * height);

        if (openFile("KinectV2", width, height)) {
            while (curr < dataEnd) {
                // Get depth in millimeters
                unsigned short depth = (*curr++);

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
    
    if (frame) frame->Release();
    

    return returnValue;
}

/// trigger one shot and save to file
void KinectV2Shot() {
    KinectV2MultipleShots(1);
}

/// trigger multiple shots and save to file
void KinectV2MultipleShots(const int numberOfShots) {
    if (initKinectV2()) {
        for (size_t i = 0; i < numberOfShots; i++) {
            getKinectV2Data();
            Sleep(40);
        }
    }
    closeKinectV2();
}

/// capture multiple shots periodically and save to file
void KinectV2Periodic(const bool keepOn, const int waitMS, const int numberOfShots, const int numberOfRounds) {
    if (keepOn) {
        if (initKinectV2()) {
            for (auto round = 0; round < numberOfRounds - 1; round++) {
                cout << "Captruing round " << (round + 1) << endl;
                for (size_t i = 0; i < numberOfShots; i++) {
                    getKinectV2Data();
                    Sleep(40);  //wait for another frame, the code is otherwise too fast and the Kinect API does not have the option to wait for a new frame and the next call to capture will fail
                }
                cout << "Sleep commencing" << endl;
                Sleep(waitMS);
            }
            cout << "Captruing round " << numberOfRounds << endl;
            for (size_t i = 0; i < numberOfShots; i++) {
                getKinectV2Data();
                Sleep(40);  //wait for another frame, the code is otherwise too fast and the Kinect API does not have the option to wait for a new frame and the next call to capture will fail
            }
        }
        closeKinectV2();
    }
    else {
        for (auto round = 0; round < numberOfRounds - 1; round++) {
            if (initKinectV2()) {
                cout << "Captruing round " << (round + 1) << endl;
                for (size_t i = 0; i < numberOfShots; i++) {
                    getKinectV2Data();
                    Sleep(40);  //wait for another frame, the code is otherwise too fast and the Kinect API does not have the option to wait for a new frame and the next call to capture will fail
                }
            }
            closeKinectV2();
            cout << "Sleep commencing" << endl;
            Sleep(max(0, waitMS - startUpWaitTimeMsV2));
        }
        if (initKinectV2()) {
            cout << "Captruing round " << numberOfRounds << endl;
            for (size_t i = 0; i < numberOfShots; i++) {
                getKinectV2Data();
                Sleep(40);  //wait for another frame, the code is otherwise too fast and the Kinect API does not have the option to wait for a new frame and the next call to capture will fail
            }
        }
        closeKinectV2();
    }
}
