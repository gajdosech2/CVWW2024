/*
 * Application for range image capture
 *  - Azure Kinect management
 *
 * Developed for Bachelor thesis by Katarina Osvaldova, 2023
 *
 * Inspired by
 * code accessible at https://github.com/microsoft/Azure-Kinect-Sensor-SDK/tree/develop/examples/viewer/opengl
 * and https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/a227f73b8c78e17db843e7a962a1fabd571c6213/examples/fastpointcloud/main.cpp#L54
 */

// ------------------DISCLAIMER------------------
//   The code has never been tested with a device
//   as is, it might not work properly
// 
//   For example, sleep at the end of initialisation may be needed (just as Kinect v1 and v2 need) 
//   or other modifications that I was not able to experience the need for without the device



 /// includes
#include <k4a/k4a.hpp>  // Azure Kinect SDK

#include <Windows.h>
#include "KinectV3Azure.h"

#include <iostream>

#include "dataManipulation.h"
#include "misc.h"


k4a::device dev;

CameraState cameraV3state = notInitialisedYet;
string modeFlagV3 = "";


bool initKinectV3(const k4a_depth_mode_t mode, k4a_fps_t fps) {
    cout << "\nStarting Azure Kinect initialisation" << endl;
    if (mode == K4A_DEPTH_MODE_WFOV_UNBINNED && fps == K4A_FRAMES_PER_SECOND_30) {
        fps = K4A_FRAMES_PER_SECOND_15;
        cout << "30 fps is not supported in this mode, will continue initialisation with 15 fps" << endl; // according to documentation of the device https://learn.microsoft.com/en-us/azure/kinect-dk/hardware-specification
    }
    try {
        // Check for devices
        if (k4a::device::get_installed_count() == 0) {
            cout << "No Azure Kinect devices detected" << endl;
            cameraV3state = unableToInitialise;
            return false;
        }

        // Start the device
        k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
        config.camera_fps = fps;
        config.depth_mode = mode;
        config.synchronized_images_only = true;

        dev = k4a::device::open(K4A_DEVICE_DEFAULT);
        dev.start_cameras(&config);
                
    } catch (const std::exception& e) {
        cout << "Azure Kinect could not be initialised:\n\t" << e.what() << endl;
        cameraV3state = unableToInitialise;
        return false;
    }

    cout << "Azure Kinect has been successfully initialised" << endl;
    cameraV3state = initialised;

    string flagFps = (fps == K4A_FRAMES_PER_SECOND_30) ?
                        "_30" :
                        ((fps == K4A_FRAMES_PER_SECOND_15) ?
                            "_15" :
                            "_05");
    string flagType = (mode == K4A_DEPTH_MODE_NFOV_2X2BINNED || mode == K4A_DEPTH_MODE_NFOV_UNBINNED) ? 
                          "_N" : 
                          "_W";
    string flagBin = (mode == K4A_DEPTH_MODE_NFOV_2X2BINNED || mode == K4A_DEPTH_MODE_WFOV_2X2BINNED) ?
        "_Binned" :
        "";
    modeFlagV3 = flagFps + flagType + flagBin;
    return true;
}

/// turn of emmiter and disconnect
void closeKinectV3() {
    if (cameraV3state == initialised) {

        dev.close();
        cout << "\nAzure Kinect has been closed" << endl;
        cameraV3state = notInitialisedYet;
    }
}

/// access depth map and save to file
bool getKinectV3Data() {
    cout << "\nStarting capture by Azure Kinect" << flush;  
    
    bool returnValue = false;

    try {
        k4a::capture capture;
        if (dev.get_capture(&capture, std::chrono::milliseconds(300))) {
            const k4a::image depthImage = capture.get_depth_image();
            
            // Depth data is in the form of uint16_t's representing the distance in
            // millimeters of the pixel from the camera.

            const k4a_image_t depth_image = depthImage.handle();


            const int width = k4a_image_get_width_pixels(depth_image);
            const int height = k4a_image_get_height_pixels(depth_image);

            const unsigned short* curr = (uint16_t*)(void*)k4a_image_get_buffer(depth_image);
            const unsigned short* dataEnd = curr + (width * height);

            if (openFile("KinectAzure", width, height, modeFlagV3)) {
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
            cout << "  -  unsuccessful (couldn't get a frame from the camera)" << endl;
            return false;
        }
    } catch (const std::exception& e) {
        cout << "  -  unsuccessful (" << e.what() << ")" << endl;
    }
    return returnValue;
}


/// trigger one shot and save to file
void KinectAzureShot(const k4a_depth_mode_t mode, const k4a_fps_t fps) {
    KinectAzureMultipleShots(1, mode, fps);
}

/// trigger multiple shots and save to file
void KinectAzureMultipleShots(const int numberOfShots, const k4a_depth_mode_t mode, const k4a_fps_t fps) {
    if (initKinectV3(mode, fps)) {
        for (size_t i = 0; i < numberOfShots; i++) {
            getKinectV3Data();
        }
    }
    closeKinectV3();
}

/// capture multiple shots periodically and save to file
void KinectAzurePeriodic(const bool keepOn, const int waitMS, const int numberOfShots, const int numberOfRounds, const k4a_depth_mode_t mode, const k4a_fps_t fps) {
    if (keepOn) {
        if (initKinectV3(mode, fps)) {
            cout << "Waiting 1s for the device to start up properly" << endl;
            Sleep(1000);
            for (auto round = 0; round < numberOfRounds - 1; round++) {
                cout << "Captruing round " << (round + 1) << endl;
                for (size_t i = 0; i < numberOfShots; i++) {
                    getKinectV3Data();
                }
                cout << "Sleep commencing" << endl;
                Sleep(waitMS);
            }
            cout << "Captruing round " << numberOfRounds << endl;
            for (size_t i = 0; i < numberOfShots; i++) {
                getKinectV3Data();
            }
        }
        closeKinectV3();
    }
    else {
        for (auto round = 0; round < numberOfRounds - 1; round++) {
            if (initKinectV3(mode, fps)) {
                cout << "Waiting 1s for the device to start up properly (this 1s has been subtracted from wait time)" << endl;
                Sleep(1000);
                cout << "Captruing round " << (round + 1) << endl;
                for (size_t i = 0; i < numberOfShots; i++) {
                    getKinectV3Data();
                }
            }
            closeKinectV3();
            cout << "Sleep commencing" << endl;
            Sleep(max(0, waitMS - 1000));
        }
        if (initKinectV3(mode, fps)) {
            cout << "Waiting 1s for the device to start up properly (this 1s has been subtracted from wait time)" << endl;
            Sleep(1000);
            cout << "Captruing round " << numberOfRounds << endl;
            for (size_t i = 0; i < numberOfShots; i++) {
                getKinectV3Data();
            }
        }
        closeKinectV3();
    }
}
