#pragma once
/*
 * Application for range image capture
 *  - Kinect v1 management
 *
 * Developed for Bachelor thesis by Katarina Osvaldova, 2023
 */

 /// includes
#include <Windows.h>
#include <NuiApi.h>
#include "misc.h"


/// connect to the device
bool initKinectV1(const int index, const bool nearMode = false, const NUI_IMAGE_RESOLUTION resolution = NUI_IMAGE_RESOLUTION_640x480);

/// initiate the device (turn on IR projector) and open depth stream
bool initDevice(const int index, INuiSensor* &sensor, HANDLE &depthStream, CameraState &state, DWORD flags, NUI_IMAGE_RESOLUTION resolution);

/// turn of projector and disconnect
void closeKinectV1();

/// trigger one shot and save to file
void KinectV1Shot(const bool nearMode, const NUI_IMAGE_RESOLUTION resolution, const int indexOverride = -1);

/// trigger multiple shots and save to file
void KinectV1MultipleShots(const int numberOfShots, const bool nearMode = false, const NUI_IMAGE_RESOLUTION resolution = NUI_IMAGE_RESOLUTION_640x480, const int indexOverride = -1);

/// capture multiple shots periodically and save to file
void KinectV1Periodic(const bool keepOn, int waitMS, const int numberOfShots, const int numberOfRounds, const bool nearMode, const NUI_IMAGE_RESOLUTION resolution, const int indexOverride = -1);
