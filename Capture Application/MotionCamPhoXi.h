#pragma once
/*
 * Application for range image capture
 *  - PhoXi management (MotionCam-3D is accessed through the same library)
 *
 * Developed for Bachelor thesis by Katarina Osvaldova, 2023
 */

 /// includes
#include <vector>
#include "PhoXi.h"


/// warm-up device
void PhoXiWarmUp(const int runForMs, const bool start = true, const bool end = true);

/// capture one shot and save to file
void PhoXiShot();

/// capture multiple shots and save to file
void PhoXiMultipleShots(const int numberOfShots);

/// capture multiple shots periodically and save to file
void PhoXiPeriodic(const bool keepOn, const int waitMs, const int numberOfShots, const int numberOfRounds);



// helper functions for accessing infor from PhoXi Control
void printDeviceInfoList(const std::vector<pho::api::PhoXiDeviceInformation>& DeviceList);
void printDeviceInfo(const pho::api::PhoXiDeviceInformation& DeviceInfo);
