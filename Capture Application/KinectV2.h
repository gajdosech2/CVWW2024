#pragma once
/*
 * Application for range image capture
 *  - Kinect v2 management
 *
 * Developed for Bachelor thesis by Katarina Osvaldova, 2023
 */


/// connect to the device
bool initKinectV2();

/// turn of emmiter and disconnect
void closeKinectV2();

/// trigger one shot and save to file
void KinectV2Shot();

/// trigger multiple shots and save to file
void KinectV2MultipleShots(const int numberOfShots);

/// capture multiple shots periodically and save to file
void KinectV2Periodic(const bool keepOn, const int waitMS, const int numberOfShots, const int numberOfRounds);
