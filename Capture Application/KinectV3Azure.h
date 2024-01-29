#pragma once
/*
 * Application for range image capture
 *  - Azure Kinect management
 *
 * Developed for Bachelor thesis by Katarina Osvaldova, 2023
 */

///includes
#include <k4a/k4atypes.h>
#include <Windows.h>


/// connect to the device and start emmiter
bool initKinectV3(const k4a_depth_mode_t mode = K4A_DEPTH_MODE_WFOV_2X2BINNED,
				  k4a_fps_t fps = K4A_FRAMES_PER_SECOND_30);


/// turn of emmiter and disconnect
void closeKinectV3();


/// trigger one shot and save to file
void KinectAzureShot(const k4a_depth_mode_t mode = K4A_DEPTH_MODE_WFOV_2X2BINNED,
	                 const k4a_fps_t fps = K4A_FRAMES_PER_SECOND_30);

/// trigger multiple shots and save to file
void KinectAzureMultipleShots(const int numberOfShots,
							 const k4a_depth_mode_t mode = K4A_DEPTH_MODE_WFOV_2X2BINNED, 
							 const k4a_fps_t fps = K4A_FRAMES_PER_SECOND_30);

/// capture multiple shots periodically and save to file
void KinectAzurePeriodic(const bool keepOn,
				 		 const int waitMS, 
				 		 const int numberOfShots, 
				 		 const int numberOfRounds, 
				 		 const k4a_depth_mode_t mode = K4A_DEPTH_MODE_WFOV_2X2BINNED, 
				         const k4a_fps_t fps = K4A_FRAMES_PER_SECOND_30);
