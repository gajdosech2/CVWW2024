/*
 * Application for range image capture
 *  - main file
 * 
 * Developed for Bachelor thesis by Katarina Osvaldova, 2023
 */

/// includes
// general
#include <Windows.h>
#include <Ole2.h>
#include <iostream>
#include <string>

// custom general
#include "dataManipulation.h"
#include "misc.h"

// Kinect v1
#include "KinectV1.h"
#include <NuiApi.h>

// Kinect v2
#include "KinectV2.h"

// Azure Kinect
#include "KinectV3Azure.h"
#include <k4a/k4atypes.h>

// MotionCam-3D
#include "MotionCamPhoXi.h"


using namespace std;



// camera modes and settings
bool v1NearMode = false;
NUI_IMAGE_RESOLUTION v1Resolution = NUI_IMAGE_RESOLUTION_640x480;
k4a_fps_t azureFps = K4A_FRAMES_PER_SECOND_30;
k4a_depth_mode_t azureMode = K4A_DEPTH_MODE_WFOV_2X2BINNED;

// default max count of Kinect v1 devices able to connect at the same time
const int maxKinectV1DeviceIndex = 2;



// individual functionalities of the application:

/**
 * take one range image
 */
void singleShot() {
    cout << "\nCapture single shot by\n\t"
        << "1\tKinect v1\n\t"
        << "2\tKinect v2\n\t"
        << "3\tKinect Azure\n\t"
        << "4\tPhoXi\n\t"
        << "5\tAll cameras" << endl;
    char command;
    while (true) {
        cin >> command;
        switch (command)
        {
        case '1':
            KinectV1Shot(v1NearMode, v1Resolution);
            return;
        case '2':
            KinectV2Shot();
            return;
        case '3':
            KinectAzureShot(azureMode, azureFps);
            return;
        case '4':
            PhoXiShot();
            return;
        case '5':
            for (int index = 0; index <= maxKinectV1DeviceIndex; index++) {
                KinectV1Shot(v1NearMode, v1Resolution, index);
            }
            KinectV2Shot();
            KinectAzureShot(azureMode, azureFps);
            PhoXiShot();
            return;
        default:
            cout << "Invalid option, try again" << endl;
            break;
        }
    }
}

/**
 * take n range images (n specified by the user)
 */
void multipleShots() {
    int nOfShots;
    getNumberFromConsole(nOfShots, "\nCapture\nnumber: ", "Choose a number from <1, 5000>\nnumber: ", 1, 5000);

    cout << "of shots by\n\t"
        << "1\tKinect v1\n\t"
        << "2\tKinect v2\n\t"
        << "3\tKinect Azure\n\t"
        << "4\tPhoXi\n\t"
        << "5\tAll cameras" << endl;
    char command;
    while (true) {
        cin >> command;
        switch (command)
        {
        case '1':
            KinectV1MultipleShots(nOfShots, v1NearMode, v1Resolution);
            return;
        case '2':
            KinectV2MultipleShots(nOfShots);
            return;
        case '3':
            KinectAzureMultipleShots(nOfShots, azureMode, azureFps);
            return;
        case '4':
            PhoXiMultipleShots(nOfShots);
            return;
        case '5':
            for (int index = 0; index <= maxKinectV1DeviceIndex; index++){
                KinectV1MultipleShots(nOfShots, v1NearMode, v1Resolution, index);
            }
            KinectV2MultipleShots(nOfShots);
            KinectAzureMultipleShots(nOfShots, azureMode, azureFps);
            PhoXiMultipleShots(nOfShots);
            return;
        default:
            cout << "Invalid option, try again" << endl;
            break;
        }
    }
}
/**
 * take A range images C times with B ms pauses (A, B, C specified by the user)
 * user also specifies, if the emmiter/projector of the camera stays on during the pauses
 */
void periodicCapture() {
    
    cout << "\nCapture A shots every B milliseconds for C rounds" << endl;

    int nOfShots;  // A 
    getNumberFromConsole(nOfShots, "A: ", "Choose a number from <1, 200>\nA: ", 1, 200);

    int sleepTime;  // B
    getNumberFromConsole(sleepTime, "B: ", "Choose a number from <1, 1000000>\nB: ", 1, 1000000);

    int nOfRounds;  // C
    getNumberFromConsole(nOfRounds, "C: ", "Choose a number from <1, 1000>\nC: ", 1, 1000);


    cout << "Keep on during \"rest\"? (y/n)" << endl;
    char workWhileWaiting;
    while (true) {
        cin >> workWhileWaiting;
        if (workWhileWaiting == 'y' || workWhileWaiting == 'n') {
            break;
        }
        else {
            cout << "choose y or n: ";
        }
    }

    cout << "Capture by\n\t"
        << "1\tKinect v1\n\t"
        << "2\tKinect v2\n\t"
        << "4\tPhoXi\n\t" << endl;
    char command;
    while (true) {
        cin >> command;
        switch (command)
        {
        case '1':
            KinectV1Periodic(workWhileWaiting == 'y', sleepTime, nOfShots, nOfRounds, v1NearMode, v1Resolution);
            return;
        case '2':
            KinectV2Periodic(workWhileWaiting == 'y', sleepTime, nOfShots, nOfRounds);
            return;
        case '3':
            KinectAzurePeriodic(workWhileWaiting == 'y', sleepTime, nOfShots, nOfRounds, azureMode, azureFps);
            return;
        case '4':
            PhoXiPeriodic(workWhileWaiting == 'y', sleepTime, nOfShots, nOfRounds);
            return;
        default:
            cout << "Invalid option, try again" << endl;
            break;
        }
    }
}


/**
 * settings change
 */
void settings() {
    cout << "0\tGo back\n"
        << "1\tKinect v1 turn " << (v1NearMode ? "off" : "on") << " near mode\n"
        << '\n'
        << "2\tSet Kinect v1 resolution to 80x60\n"
        << "3\tSet Kinect v1 resolution to 320x240\n"
        << "4\tSet Kinect v1 resolution to 640x480\n"
        << "5\tSet Kinect v1 resolution to 1280x960\n"
        << '\n'
        << "6\tSet Azure Kinect mode to NFOV_2X2BINNED\n"
        << "7\tSet Azure Kinect mode to NFOV_UNBINNED\n"
        << "8\tSet Azure Kinect mode to WFOV_2X2BINNED\n"
        << "9\tSet Azure Kinect mode to WFOV_UNBINNED\n"
        << '\n'
        << "10\tSet Azure Kinect FPS to 5\n"
        << "11\tSet Azure Kinect FPS to 15\n"
        << "12\tSet Azure Kinect FPS to 30\n"
        << endl;
    int command;
    while (true) {
        cin >> command;
        switch (command)
        {
        case 0:
            return;
        case 1:
            v1NearMode = !v1NearMode;
            return;
        case 2:
            v1Resolution = NUI_IMAGE_RESOLUTION_80x60;
            return;
        case 3:
            v1Resolution = NUI_IMAGE_RESOLUTION_320x240;
            return;
        case 4:
            v1Resolution = NUI_IMAGE_RESOLUTION_640x480;
            return;
        case 5:
            v1Resolution = NUI_IMAGE_RESOLUTION_1280x960;
            break;
        case 6:
            azureMode = K4A_DEPTH_MODE_NFOV_2X2BINNED;
            break;
        case 7:
            azureMode = K4A_DEPTH_MODE_NFOV_UNBINNED;
            break;
        case 8:
            azureMode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
            break;
        case 9:
            azureMode = K4A_DEPTH_MODE_WFOV_UNBINNED;
            break;
        case 10:
            azureFps = K4A_FRAMES_PER_SECOND_5;
            break;
        case 11:
            azureFps = K4A_FRAMES_PER_SECOND_15;
            break;
        case 12:
            azureFps = K4A_FRAMES_PER_SECOND_30;
            break;
        default:
            cout << "Invalid option, try again" << endl;
            break;
        }
    }
}


/**
 * warmin-up of the devices, essentialy turning them on and waiting for a period of time
 * unlike with other functions, warm-up of all cameras is performed parallelly
 */
void warmUp() {
    int runForS;
    getNumberFromConsole(runForS, "\nWarm up (i.e. keep running) for seconds: ", "Choose a number from <1, 600>\nnumber:", 1, 600);
    int runForMs = 1000 * runForS;

    cout << "Warm up device:\n\t"
        << "1\tKinect v1\n\t"
        << "2\tKinect v2\n\t"
        << "3\tKinect Azure\n\t"
        << "4\tPhoXi\n\t"
        << "5\tAll cameras" << endl;
    char command;
    while (true) {
        cin >> command;
        switch (command)
        {
        case '1':
            int index;
            getNumberFromConsole(index, "\nChoose Kinect v1 camera\nindex: ", "Choose a number from <0, 2>\nindex: ", 0, 2);
            initKinectV1(index);
            cout << "Sleep commencing" << endl;
            Sleep(runForMs);
            closeKinectV1();
            return;
        case '2':
            initKinectV2();
            cout << "Sleep commencing" << endl;
            Sleep(runForMs);
            closeKinectV2();
            return;
        case '3':
            initKinectV3();
            cout << "Sleep commencing" << endl;
            Sleep(runForMs);
            closeKinectV3();
            return;
        case '4':
            PhoXiWarmUp(runForMs);
            return;
        case '5':
            // starting all devices up
            initKinectV1(0);
            initKinectV1(1);
            initKinectV1(2);
            initKinectV2();
            initKinectV3();
            PhoXiWarmUp(0, true, false);

            // sleeping
            cout << "Sleep commencing" << endl;
            Sleep(runForMs);

            // closing all devices
            closeKinectV1();
            closeKinectV2();
            closeKinectV3();
            PhoXiWarmUp(0, false, true);
            cout << "Warm up finished" << endl;
            return;
        default:
            cout << "Invalid option, try again" << endl;
            break;
        }
    }
}

/**
 * helper function for boolean option choice
 */
bool startClose() {
    cout << "\t1\tStart\n"
        << "\t2\tClose" << endl;
    char startEnd;
    while (true) {
        cin >> startEnd;
        switch (startEnd) {
        case '1':
            return true;
        case '2':
            return false;
        default:
            cout << "Invalid option, try again" << endl;
            break;
        }
    }
}

/**
 * manual turn on/off of device emmiter/projector
 * allows warm-up without blocking other operations
 */
void manualStartEnd() {
    bool start = startClose();

    cout << "Device:\n\t"
        << "1\tKinect v1\n\t"
        << "2\tKinect v2\n\t"
        << "3\tKinect Azure\n\t"
        << "4\tPhoXi\n\t"
        << "5\tAll cameras" << endl;
    char command;
    while (true) {
        cin >> command;
        switch (command)
        {
        case '1':
            if (!start) {
                closeKinectV1();
                return;
            }
            int index;
            getNumberFromConsole(index, "\nChoose Kinect v1 camera\nindex: ", "Choose a number from <0, 2>\nindex: ", 0, 2);
            initKinectV1(index);
            return;
        case '2':
            (start) ? initKinectV2() : closeKinectV2();
            return;
        case '3':
            (start) ? initKinectV3() : closeKinectV3();
            return;
        case '4':
            (start) ? PhoXiWarmUp(0, true, false) : PhoXiWarmUp(0, false, true);
            return;
        case '5':
            if (start) {
                initKinectV1(0);
                initKinectV1(1);
                initKinectV1(2);
                initKinectV2();
                PhoXiWarmUp(0, true, false);
                return;
            }
            closeKinectV1();
            closeKinectV2();
            PhoXiWarmUp(0, false, true);
            return;
        default:
            cout << "Invalid option, try again" << endl;
            break;
        }
    }
}


/**
 * main function
 * base text interface
 */
int main(int argc, char* argv[]) {
    // start of sesstion, note of timestamp for file designation
    initTimestamp();
    cout << "\nDefault modes are:\n\tKinect v1:\n\t\tnear mode: "
        << (v1NearMode ? "true" : "false")
        << "\n\t\tresolution: 640x480"
        << "\n\tAzure Kinect\n\t\tmode: Binned wide field-of-view \n\t\tFPS: 30" << endl;


    // main loop
    cout << "\nChoose actions by inputing numbers assigned to the options from displayed lists" << endl;

    char command;
    while (true) {
        cout << "\n1\tCapture 1 shot\n"
            << "2\tCapture multiple shots\n"
            << "3\tCapture shots periodically\n"
            << "4\tLet device(s) work for warm-up\n"
            << "5\tManual start/end of warm-up\n"
            << "6\tChange modes/fps\n"
            << "7\tEnd" << endl;
        cin >> command;
        switch (command)
        {
        case '1':
            singleShot();
            break;
        case '2':
            multipleShots();
            break;
        case '3':
            periodicCapture();
            break;
        case '4':
            warmUp();
            break;
        case '5':
            manualStartEnd();
            break;
        case '6':
            settings();
            break;
        case '7':
            cout << "Good bye" << endl;
            return 0;
        default:
            cout << "Invalid option, try again" << endl;
            break;
        }
    }
}