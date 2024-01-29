/*
 * Application for range image capture
 *  - PhoXi management (MotionCam-3D is accessed through the same library)
 *
 * Developed for Bachelor thesis by Katarina Osvaldova, 2023
 *
 * Inspired by MinimalOpenCVExample.cpp from examples attached with Photoneo's PhoXi API
 */

/// includes
#include <iostream>
#include <string>
#include <vector>
#include <climits>
#include <cmath>
#include <Windows.h>

#include "PhoXi.h"  //PhoXi API

#include "MotionCamPhoXi.h"
#include "misc.h"
#include "dataManipulation.h"


using namespace std;

pho::api::PPhoXi PhoXiDevice;


CameraState cameraPhoXiState = notInitialisedYet;

// connect to the device (needs to be connected in PhoXi Control!)
bool initPhoXi() {
    cout << "\nStarting PhoXi initialisation" << endl;
    pho::api::PhoXiFactory Factory;

    // Check if the PhoXi Control Software is running
    if (!Factory.isPhoXiControlRunning()) {
        cout << "PhoXi Control Software is not running" << endl;
        cameraPhoXiState = unableToInitialise;
        return false;
    }

    // Get List of available devices on the network
    std::vector<pho::api::PhoXiDeviceInformation> DeviceList =
        Factory.GetDeviceList();
    if (DeviceList.empty()) {
        cout << "PhoXi Factory has found 0 devices" << endl;
        cameraPhoXiState = unableToInitialise;
        return false;
    }
    // information about available devices, with manual connection (see following else branch) not relevant
    /*cout << "Available devices:" << endl;
    printDeviceInfoList(DeviceList);*/

    // Try to connect device opened in PhoXi Control, if any
    PhoXiDevice = Factory.CreateAndConnectFirstAttached();
    if (PhoXiDevice) {
        cout << "You have already PhoXi device opened in PhoXi Control, "
            "connected to device: "
            << (std::string)PhoXiDevice->HardwareIdentification
            << std::endl;
    } else {
        // to switch from manual device selection in PhoXi Control, uncomment the block and comment the rest of this else branch

        /*cout
            << "You have no PhoXi device opened in PhoXi Control, ";
        for (size_t i = 0; i < DeviceList.size(); i++) {
            cout << "will try to connect to ..."
                << DeviceList.at(i).HWIdentification << std::endl;
            // wait 5 second for scanner became ready
            PhoXiDevice = Factory.CreateAndConnect(
                DeviceList.at(i).HWIdentification, 5000);
            if (PhoXiDevice) {
                cout << "successfully connected" << std::endl;
                break;
            }
            if (i == DeviceList.size() - 1) {
                std::cout << "Cannot connect to any device" << std::endl;
            }
        }*/

        cout << "You have no PhoXi device opened in PhoXi Control, please open a device and try again then" << endl;

    }

    // Check if device was created
    if (!PhoXiDevice) {
        cout << "Your device was not created" << endl;
        cameraPhoXiState = unableToInitialise;
        return false;
    }

    // Check if device is connected
    if (!PhoXiDevice->isConnected()) {
        cout << "Your device is not connected" << endl;
        cameraPhoXiState = unableToInitialise;
        return false;
    }

    cout << "PhoXi has been successfully initialised" << endl;
    cameraPhoXiState = initialised;
    return true;
}


void closePhoXi() {
    // Disconnect PhoXi device
    // recommended not to call at all, but rather disconnect the device in PhoXi Control manualy
    //   after programmatically disconnecting, further use will need a new connection (manual in PhoXi Control (or see initPhoXi())) and the device may take a few seconds to become available again
    if (cameraPhoXiState == initialised && PhoXiDevice->isConnected()) {
        PhoXiDevice->Disconnect(true, false);
    }
    cout << "\nPhoXi has been closed" << endl;
    cameraPhoXiState = notInitialisedYet;
}

/// warm-up device
void PhoXiWarmUp(const int runForMs, const bool start, const bool end) {
    if (start) {
        initPhoXi();
        if (PhoXiDevice->isAcquiring()) {
            // Stop acquisition to change trigger mode
            PhoXiDevice->StopAcquisition();
        }

        PhoXiDevice->TriggerMode = pho::api::PhoXiTriggerMode::Freerun;
        PhoXiDevice->ClearBuffer();
        PhoXiDevice->StartAcquisition();
        if (!PhoXiDevice->isAcquiring()) {
            cout << "Couldn't start the PhoXi device, warm up usuccessful" << endl;
            return;
        }
    }
    if (start && end) {
        cout << "Sleep commencing" << endl;
        Sleep(runForMs);
    }
    if (end) {
        PhoXiDevice->StopAcquisition();
        //closePhoXi();  // see deffinition before uncommenting
    }
    
    if (start && end) {
        cout << "Warm up finished" << endl;
    }
}

/// float to ushort rounding, no longer in use, as .dpthf format was created
static short floatToUnsignedShort(float x) {
    if (x < 0) {
        return 0;
    }
    if (x > USHRT_MAX) {
        return USHRT_MAX;
    }
    return (unsigned short) round(x);
}

/// read depth map and save to file
bool saveFrameToFile(const pho::api::PFrame &Frame) {
    if (!Frame->DepthMap.Empty()) {

        const auto width = Frame->DepthMap.Size.Width;
        const auto height = Frame->DepthMap.Size.Height;

        const pho::api::Depth_32f* curr = Frame->DepthMap.GetDataPtr();
        const pho::api::Depth_32f* dataEnd = curr + (width * height);

        string tempString = "";
        for (double& temp : Frame->Info.Temperatures) {
            tempString += "T" + to_string(temp);
        }

        if (openFile(tempString + "PhoXi", width, height, "", true)) {
            while (curr < dataEnd) {
                /*Depth map [consists] of 32 bit floats coding orthogonal distances
                from the internal camera in mm. [definition from documentation]*/

                // history of code comment:
                // as all other cameras give natural numbers, I chose to round the numbers
                // (pho::api::float32_t is a float)
                // !!!! this has been later seen as a wrong decision and source of quantization noise and the decision has been changed to use float despite the need for twice the storage size
                float depth = *curr++;

                // save it into file
                writeFloat(depth);
            }
            closeFile();
            cout << "  -  successful (" << getLastFilename() << ")" << endl;
            return true;
        } else {
            cout << "  -  unsuccessful (couldn't open file to write to)" << endl;
        }
    } else {
        cout << "  -  unsuccessful (captured frame is empty)" << endl;
    }
    return false;
}

/// capture a frame and save to file (!!! does not manage start of projector)
bool PhoXiCapture(bool printStart) {
    if (printStart) {
        cout << "\nStarting capture by PhoXi" << flush;
    }   

    pho::api::PFrame Frame = PhoXiDevice->GetFrame(pho::api::PhoXiTimeout::LastStored);

    if (!Frame) {
        cout << "  -  unsuccessful (failed to retrieve the frame)" << endl;
        return false;
    } 
    return saveFrameToFile(Frame);
}

/// start projector and take n shots
bool getPhoXiData(const int numberOfShots) {
    cout << "\nStarting capture by PhoXi" << flush;
    if (PhoXiDevice->isAcquiring()) {
        // Stop acquisition to change trigger mode
        PhoXiDevice->StopAcquisition();
    }

    PhoXiDevice->TriggerMode = pho::api::PhoXiTriggerMode::Freerun;
    PhoXiDevice->ClearBuffer();
    PhoXiDevice->StartAcquisition();
    if (!PhoXiDevice->isAcquiring()) {
        cout << "  -  unsuccessful (couldn't start acquisition)" << endl;
        return false;
    }

    for (size_t i = 0; i < numberOfShots; i++) {
        PhoXiCapture(i != 0);
    }

    PhoXiDevice->StopAcquisition();
    return true;
}

/// capture one shot and save to file
void PhoXiShot() {
    PhoXiMultipleShots(1);
}

/// capture multiple shots and save to file
void PhoXiMultipleShots(const int numberOfShots) {
    if (initPhoXi()) {
        getPhoXiData(numberOfShots);
    }
    //closePhoXi();  // see deffinition before uncommenting
}

/// capture multiple shots periodically and save to file
void PhoXiPeriodic(bool keepOn, int waitMs, int numberOfShots, int numberOfRounds) {
    if (keepOn) {
        if (initPhoXi()) {
            // just as getPhoXiData, but added sleep and acquisition end at the end
            if (PhoXiDevice->isAcquiring()) {
                // Stop acquisition to change trigger mode
                PhoXiDevice->StopAcquisition();
            }

            PhoXiDevice->TriggerMode = pho::api::PhoXiTriggerMode::Freerun;
            PhoXiDevice->ClearBuffer();
            PhoXiDevice->StartAcquisition();
            if (!PhoXiDevice->isAcquiring()) {
                cout << "\nStarting capture by PhoXi  -  unsuccessful (couldn't start acquisition)" << endl;
                return;
            }

            for (auto round = 0; round < numberOfRounds - 1; round++) {
                cout << "Captruing round " << (round + 1) << endl;
            
                for (size_t i = 0; i < numberOfShots; i++) {
                    PhoXiCapture(true);
                }
                cout << "Sleep commencing" << endl;
                Sleep(waitMs);
            }
            cout << "Captruing round " << numberOfRounds << endl;
            for (size_t i = 0; i < numberOfShots; i++) {
                PhoXiCapture(true);
            }
        }
        PhoXiDevice->StopAcquisition();
        //closePhoXi();  // see deffinition before uncommenting
    } else {
        if (initPhoXi()) {
            for (auto round = 0; round < numberOfRounds - 1; round++) {
                cout << "Captruing round " << (round + 1) << endl;
                getPhoXiData(numberOfShots);
                cout << "Sleep commencing" << endl;
                Sleep(waitMs);
            }
            cout << "Captruing round " << numberOfRounds << endl;
            getPhoXiData(numberOfShots);
            //closePhoXi();  // see deffinition before uncommenting
        }
    }
}

// helper function for accessing infor from PhoXi Control
void printDeviceInfoList(
        const std::vector<pho::api::PhoXiDeviceInformation> &DeviceList) {
    for (std::size_t i = 0; i < DeviceList.size(); ++i) {
        std::cout << "Device: " << i << std::endl;
        printDeviceInfo(DeviceList[i]);
    }
}

// helper functions for accessing infor from PhoXi Control
void printDeviceInfo(const pho::api::PhoXiDeviceInformation& DeviceInfo) {
    std::cout << "  Name:                    " << DeviceInfo.Name << std::endl;
    std::cout << "  Hardware Identification: " << DeviceInfo.HWIdentification << std::endl;
    std::cout << "  Type:                    " << std::string(DeviceInfo.Type) << std::endl;
    std::cout << "  Firmware version:        " << DeviceInfo.FirmwareVersion << std::endl;
    std::cout << "  Variant:                 " << DeviceInfo.Variant << std::endl;
    std::cout << "  IsFileCamera:            " << (DeviceInfo.IsFileCamera ? "Yes" : "No") << std::endl;
    std::cout << "  Feature-Alpha:           " << (DeviceInfo.CheckFeature("Alpha") ? "Yes" : "No") << std::endl;
    std::cout << "  Feature-Color:           " << (DeviceInfo.CheckFeature("Color") ? "Yes" : "No") << std::endl;
    std::cout << "  Status:                  "
        << (DeviceInfo.Status.Attached
            ? "Attached to PhoXi Control. "
            : "Not Attached to PhoXi Control. ")
        << (DeviceInfo.Status.Ready ? "Ready to connect" : "Occupied")
        << std::endl
        << std::endl;
}