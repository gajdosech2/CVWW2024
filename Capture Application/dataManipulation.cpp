/*
 * Application for range image capture
 *  - file manipulation functions
 *
 * Developed for Bachelor thesis by Katarina Osvaldova, 2023
 */

 /// includes
#include <sstream>
#include <chrono>
#include <iomanip>
#include <string>
#include <fstream>
#include <iostream>

#include "dataManipulation.h"

using namespace std;

/// set output destination
const string homePath = getenv("USERPROFILE");
const string destinationFolderAbsolutePath = homePath + "/Documents/Depth map output folder/";

/// session timestamp
string timestamp = "";


/// initate session timestamp
void initTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto nowAsTimeT = std::chrono::system_clock::to_time_t(now);
    ostringstream oss;
    oss << put_time(localtime(&nowAsTimeT), "%d-%m-%Y_%H-%M-%S");
    timestamp = oss.str();
    cout << "Recording session started with timestamp " << timestamp << endl;
};

/// access session timestamp
string getTimestamp() {
    return timestamp;
};



/// .dpth(f) file writing:

fstream currentFile;

string lastWrittenTo = "";
string currentFilename = "";

/// create a file and write its header
bool openFile(const string &cameraName, const int width, const int height, string flags, const bool floatFile) {
    if (cameraName.empty() || width == 0 || height == 0) {
        return false;
    }

    // time of creation
    const auto now = std::chrono::system_clock::now();
    const auto nowAsTimeT = std::chrono::system_clock::to_time_t(now);
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    ostringstream timestampCapture;
    timestampCapture << put_time(localtime(&nowAsTimeT), "%H-%M-%S")<< '.' << std::setfill('0') << std::setw(3) << nowMs.count();

    
    currentFilename = timestamp + "_" + cameraName + "_" + timestampCapture.str() + flags + ".dpth" + (floatFile ? "f" : "");

    currentFile.open(destinationFolderAbsolutePath + currentFilename, ios_base::out | ios_base::binary);

    if (!currentFile.is_open()) {
        return false;
    }

    // write headder 
    writeUnsignedShort(1); // endian check
    writeUnsignedShort(width);
    writeUnsignedShort(height);
    return true;
};

/// writing values to file
bool writeUnsignedShort(const unsigned short value) {
    if (!currentFile.is_open()) {
        return false;
    }
    currentFile.write(reinterpret_cast<const char*>(&value), sizeof(unsigned short));
    return true;
};

/// writing values to file
bool writeFloat(const float value) {
    if (!currentFile.is_open()) {
        return false;
    }
    currentFile.write(reinterpret_cast<const char*>(&value), sizeof(float));
    return true;
};

/// close file
void closeFile() {
    if (currentFile.is_open()) {
        currentFile.close();
        lastWrittenTo = currentFilename;
    }
};

/// access last filename written
string getLastFilename() {
    return lastWrittenTo;
}
