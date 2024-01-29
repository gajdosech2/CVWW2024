# pragma once
/*
 * Application for range image capture
 *  - file manipulation functions
 *
 * Developed for Bachelor thesis by Katarina Osvaldova, 2023
 */

 /// includes
#include <string>

using namespace std;

/// initate session timestamp
void initTimestamp();

/// access session timestamp
string getTimestamp();



/// .dpth(f) file manipultion:

/// create a file and write its header
bool openFile(const string& cameraName, const int width, const int height, string flags="", const bool floatFile = false);

/// writing values to file
bool writeUnsignedShort(const unsigned short value);

/// writing values to file
bool writeFloat(const float value);

/// close file
void closeFile();

/// access last filename written
string getLastFilename();


