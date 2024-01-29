#pragma once

/*
 * Application for range image capture
 *  - basic helper functions
 *
 * Developed for Bachelor thesis by Katarina Osvaldova, 2023
 */

 /// includes
#include <string>


/// States of camera initialisation
enum CameraState {
    notInitialisedYet, unableToInitialise, initialised
};

/// display prompt, ask for a number and repeat until valid number is supplied; return the number in supplied output variable
void getNumberFromConsole(int& output, const std::string& firstMessage, const std::string& prompt, const int lowerBound, const int upperBound);
