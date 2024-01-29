/*
 * Application for range image capture
 *  - basic helper functions
 *
 * Developed for Bachelor thesis by Katarina Osvaldova, 2023
 */

 /// includes
#include "misc.h"
#include <string>
#include <iostream>


/**
* display prompt, ask for a number and repeat until valid number is supplied
*/
void getNumberFromConsole(int& output, const std::string &firstMessage, const std::string &prompt, const int lowerBound, const int upperBound) {
    std::cout << firstMessage;

    int readFromConsole;
    while (true) {
        
        std::cin >> readFromConsole;
        if (readFromConsole <= (upperBound) && readFromConsole >= (lowerBound)) {
            output = (readFromConsole);
            break;
        }
        else {
            std::cout << prompt;
        }
    }
}



