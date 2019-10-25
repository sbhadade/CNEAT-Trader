//
// Created by alcon on 07.10.19.
//

#ifndef CNEAT_TRADER_ERRORLOG_HPP
#define CNEAT_TRADER_ERRORLOG_HPP

// Std
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <ctime>

class ErrorLog {
public:
    static void LogError(const std::string s_message, const std::string s_filepath) {

        // Open file
        std::ofstream fs_error;
        fs_error.open(s_filepath, std::ios::app);

        // Check if file is open
        if (!fs_error.is_open()) {

            std::cerr << "Could not open " << s_filepath << std::endl;
        }

        // Get time stuff
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);

        // Output to file
        fs_error << std::put_time(&tm, "%d-%m-%Y %H-%M-%S") << " | " << s_message << std::endl;

        // Close ofstream
        fs_error.close();

    }
};

#endif //CNEAT_TRADER_ERRORLOG_HPP
