/**
 *
 *  Copyright (c) 2019 Jens Broerken <jens.broerken@hs-augsburg.de>
 *
 *  This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable for any damages arising
 *  from the use of this software.
 *
 *  Permission is granted to anyone to use this software for any purpose, including commercial applications,
 *  and to alter it and redistribute it freely, subject to the following restrictions:
 *
 *      1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software.
 *         If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
 *
 *      2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
 *
 *      3. This notice may not be removed or altered from any source distribution.
 *
 */

#ifndef CONVERTCSV_HPP
#define CONVERTCSV_HPP

// C++
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <exception>
#include <cstddef>

// External

// Project


namespace ConvertCSV {
    /**
     *  Convert a given CSV file.
     *
     *  \param s_FilePath The full path to the file.
     *
     *  \return A vector containing a vector of doubles.
     */

    std::vector<std::vector<double>> Convert(std::string s_FilePath, size_t us_Seperator) {
        // Open the file
        std::ifstream f_File(s_FilePath);

        if (f_File.is_open() == false || f_File.bad() == true) {
            throw std::runtime_error("File is not open!");
        }

        // Define out used info
        std::vector<std::vector<double>> v_Result;
        std::vector<double> *p_Line;
        std::string s_Line;
        size_t us_Current;
        size_t us_Next;

        // Loop file until EOF
        while (std::getline(f_File, s_Line)) {
            // Skip lines which are either invalid or commented out with #
            if (s_Line.size() == 0 || s_Line[0] == '#') {
                continue;
            }

            // We now know this line has data, so add the line.
            // We also reference this newly added line, so that we're able to access it faster
            v_Result.push_back(std::vector<double>());
            p_Line = &(v_Result[v_Result.size() - 1]);

            us_Current = 0;

            // Read until all values are found
            do {
                // Get position of the next seperator in line
                if ((us_Next = s_Line.find_first_of(us_Seperator, us_Current)) == std::string::npos) {
                    us_Next = s_Line.size();
                }

                // Try to convert the value and add it to the line
                try {
                    p_Line->push_back(std::stod(s_Line.substr(us_Current, us_Next - us_Current)));
                }
                catch (std::exception &e) {
                    // Something went wrong (characters, etc) - clean up and pass exception (in a nicer format)
                    f_File.close();
                    throw std::string(e.what());
                }

                // Increment position towards the next value
                // +1 is used because us_Next defined the position of the seperator
                us_Current = us_Next + 1;
            } while (us_Next < s_Line.size()); // Exit once we readed the line end
        }

        // All done, clean up after ourselfs
        f_File.close();
        return v_Result;
    }
}

#endif /* CONVERTCSV_HPP */
