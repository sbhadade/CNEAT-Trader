//
//  OHLCVManager.hpp
//  CNT
//
//  Created by Liam Briegel on 25.10.19.
//  Copyright © 2019 Liam Briegel. All rights reserved.
//

#ifndef OHLCVManager_hpp
#define OHLCVManager_hpp

// C / C++
#include <cstdio>
#include <vector>
#include <iostream>
#include <string>

// External
#include <convertcsv.hpp>

// Project


namespace OHLCVManager {
    /**************************************************************************************
     * Getters
     * -------
     * OHLCVManager getters.
     **************************************************************************************/

    /**
     *  Get local OHLCV.
     *
     *  \param s_filepath The file path to the dataset.
     *  \param us_window_size The candle window size.
     *
     *  \return The dataset section requested.
     */

    std::vector<std::vector<double>> getlocalOHLCV(std::string s_filepath, size_t us_window_size) {
        std::cout << "Loading Dataset..." << std::endl;

        std::vector<std::vector<double>> data = ConvertCSV::Convert(s_filepath, ',');


        for (size_t i = data.size() - 1; i >= us_window_size; --i)
        {
            for (size_t j = 1; j <= us_window_size; ++j)
            {
                data[i].insert(data[i].end(), data[i - j].begin(), data[i - j].end());
            }
        }

        // Erase the ones that we won't use
        auto it = data.begin();
        auto it_unused = data.begin();
        it_unused += us_window_size;
        data.erase(it, it_unused);

        size_t lenvec = data[0].size();

        for (size_t i = 0; i < data.size(); i++)
        {
            if (lenvec != data[i].size())
            {
                std::cerr << "False fucking len in row" << i << std::endl;
            }
        }

        std::cout << "Finished Loading Dataset" << std::endl;

        return data;
    }

    /**
     *  Get local OHLCV delta.
     *
     *  \param s_filepath The file path to the dataset.
     *  \param us_window_size The candle window size.
     *
     *  \return The dataset section delata requested.
     */

    std::vector<std::vector<double>> getlocalOHLCV_delta(std::string s_filepath, size_t us_window_size)
    {
        std::cerr << "Loading Dataset..." << std::endl;
        std::vector<std::vector<double>> data = ConvertCSV::Convert(s_filepath, ',');

        for (size_t i = data.size() - 1; i >= us_window_size; --i)
        {
            for (size_t j = 1; j <= us_window_size; ++j)
            {
                // ?????
            }
        }

        // Erase the ones that we won't use
        auto it = data.begin();
        auto it_unused = data.begin();
        it_unused += us_window_size;

        data.erase(it, it_unused);

        /**
         * Check the length of the datarows
         */

        size_t lenvec = data[0].size();

        for (size_t i = 0; i < data.size(); i++)
        {
            if (lenvec != data[i].size())
            {
                std::cerr << "False fucking len in row" << i << std::endl;
            }
        }

        std::cerr << "Finished Loading Dataset" << std::endl;

        return data;
    }
}

#endif /* OHLCVManager_hpp */
