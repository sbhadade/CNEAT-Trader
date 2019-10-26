//
//  Main.cpp
//  CNT
//
//  Created by Liam Briegel on 25.10.19.
//  Copyright © 2019 Liam Briegel. All rights reserved.
//

// C / C++
#include <unistd.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <csignal>

// External
#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>
#include <ncurses.h>

// Project
#include "./OHLCVManager.hpp"
#include "./EvalFunctions.h"



/**************************************************************************************
 * Signal Handler
 * --------------
 * Handle OS signals.
 **************************************************************************************/

extern "C"
{
void SignalHandle(int i_Signal) {
    printf("signal %d\n", i_Signal);
}
}


/**************************************************************************************
 * Main
 * ----
 * Program starting point.
 **************************************************************************************/

int main(int argc, const char *argv[]) {
    //std::signal(SIGSEGV, SignalHandle);
    int outputs;
    double fitness_threshold;

    // Define vars via argv
    if(argc == 1)
    {
        outputs = 2;
        fitness_threshold = 2000;
    }
    else if(argc == 2){
        outputs = std::atoi(argv[1]);
        fitness_threshold = 2000.f;
    }
    else{
        outputs = std::atoi(argv[1]);
        fitness_threshold = std::atof(argv[2]);
    }

    // Define
    std::vector<std::vector<double>> v_Data;
    std::vector<std::thread> v_Thread;

    // Get home dir
    std::string execPath(argv[0]);
    execPath.erase(execPath.find("CNEAT_Trader", 12));
    std::string home_directory = execPath + "../../res";

    // Timestuff
    std::chrono::high_resolution_clock::time_point s_GenerationStart;
    std::chrono::high_resolution_clock::time_point s_EvalStart;
    std::chrono::high_resolution_clock::time_point s_EvalEnd;
    std::chrono::high_resolution_clock::time_point s_EvolutionStart;
    std::chrono::high_resolution_clock::time_point s_TotalStart = std::chrono::high_resolution_clock::now();
    unsigned int ui_AdditionalThreadCount;


    // Load dataset
#ifdef __APPLE__
    chdir("/Users/Jens/Desktop/CNT/res/APPL_ROOT");
#endif
    v_Data = OHLCVManager::getlocalOHLCV(home_directory + "/dataset/ForexData/EURUSD/EURUSD240_icmarket_edit.csv", 30);
    unsigned int i_Input = v_Data[0].size(); // = 0


    // Create thread info
    TraderPool s_Pool(home_directory, i_Input, outputs);
    ThreadSync s_ThreadSync;
    ForexEval s_forexEval;

    // Make archive with config for evaluation

    {
        std::ifstream fs_evalConfig;
        fs_evalConfig.open(home_directory + "/config/EvalSettings.json");
        cereal::JSONInputArchive c_evalConfig(fs_evalConfig);
        s_forexEval.serialization(c_evalConfig);
    }

    // Start all worker threads needed
    ui_AdditionalThreadCount = std::thread::hardware_concurrency() - 1;

    for (unsigned int i = 0; i < ui_AdditionalThreadCount; ++i)
    {
        v_Thread.push_back(
                std::thread(ForexEval::evaluate, s_forexEval, &s_Pool, &s_ThreadSync, std::ref(v_Data), false));
    }

    // TODO: Condition variable, remove busy loop and counter!
    while (s_ThreadSync.GetWaiting() < ui_AdditionalThreadCount);

    std::cout << "Evaluation in progress... Press CTRL-C to quit." << std::endl << std::endl;


    /**
     * Becuase i am a fancy guy i need curses
     */
    initscr();
    WINDOW * win = newwin(20, 80, 0, 0);
    mvwaddstr(win, 18, 1, "Evaluation in progress... Press CTRL-C to quit.");
    std::string cursesUpdate;

    // Evaluate until the required fitness was reached
    while (true) {

        cursesUpdate = "***** Running Generation " + std::to_string(s_Pool.GetGeneration()) + " *****";
        mvwaddstr(win, 1, 1, cursesUpdate.c_str());


        // Reset
        s_GenerationStart = std::chrono::high_resolution_clock::now();

        s_Pool.Reset();

        // Evaluate using the main thread
        s_EvalStart = std::chrono::high_resolution_clock::now();
        ForexEval::evaluate(s_forexEval, &s_Pool, &s_ThreadSync, std::ref(v_Data), true);
        s_EvalEnd = std::chrono::high_resolution_clock::now();

        // Wait for the threads if finished first
        // TODO: Condition variable, remove busy loop and counter!
        while (s_ThreadSync.GetWaiting() < ui_AdditionalThreadCount);

        // Check resulting fitness
        if (s_Pool.GetMaxFitness() >= fitness_threshold) {
            break;
        }


        s_EvolutionStart = std::chrono::high_resolution_clock::now();
        // We need a new generation
        s_Pool.NewGeneration();

        // Print current fitness
        cursesUpdate = "Current max fitness: ";
        mvwaddstr(win, 3, 1, cursesUpdate.c_str());
        cursesUpdate = std::to_string(s_Pool.GetMaxFitness());
        mvwaddstr(win, 3, 35, cursesUpdate.c_str());

        cursesUpdate = "Number of Species: ";
        mvwaddstr(win, 4, 1, cursesUpdate.c_str());
        cursesUpdate = std::to_string(s_Pool.GetSpeciesSize());
        mvwaddstr(win, 4, 35, cursesUpdate.c_str());

        cursesUpdate = "Population size: ";
        mvwaddstr(win, 5, 1, cursesUpdate.c_str());
        cursesUpdate = std::to_string(s_Pool.GetPopulationSize());
        mvwaddstr(win, 5, 35, cursesUpdate.c_str());

        cursesUpdate = "*****Best Genome*****";
        mvwaddstr(win, 7, 1, cursesUpdate.c_str());

        cursesUpdate = "Innovation nbr:";
        mvwaddstr(win, 8, 1, cursesUpdate.c_str());
        cursesUpdate = std::to_string(s_Pool.GetBestKey());
        mvwaddstr(win, 8, 35, cursesUpdate.c_str());

        cursesUpdate = "Fitness: ";
        mvwaddstr(win, 9, 1, cursesUpdate.c_str());
        cursesUpdate = std::to_string(s_Pool.GetBestGenomeFitness());
        mvwaddstr(win, 9, 35, cursesUpdate.c_str());

        cursesUpdate = "Number of nodes:";
        mvwaddstr(win, 10, 1, cursesUpdate.c_str());
        cursesUpdate = std::to_string(s_Pool.GetBestNodeCnt());
        mvwaddstr(win, 10, 35, cursesUpdate.c_str());

        cursesUpdate = "Number of connections:";
        mvwaddstr(win, 11, 1, cursesUpdate.c_str());
        cursesUpdate = std::to_string(s_Pool.GetBestConnCnt());
        mvwaddstr(win, 11, 35, cursesUpdate.c_str());

        cursesUpdate = "Eval-time(sec):";
        mvwaddstr(win, 13, 1, cursesUpdate.c_str());
        cursesUpdate = std::to_string(
                std::chrono::duration_cast<std::chrono::duration<double>>(s_EvalEnd - s_EvalStart).count());
        mvwaddstr(win, 13, 35, cursesUpdate.c_str());

        cursesUpdate = "Evolution-time(sec):";
        mvwaddstr(win, 14, 1, cursesUpdate.c_str());
        cursesUpdate = std::to_string(std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::high_resolution_clock::now() - s_EvolutionStart).count());
        mvwaddstr(win, 14, 35, cursesUpdate.c_str());

        cursesUpdate = "Evaltime(sec):";
        mvwaddstr(win, 15, 1, cursesUpdate.c_str());
        cursesUpdate = std::to_string(std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::high_resolution_clock::now() - s_GenerationStart).count());
        mvwaddstr(win, 15, 35, cursesUpdate.c_str());

        wrefresh(win);
    }

    endwin();

    // Stop and join threads.
    s_ThreadSync.SetKillThreads(true);
    s_ThreadSync.NotifyWaiting();

    for (size_t i = 0; i < v_Thread.size(); ++i)
    {
        v_Thread[i].join();
    }

    // Export winner to file
    double f64_BestFitness = -999.f;
    cneat::genome *p_CurrentGenome;
    cneat::genome *p_WinnerGenome = NULL;

    s_Pool.Reset();
    while ((p_CurrentGenome = s_Pool.GetNextGenome()) != NULL)
    {
        if (p_CurrentGenome->fitness > f64_BestFitness)
        {
            p_WinnerGenome = p_CurrentGenome;
            break;
        }
    }

    if (p_WinnerGenome != NULL)
    {
        cann::feed_forward_network s_NeuralNet; // Change here for recurrent
        s_NeuralNet.from_genome(*p_WinnerGenome);

        // Write winner ANN to file
        {
            std::ofstream ofs_ann;
            ofs_ann.open(s_Pool.GetSavePath() + "/Winner.cann", std::ios::binary);

            cereal::BinaryOutputArchive outArchive(ofs_ann);
            s_NeuralNet.serialization(outArchive);
        }

        // Write Winner Genome to File
        {
            std::ofstream ofs_genome;
            ofs_genome.open(s_Pool.GetSavePath() + "/Winner.genome", std::ios::binary);

            cereal::BinaryOutputArchive outArchive_genome(ofs_genome);
            p_WinnerGenome->serialize(outArchive_genome);

        }


        /**
         * Write genome and ann to a human readable json file
         */
        {
            std::ofstream ofs_ann;
            ofs_ann.open(s_Pool.GetSavePath() + "/Winner_cann.json");

            cereal::JSONOutputArchive outArchive(ofs_ann);
            s_NeuralNet.serialization(outArchive);
        }


        {
            std::ofstream ofs_genome;
            ofs_genome.open(s_Pool.GetSavePath() + "/Winner_genome.json");

            cereal::JSONOutputArchive outArchive_genome(ofs_genome);
            p_WinnerGenome->serialize(outArchive_genome);

        }


    }

    // Result
    std::cout << "***** WinnerGenome *****" << std::endl << "Fitness: " << p_WinnerGenome->fitness << " / "
              << fitness_threshold << std::endl;
    std::cout << "Size: N: " << p_WinnerGenome->node_genes.size() << " C: " << p_WinnerGenome->connection_genes.size()
              << std::endl;
    std::cout << "Fitness reached in " << std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - s_TotalStart).count() << " seconds." << std::endl;
    return EXIT_SUCCESS;
}
