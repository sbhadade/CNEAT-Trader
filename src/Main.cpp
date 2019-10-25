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

// Project
#include "./OHLCVManager.hpp"
#include "./EvalFunctions.hpp"

namespace {
    // Settings
    int outputs = 2;
    double capital = 1000;
    int leverage = 10;
    double exposure = 0.1; // 10% -> 0.1
    double fee = 0.00125; // 0.125% -> 0.00125
    double fitness_threshold = 2000;
}


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
 * Evaluate Crypt
 * --------------
 * Evaluate pool genomes with crypto dataset.
 **************************************************************************************/

static void Evaluate_Crypt(TraderPool *p_Pool, ThreadSync *p_ThreadSync, std::vector<std::vector<double>> &v_Data,
                           bool b_MainThread) {
    // Needed Variables
    double starting_money = capital;
    double current_money = starting_money;
    double quantity = 0;
    double open_price;
    double liq_level = 0;
    double pos_value = 0;
    int action;
    double tradefee;
    int numact = 0;
    double close;
    double fitness;

    //long vars
    double buy_units;
    double total_buy;
    double buyvalue;

    //short vars
    double sell_units;
    double total_sell;
    double sellvalue;

    size_t datasize = v_Data.size();
    std::vector<double> out;
    cneat::genome *working_genome;
    cann::feed_forward_network nn;

    do {
        if (!b_MainThread) {
            // Wait for available data
            p_ThreadSync->WaitForSignal();

            // Thread kill
            if (p_ThreadSync->GetKillThreads()) {
                return;
            }
        } else {
            // Signal other threads
            p_ThreadSync->NotifyWaiting();
        }

        // Work on pool
        while ((working_genome = p_Pool->GetNextGenome()) != NULL) {
            // Create ANN
            nn.from_genome(*working_genome);

            // Simulate Trading
            starting_money = capital;
            current_money = starting_money;
            quantity = 0;
            liq_level = 0;
            pos_value = 0;
            numact = 0;

            // Backtesting
            for (unsigned int it = 0; it < datasize; it++) {
                close = v_Data[it][3];

                // get action from ann
                nn.activate(v_Data[it], out);

                // get action: 1 == long ; -1 == short; 0 == nothing
                if (out[0] > 0.5 && out[1] < 0.5) {
                    action = 1;
                } else if (out[0] < 0.5 && out[1] > 0.5) {
                    action = -1;
                } else {
                    action = 0;
                }

                // Check if money is available
                if (current_money <= 0) {
                    current_money = 0;
                    break;
                }

                // check for Liquidations
                if (quantity < 0 && liq_level < close) {
                    // substract liquidation fee
                    tradefee = quantity * close;
                    tradefee = tradefee * fee;
                    current_money -= tradefee;

                    // Kill Position
                    quantity = 0;
                    liq_level = 0;
                    open_price = 0;
                } else if (quantity > 0 && close < liq_level) {
                    // substract liquidation fee
                    tradefee = quantity * close;
                    tradefee = tradefee * fee;
                    current_money -= tradefee;

                    // Kill position
                    quantity = 0;
                    liq_level = 0;
                    open_price = 0;
                }

                // IF ACTION == LONG
                if (action == 1) {
                    numact += 1;

                    if (quantity == 0) {
                        // Open new Long position if there is no current position
                        // Units to buy
                        buy_units = current_money * exposure;
                        buy_units = buy_units / close;
                        // Price of position
                        total_buy = buy_units * close;
                        current_money -= total_buy;
                        pos_value = total_buy;
                        // trade fee
                        tradefee = total_buy * fee;
                        tradefee = tradefee * leverage;
                        current_money -= tradefee;
                        // actualize quantity and open price
                        quantity = buy_units * leverage;
                        open_price = close;
                    } else if (quantity < 0) {
                        // Sell Shortposition if there is one
                        // Get how much you get back from selling the position
                        buyvalue = close - open_price;
                        buyvalue = buyvalue * quantity;
                        buyvalue = pos_value + buyvalue;
                        current_money += buyvalue;

                        // Kill Position
                        quantity = 0;
                        open_price = 0;
                        pos_value = 0;
                    }
                }

                    // If ANN Shorts
                else if (action == -1) {
                    numact += 1;

                    if (quantity == 0) {
                        // Open new Short Position
                        // Units to sell
                        sell_units = current_money * exposure;
                        sell_units = sell_units / close;

                        // Price of Position
                        total_sell = sell_units * close;
                        current_money -= total_sell;
                        pos_value = 0 - sell_units;

                        // tradefee
                        tradefee = total_sell * fee;
                        tradefee = tradefee * leverage;
                        current_money -= tradefee;

                        // actualize quantity and open_price
                        quantity = 0 - sell_units;
                        open_price = close;
                    } else if (quantity > 0) {
                        // Get how much you get back from selling the position
                        sellvalue = close - open_price;
                        sellvalue = sellvalue * quantity;
                        sellvalue = pos_value + sellvalue;
                        current_money += sellvalue;

                        // Kill Position
                        quantity = 0;
                        open_price = 0;
                        pos_value = 0;
                    }
                }
            }

            // Get fitness
            fitness = current_money - starting_money;
            fitness = fitness / starting_money;
            fitness = fitness * 100;

            // If numact == 0 ==> set fitness to -300
            if (numact == 0) {
                fitness = -300;
            }

            // Write fitness
            working_genome->fitness = fitness;
        }
    } while (!b_MainThread);
}

/**************************************************************************************
 * Main
 * ----
 * Program starting point.
 **************************************************************************************/

int main(int argc, const char *argv[]) {
    //std::signal(SIGSEGV, SignalHandle);

    // Define
    std::vector<std::vector<double>> v_Data;
    std::vector<std::thread> v_Thread;

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
    v_Data = OHLCVManager::getlocalOHLCV("../dataset/ForexData/EURUSD/EURUSD240_icmarket_edit.csv", 30);
    unsigned int i_Input = v_Data[0].size(); // = 0



    // Create thread info
    TraderPool s_Pool(i_Input, outputs);
    ThreadSync s_ThreadSync;
    ForexEval s_forexEval;

    // Start all worker threads needed
    ui_AdditionalThreadCount = std::thread::hardware_concurrency() - 1;

    for (unsigned int i = 0; i < ui_AdditionalThreadCount; ++i) {
        v_Thread.push_back(
                std::thread(ForexEval::evaluate, s_forexEval, &s_Pool, &s_ThreadSync, std::ref(v_Data), false));
    }

    // TODO: Condition variable, remove busy loop and counter!
    while (s_ThreadSync.GetWaiting() < ui_AdditionalThreadCount);

    std::cout << "Evaluation in progress... Press CTRL-C to quit." << std::endl << std::endl;

    // Evaluate until the required fitness was reached
    while (true) {

        std::cout << "***** Running Generation " << s_Pool.GetGeneration() << " *****" << std::endl;
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
        std::cout << "Current max fitness: " << s_Pool.GetMaxFitness() << std::endl;
        std::cout << "Total of " << s_Pool.GetSpeciesSize() << " species" << std::endl;
        std::cout << "Total of " << s_Pool.GetPopulationSize() << " Genomes in Population" << std::endl;

        std::cout << "Eval time: "
                  << std::chrono::duration_cast<std::chrono::duration<double>>(s_EvalEnd - s_EvalStart).count()
                  << " seconds" << std::endl;
        std::cout << "Evolution time: " << std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::high_resolution_clock::now() - s_EvolutionStart).count() << " seconds" << std::endl;
        std::cout << "Generation time: " << std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::high_resolution_clock::now() - s_GenerationStart).count() << " seconds" << std::endl
                  << std::endl;
    }

    // Stop and join threads.
    s_ThreadSync.SetKillThreads(true);
    s_ThreadSync.NotifyWaiting();

    for (size_t i = 0; i < v_Thread.size(); ++i) {
        v_Thread[i].join();
    }

    // Export winner to file
    double f64_BestFitness = -999.f;
    cneat::genome *p_CurrentGenome;
    cneat::genome *p_WinnerGenome = NULL;

    s_Pool.Reset();
    while ((p_CurrentGenome = s_Pool.GetNextGenome()) != NULL) {
        if (p_CurrentGenome->fitness > f64_BestFitness) {
            p_WinnerGenome = p_CurrentGenome;
            break;
        }
    }

    if (p_WinnerGenome != NULL) {
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
