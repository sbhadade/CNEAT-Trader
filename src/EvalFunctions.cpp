//
//  EvalFunctions.cpp
//  CNT
//
//  Created by Liam Briegel on 25.10.19.
//  Copyright © 2019 Liam Briegel. All rights reserved.
//

// C / C++

// External

// Project
#include "./EvalFunctions.hpp"


/**************************************************************************************
 * Constructor / Destructor
 * ------------------------
 * Called on new and delete.
 **************************************************************************************/

ForexEval::ForexEval() noexcept {
    outputs = 2;
    capital = 1000;
    leverage = 10;
    exposure = 0.1; // 10% -> 0.1
    fee = 0.00125; // 0.125% -> 0.00125
}

ForexEval::~ForexEval() noexcept {}

/**************************************************************************************
 * Evaluate Forex
 * --------------
 * Evaluate pool genomes with Forex dataset.
 **************************************************************************************/

void ForexEval::evaluate(ForexEval p_ForexEval, TraderPool *p_Pool, ThreadSync *p_ThreadSync,
                         std::vector<std::vector<double>> &v_Data, bool b_MainThread) {
    // Needed Variables
    double starting_money = p_ForexEval.capital;
    double current_money = starting_money;
    double quantity = 0;
    double open_price = 0.f;
    int action;
    int numact = 0;
    double close;

    size_t datasize = v_Data.size();
    std::vector<double> out(p_Pool->GetOutputSize()); // Create out vector with size of the output nodes
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
            starting_money = p_ForexEval.capital;
            current_money = starting_money;
            quantity = 0;
            numact = 0;

            // Backtesting
            for (size_t it = 0; it < datasize; it++) {
                /*****************************************
                 * Get close price
                 *****************************************/

                close = v_Data[it][3];

                /*****************************************
                 * Get Actions from ANN
                 *****************************************/

                action = p_ForexEval.getAction(nn, v_Data[it], out);

                /*****************************************
                 * Money Check
                 *****************************************/

                // Check if money is available
                if (current_money <= 0) {
                    current_money = 0;
                    break;
                }

                /*****************************************
                 * Liquidation check
                 *****************************************/

                p_ForexEval.checkLiquidation(current_money, quantity, open_price, close);

                /*****************************************
                 * LONG POSITION
                 *
                 * Either close an open short position
                 * or
                 * Open a new Long position
                 *
                 *****************************************/

                if (action == 1) {
                    numact += 1;

                    // Open new Long Position
                    if (quantity == 0) {
                        p_ForexEval.buyLong(quantity, open_price, close);
                    }

                        // Close current short position
                    else if (quantity < 0) {
                        p_ForexEval.sellShort(current_money, quantity, open_price, close);
                    }
                }

                    /*****************************************
                     * SHORT POSITION
                     *
                     * Either close an open long position
                     * or
                     * open a ne short position
                     *
                     *****************************************/

                else if (action == -1) {
                    numact += 1;

                    // Open new short position
                    if (quantity == 0) {
                        p_ForexEval.buyShort(quantity, open_price, close);
                    }

                        // Close existing Long position
                    else if (quantity > 0) {
                        p_ForexEval.sellLong(current_money, quantity, open_price, close);
                    }
                }
            }

            // Write fitness
            working_genome->fitness = p_ForexEval.getFitness(current_money, starting_money, numact);
        }
    } while (!b_MainThread);
}

/**************************************************************************************
 * Contract Buy / Sell contract
 * ----------------------------
 * Handle buying and selling.
 **************************************************************************************/

void ForexEval::buyLong(double &quantity, double &open_price, double &close) {
    // Always have a position of 0.5 Lots TODO: make it dynamic
    quantity = 0.05f;
    open_price = close;
}

void ForexEval::sellLong(double &current_money, double &quantity, double &open_price, double &close) {
    // get how much we won or lost
    current_money += ((close - open_price) * (100000.f * quantity));

    // kill position
    quantity = 0.f;
    open_price = 0.f;
}

void ForexEval::buyShort(double &quantity, double &open_price, double &close) {
    // Always sell 0.5 Lots TODO: Make it dynamic
    quantity = -0.05f;
    open_price = close;
}

void ForexEval::sellShort(double &current_money, double &quantity, double &open_price, double &close) {
    // get how much we won or lost
    current_money += ((open_price - close) * (100000.f * (quantity * -1.f)));

    // Kill Position
    quantity = 0.f;
    open_price = 0.f;
}

/**************************************************************************************
 * Check Liquidation
 * -----------------
 * Check if position margin is insufficient for sustaining position.
 **************************************************************************************/

void ForexEval::checkLiquidation(double &current_money, double &quantity, double &open_price, double &close) {
    if (quantity > 0.f && ((close - open_price) * (100000.f * quantity)) > (current_money / 2.f)) {
        // get how much we won or lost
        current_money += ((close - open_price) * (100000.f * quantity));

        // kill position
        quantity = 0.f;
        open_price = 0.f;
    } else if (quantity < 0.f && ((close - open_price) * (100000.f * quantity)) > (current_money / 2.f)) {
        // get how much we won or lost
        current_money += ((close - open_price) * (100000.f * quantity));

        // kill position
        quantity = 0.f;
        open_price = 0.f;
    }
}

/**************************************************************************************
 * Ann / Fitness
 * -------------
 * ANN interaction.
 **************************************************************************************/

int ForexEval::getAction(cann::feed_forward_network &FFN, std::vector<double> &dataRow, std::vector<double> &vec_out) {
    FFN.activate(dataRow, vec_out);

    // get action: 1 == long ; -1 == short; 0 == nothing
    if (vec_out[0] > 0.5 && vec_out[1] < 0.5) {
        return 1;
    } else if (vec_out[0] < 0.5 && vec_out[1] > 0.5) {
        return -1;
    }

    return 0;
}

double ForexEval::getFitness(double &current_money, double &starting_money, int num_act) {
    // Get fitness
    double fitness = current_money - starting_money;
    fitness = fitness / starting_money;
    fitness = fitness * 100.f;

    // If numact == 0 ==> set fitness to -300
    if (num_act == 0) {
        fitness = -300.f;
    }

    return fitness;
}

