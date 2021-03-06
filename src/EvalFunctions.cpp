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
#include "./EvalFunctions.h"


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
                        p_ForexEval.buyLong(current_money, quantity, open_price, close);
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
                        p_ForexEval.buyShort(current_money, quantity, open_price, close);
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

void ForexEval::buyLong(double &current_money, double &quantity, double &open_price, double &close) {
    // Calculate quantity
    quantity = ((current_money * this->exposure) * this->leverage) / 100000;
    open_price = close;
}

void ForexEval::sellLong(double &current_money, double &quantity, double &open_price, double &close) {
    // get how much we won or lost
    current_money += ((close - open_price) * (100000.f * quantity));

    // kill position
    quantity = 0.f;
    open_price = 0.f;
}

void ForexEval::buyShort(double &current_money, double &quantity, double &open_price, double &close) {
    // Always sell 0.5 Lots TODO: Make it dynamic
    quantity = (((current_money * this->exposure) * this->leverage) / 100000) * -1;
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




/***********************************************************************************************************************
 *
 *
 * Definitions for class CryptoEval
 *
 *
 ***********************************************************************************************************************/


/**************************************************************************************
 * evaluate
 * --------------
 * Evaluate pool genomes with crypto dataset.
 **************************************************************************************/

void CryptoEval::evaluate(CryptoEval p_cryptoEval, TraderPool *p_Pool, ThreadSync *p_ThreadSync,
                          std::vector<std::vector<double>> &v_Data, bool b_MainThread)
{
    // Needed Variables
    double starting_money = p_cryptoEval.capital;
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
            starting_money = p_cryptoEval.capital;
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
                    tradefee = tradefee * p_cryptoEval.fee;
                    current_money -= tradefee;

                    // Kill Position
                    quantity = 0;
                    liq_level = 0;
                    open_price = 0;
                } else if (quantity > 0 && close < liq_level) {
                    // substract liquidation fee
                    tradefee = quantity * close;
                    tradefee = tradefee * p_cryptoEval.fee;
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
                        buy_units = current_money * p_cryptoEval.exposure;
                        buy_units = buy_units / close;
                        // Price of position
                        total_buy = buy_units * close;
                        current_money -= total_buy;
                        pos_value = total_buy;
                        // trade fee
                        tradefee = total_buy * p_cryptoEval.fee;
                        tradefee = tradefee * p_cryptoEval.leverage;
                        current_money -= tradefee;
                        // actualize quantity and open price
                        quantity = buy_units * p_cryptoEval.leverage;
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
                        sell_units = current_money * p_cryptoEval.exposure;
                        sell_units = sell_units / close;

                        // Price of Position
                        total_sell = sell_units * close;
                        current_money -= total_sell;
                        pos_value = 0 - sell_units;

                        // tradefee
                        tradefee = total_sell * p_cryptoEval.fee;
                        tradefee = tradefee * p_cryptoEval.leverage;
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
