//
//  EvalFunctions.hpp
//  CNT
//
//  Created by Liam Briegel on 25.10.19.
//  Copyright © 2019 Liam Briegel. All rights reserved.
//

#ifndef EvalFunctions_hpp
#define EvalFunctions_hpp

// C / C++

// External
#include <../include/cereal/archives/json.hpp>

// Project
#include "./TraderPool.hpp"
#include "./ThreadSync.hpp"


class ForexEval {
public:

    /**********************************************************************************************
     * Constructor / Destructor
     **********************************************************************************************/

    /**
     *  Default constructor.
     */

    ForexEval() noexcept;

    /**
     *  Default destructor.
     */

    ~ForexEval() noexcept;

    /**********************************************************************************************
     * EvalFunction
     **********************************************************************************************/

    /**
     *  Evaluate.
     *
     *  \param p_ForexEval ForexEval class object.
     *  \param p_Pool Trader pool class object.
     *  \param p_ThreadSync Thread snyc class object.
     *  \param v_Data Trading data reference.
     *  \param b_MainThread Use main thread.
     */

    static void evaluate(ForexEval p_ForexEval, TraderPool *p_Pool, ThreadSync *p_ThreadSync,
                         std::vector<std::vector<double>> &v_Data, bool b_MainThread);

    /**********************************************************************************************
     * Serialize
     **********************************************************************************************/

    /**
     *  Serialize with cereal.
     *
     *  \param s_Archive The archive to use.
     */

    template<class Archive>
    void serialize(Archive s_Archive) {
        s_Archive(CEREAL_NVP(outputs),
                  CEREAL_NVP(capital),
                  CEREAL_NVP(leverage),
                  CEREAL_NVP(exposure),
                  CEREAL_NVP(fee));
    }

private:

    /**********************************************************************************************
     * Contract Buy / Sell contract
     **********************************************************************************************/

    /**
     *  Buy Long contract at current price.
     *
     *  \param quantity Quantity reference.
     *  \param open_price Open price reference.
     *  \param close Close reference.
     */

    inline void buyLong(double &quantity, double &open_price, double &close);

    /**
     *  Sell Long contract and kill position.
     *
     *  \param current_money Current money reference.
     *  \param quantity Quantity reference.
     *  \param open_price Open price reference.
     *  \param close Close reference.
     */

    inline void sellLong(double &current_money, double &quantity, double &open_price, double &close);

    /**
     *  Buy Short contract at current price.
     *
     *  \param quantity Quantity reference.
     *  \param open_price Open price reference.
     *  \param close Close reference.
     */

    inline void buyShort(double &quantity, double &open_price, double &close);

    /**
     *  Sell Short contract and kill position.
     *
     *  \param current_money Current money reference.
     *  \param quantity Quantity reference.
     *  \param open_price Open price reference.
     *  \param close Close reference.
     */

    inline void sellShort(double &current_money, double &quantity, double &open_price, double &close);

    /**********************************************************************************************
     * Liquidation
     **********************************************************************************************/

    /**
     *  Check if position margin is insufficient for sustaining position.
     *  The position is killed if the requirement is met.
     *
     *  \param current_money Current money reference.
     *  \param quantity Quantity reference.
     *  \param open_price Open price reference.
     *  \param close Close reference.  
     */

    inline void checkLiquidation(double &current_money, double &quantity, double &open_price, double &close);

    /**********************************************************************************************
     * Ann / Fitness
     **********************************************************************************************/

    /**
     *  Get action from ANN.
     *
     *  \param FFN FFN reference.
     *  \param dataRow Data row reference.
     *  \param vec_out Output vector reference.
     *
     *  \return The action from the ANN.
     */

    inline int getAction(cann::feed_forward_network &FFN, std::vector<double> &dataRow, std::vector<double> &vec_out);

    /**
     *  Get fitness from ANN.
     *
     *  \param current_money Current money reference.
     *  \param starting_money Starting money reference.
     *  \param num_act The ANN action.
     *
     *  \return The fitness from the ANN.
     */

    inline double getFitness(double &current_money, double &starting_money, int num_act);

    /**********************************************************************************************
     * Data
     **********************************************************************************************/

    // Settings
    int outputs;
    double capital;
    int leverage;
    double exposure;
    double fee;

protected:

};

#endif /* EvalFunctions_hpp */
