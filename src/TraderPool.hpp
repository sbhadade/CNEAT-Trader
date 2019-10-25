//
//  TraderPool.hpp
//  CNT
//
//  Created by Liam Briegel on 25.10.19.
//  Copyright © 2019 Liam Briegel. All rights reserved.
//

#ifndef TraderPool_hpp
#define TraderPool_hpp

// C++
#include <string>
#include <vector>
#include <mutex>

// External
#include <CANN.hpp>

// Project


class TraderPool {
public:

    /**************************************************************************************
     * Constructor / Destructor
     **************************************************************************************/

    /**
     *  Default constructor.
     *
     *  \param i_Input The amount of inputs.
     *  \param i_Output The amount of outputs.
     */

    TraderPool(int i_Input, int i_Output);

    /**
     *  Default destructor.
     */

    ~TraderPool() noexcept;

    /**************************************************************************************
     * Reset
     **************************************************************************************/

    /**
     *  Reset the pool.
     *  This also causes GetNextGenome() to return the first genome of the first species.
     */

    void Reset();

    /**************************************************************************************
     * Update
     **************************************************************************************/

    void NewGeneration() noexcept;

    /**************************************************************************************
     * Getters
     **************************************************************************************/

    /**
     *  Get a pointer to a specific genome.
     *
     *  \param us_Specie The species to get the genome from.
     *  \param us_Genome The genome of the species.
     *
     *  \return cneat::genome on success, NULL on failure.
     */

    cneat::genome *GetGenome(size_t us_Specie, size_t us_Genome) noexcept;

    /**
     *  Get a pointer to the next genome.
     *
     *  \return A cneat::genome object on success, NULL on failure.
     */

    cneat::genome *GetNextGenome() noexcept;

    /**
     *  Get the pools maximum fitness.
     *
     *  \return The maximum fitness.
     */

    double GetMaxFitness() noexcept;

    /**
     *  Get the current generation.
     *
     *  \return The current generation.
     */

    unsigned int GetGeneration() noexcept;

    /**
     *  Get the current number of species.
     * 
     *  \return The current number of species.
     */

    unsigned int GetSpeciesSize() noexcept;

    /**
     *  Get the total population size.
     *
     *  \return The total population size.
     */

    unsigned int GetPopulationSize() noexcept;

    /**
     *  Get the session save path.
     *
     *  \return The session save path.
     */

    std::string GetSavePath() noexcept;

    /**
     *  Get the network output size.
     *
     *  \return The network output size.
     */

    unsigned int GetOutputSize() noexcept;

private:

    /**************************************************************************************
     * Data
     **************************************************************************************/

    // Our pool
    cneat::pool s_Pool;

    // Genomes
    //std::vector<cneat::specie>::iterator CurrentSpecie;
    //std::vector<cneat::genome>::iterator CurrentGenome;

    size_t us_currentSpecie;
    size_t us_currentGenome;
    size_t us_SpeciesSize;

    // Thread
    std::mutex s_Mutex;

protected:

};

#endif /* TraderPool_hpp */
