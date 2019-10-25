//
//  TraderPool.cpp
//  CNT
//
//  Created by Liam Briegel on 25.10.19.
//  Copyright © 2019 Liam Briegel. All rights reserved.
//

// C / C++

// External

// Project
#include "./TraderPool.hpp"


/**************************************************************************************
 * Constructor / Destructor
 * ------------------------
 * Called on new and delete.
 **************************************************************************************/

TraderPool::TraderPool(int i_Input, int i_Output) : s_Pool(i_Input, i_Output, false) {
    Reset();
}

TraderPool::~TraderPool() noexcept {}

/**************************************************************************************
 * Reset
 * -----
 * Reset trader pool.
 **************************************************************************************/

void TraderPool::Reset() {
    s_Mutex.lock();

    if (!s_Pool.species.empty()) {
        us_currentSpecie = 0;
    } else {
        throw std::runtime_error("RESET() : Species empty!");
    }

    if (!s_Pool.species[0].genomes.empty()) {
        us_currentGenome = 0;
        us_SpeciesSize = s_Pool.species.size();

    } else {
        throw std::runtime_error("RESET() : Genomes of species empty!");
    }

    s_Mutex.unlock();
}

/**************************************************************************************
 * Update
 * ------
 * Update trader pool.
 **************************************************************************************/

void TraderPool::NewGeneration() noexcept {
    s_Pool.new_generation();
}

/**************************************************************************************
 * Getters
 * -------
 * TraderPool getters.
 **************************************************************************************/

cneat::genome *TraderPool::GetGenome(size_t us_Specie, size_t us_Genome) noexcept {
    if (s_Pool.species.size() > us_Specie && s_Pool.species[us_Specie].genomes.size() > us_Genome) {
        return &(s_Pool.species[us_Specie].genomes[us_Genome]);
    }

    return NULL;
}

cneat::genome *TraderPool::GetNextGenome() noexcept {
    std::lock_guard<std::mutex> s_Guard(s_Mutex);

    if (us_currentSpecie >= s_Pool.species.size()) {
        return NULL;
    }

    if (us_currentGenome >= s_Pool.species[us_currentSpecie].genomes.size()) {
        if ((++us_currentSpecie) >= s_Pool.species.size()) {
            return NULL;
        }

        us_currentGenome = 0;
    }

    cneat::genome *p_Result = &(s_Pool.species[us_currentSpecie].genomes[us_currentGenome]);
    ++us_currentGenome;

    return p_Result;
}

double TraderPool::GetMaxFitness() noexcept {
    return s_Pool.max_fitness;
}

unsigned int TraderPool::GetGeneration() noexcept {
    return s_Pool.generation();
}

unsigned int TraderPool::GetSpeciesSize() noexcept {
    // @TODO: Wenn size() > int gibt es einen falschen wert zurück!
    return static_cast<unsigned int>(s_Pool.species.size());
}

unsigned int TraderPool::GetPopulationSize() noexcept {
    unsigned int sum = 0;

    for (auto species : s_Pool.species) {
        sum += species.genomes.size();
    }

    return sum;
}

std::string TraderPool::GetSavePath() noexcept {
    return s_Pool.session_path;
}

unsigned int TraderPool::GetOutputSize() noexcept {
    return s_Pool.network_info.output_size;
}
