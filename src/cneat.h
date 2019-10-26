//
// Created by alcon on 25.10.19.
//

#ifndef CNEAT_TRADER_CNEAT_H
#define CNEAT_TRADER_CNEAT_H


// C / C++
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <cmath>
#include <random>
#include <map>
#include <algorithm>
#include <list>
#include <string>
#include <climits>
#include <chrono>
#include <sys/stat.h>

// External

// Project
#include "ErrorLog.hpp"
#include "cereal/archives/binary.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/types/string.hpp"
#include "cereal/archives/json.hpp"


namespace cneat {


    template<class InputIt>
    InputIt find_key(InputIt first, InputIt last, unsigned int key) {
        for (; first != last; ++first) {
            if (first->key == key) {
                return first;
            }
        }
        return last;
    }


/**********************************************************************
 * Mutation rates
 **********************************************************************/
    typedef struct {

        // Crossover
        double crossover_chance = 0.7;

        // Connection
        double connection_add_chance = 0.7;
        double connection_delete_chance = 0.5;

        double disable_mutation_chance = 0.2;
        double enable_mutation_chance = 0.2;

        double weight_mutation_Rate = 0.5;
        double weight_mutate_chance = 0.5;

        // Node
        double node_delete_chance = 0.5;
        double node_add_chance = 0.7;

        double aggregation_mutation_chance = 0.1;
        int aggregation_choices = 3;

        double activation_mutation_chance = 0.1;
        int activation_choices = 3;

        double bias_mutation_chance = 0.5;
        double bias_mutation_rate = 0.5;

        double response_mutation_chance = 0.5;
        double response_mutation_rate = 0.5;


        // Serialization
        template<class Archive>
        void serialize(Archive &archive) {

            archive(CEREAL_NVP(crossover_chance),
                    CEREAL_NVP(connection_add_chance),
                    CEREAL_NVP(connection_delete_chance),
                    CEREAL_NVP(disable_mutation_chance),
                    CEREAL_NVP(enable_mutation_chance),
                    CEREAL_NVP(weight_mutation_Rate),
                    CEREAL_NVP(weight_mutate_chance),
                    CEREAL_NVP(node_delete_chance),
                    CEREAL_NVP(node_add_chance),
                    CEREAL_NVP(aggregation_mutation_chance),
                    CEREAL_NVP(aggregation_choices),
                    CEREAL_NVP(activation_mutation_chance),
                    CEREAL_NVP(activation_choices),
                    CEREAL_NVP(bias_mutation_chance),
                    CEREAL_NVP(bias_mutation_rate),
                    CEREAL_NVP(response_mutation_chance),
                    CEREAL_NVP(response_mutation_rate));
        }

    } mutation_rate_container;


/**********************************************************************
 *  Species parameter
 **********************************************************************/
    typedef struct {
        // Size of population
        unsigned int population = 100; //240

        // impact of factors to the genetic distance
        double delta_disjoint = 1.0f;
        double delta_weights = 0.7f;
        double delta_threshold = 3.f;

        // How long a species can survive without being erased
        unsigned int stale_species = 20;

        // How many members of a species will survive before reproduction
        double survival_threshhold = 0.2;

        // Minimum pop of a species after cutting
        int min_survivors = 2;

        // Serialization
        template<class Archive>
        void serialize(Archive &archive) {

            archive(CEREAL_NVP(population),
                    CEREAL_NVP(delta_disjoint),
                    CEREAL_NVP(delta_weights),
                    CEREAL_NVP(delta_threshold),
                    CEREAL_NVP(stale_species),
                    CEREAL_NVP(survival_threshhold),
                    CEREAL_NVP(min_survivors));
        }

    } speciating_parameter_container;


/**********************************************************************
 * Default Genome
 **********************************************************************/
    typedef struct {

        unsigned int hidden = 20; // Initial nodes to be created
        std::string connection_type = "template"; // indirect | random | direct | template
        float connect_chance = 0.2; // if connection_Type == "random": Chance for connection to be added
        unsigned int template_mutate = 3; // if connection_type == "template": How many times to mutate the loaded genome

        // Serialization
        template<class Archive>
        void serialize(Archive &archive) {

            archive(CEREAL_NVP(hidden),
                    CEREAL_NVP(connection_type),
                    CEREAL_NVP(connect_chance),
                    CEREAL_NVP(template_mutate));
        }

    } defaultGenome;


/**********************************************************************
 * Network info
 **********************************************************************/
    typedef struct {
        unsigned int input_size;
        unsigned int output_size;
        bool recurrent;
        std::string s_path;


        template<class Archive>
        void serialize(Archive &archive) {

            archive(CEREAL_NVP(input_size),
                    CEREAL_NVP(output_size),
                    CEREAL_NVP(recurrent)),
                    CEREAL_NVP(s_path);
        }

    } network_info_container;



/**********************************************************************
 * Gene structs
 **********************************************************************/

    // Node gene struct
    typedef struct {
        unsigned int key;
        unsigned int activation_function; // 0 == sigmoid ; 1 == tanh ; 2 == sin
        unsigned int aggregation_function; // 0 == sum ; 1 == product ; 2 == mean
        double bias;
        double response;

        template<class Archive>
        void serialize(Archive &archive) {

            archive(CEREAL_NVP(key),
                    CEREAL_NVP(activation_function),
                    CEREAL_NVP(aggregation_function),
                    CEREAL_NVP(bias),
                    CEREAL_NVP(response));
        }
    } node_gene;


    // Connection gene struct
    typedef struct {
        unsigned int key;
        int from_node;
        unsigned int to_node;
        double weight;
        bool enabled;

        template<class Archive>
        void serialize(Archive &archive) {

            archive(CEREAL_NVP(key),
                    CEREAL_NVP(from_node),
                    CEREAL_NVP(to_node),
                    CEREAL_NVP(weight),
                    CEREAL_NVP(enabled));
        }
    } connection_gene;








/**********************************************************************
 * Genomes and species
 **********************************************************************/



    /**
     * Genome Class
     * Every genome is an instance of a ANN
     * The genes of the genome are constructed with connection and node genes
     *
     * Inputpins are always negative
     * Ouputpins are always positive starting at 0
     */
    class genome {
    private:


        // Default constructor should not be used
        genome() {}


    public:

        // Define
        double fitness = -9999.f;
        bool can_be_recurrent = false;
        unsigned int key;

        // Important containers
        mutation_rate_container mutation_rates;
        network_info_container network_info;

        // Input and output pins
        std::vector<int> input_pins;
        std::vector<int> output_pins;

        // Node- and connection gene maps
        std::vector<node_gene> node_genes;
        std::vector<connection_gene> connection_genes;


        /***************************************************************************
         * Constructor of Genome
         ***************************************************************************/
        genome(network_info_container &info, mutation_rate_container &rates, unsigned int genome_key) {
            mutation_rates = rates;
            network_info = info;
            can_be_recurrent = info.recurrent;
            this->key = genome_key;

            // Populate input and ouptut pins
            int ipin = -1;
            for (unsigned int i = 0; i < network_info.input_size; i++) {
                input_pins.push_back(ipin);
                ipin--;
            }

            // Check if it went ok
            if (input_pins.empty()) {
                ErrorLog::LogError("Input pins is empty!", network_info.s_path + "/ErrorLog.dat");
            }
            for (unsigned int i = 0; i < network_info.output_size; i++) {
                output_pins.push_back(i);
            }

        }


        /*****************************************************************************
         * Copy Constructor
         *****************************************************************************/
        genome(const genome &) = default;


        /*****************************************************************************
         * For serialization
         *****************************************************************************/
        template<class Archive>
        void serialize(Archive &archive) {

            archive(CEREAL_NVP(fitness),
                    CEREAL_NVP(can_be_recurrent),
                    CEREAL_NVP(mutation_rates),
                    CEREAL_NVP(network_info),
                    CEREAL_NVP(input_pins),
                    CEREAL_NVP(output_pins),
                    CEREAL_NVP(node_genes),
                    CEREAL_NVP(connection_genes));

        }
    };


    /**********************************************************************
     * a specie is group of genomes which differences is smaller
     * than some threshold staleness is the number of generations
     * the species didn't come up with better fitness
     **********************************************************************/
    typedef struct {

        double top_fitness = -9999.f;
        double average_fitness = -9999.f;
        unsigned int staleness = 0;
        int spawn_amount = 0;
        std::vector<genome> genomes;

    } specie;


/**********************************************************************
 * Genetic Pool
 * -------------
 * a small world, where individuals (genomes) are making babies and evolving,
 * becoming better and better after each generation :)
 **********************************************************************/
    class pool {
    private:
        pool() {};

        /*********************************************************
         *  innovation tracking in current generation
         *********************************************************/
        unsigned int innovation_nbr;

        unsigned int get_innovation_nbr();

        unsigned int connection_key;

        unsigned int get_connection_key();

        unsigned int genome_nbr;

        unsigned int GetGenomeNbr();

        // For Generation tracking
        unsigned int generation_number = 1;
        unsigned int last_change = 0;


        /*********************************************************
         * evolutionary methods
         *********************************************************/

        // Crossover
        genome crossover(const genome &g1, const genome &g2);

        // Mutate Connection
        void mutate_weight(genome &g);

        void mutate_enable_disable(genome &g);

        void mutate_addConnection(genome &g);

        void mutate_deleteConnection(genome &g);

        bool create_cycle(std::vector<connection_gene> &connections, connection_gene &test);

        void create_connection(genome &g, int from_key, int to_key);

        // Mutate nodes
        void mutate_activation_function(genome &g);

        void mutate_aggregation_function(genome &g);

        void mutate_node(genome &g);

        void add_node(genome &g);

        void delete_node(genome &g);

        void mutate_response(genome &g);

        void mutate_bias(genome &g);

        // main mutate function
        void mutate(genome &g);

        void create_random(genome &new_genome);

        void create_structural_indirect(genome &g);

        void create_structural_direct(genome &g);

        void create_fromArchive(genome &new_genome, cereal::BinaryInputArchive &s_archive);

        // Genetic distance
        double distance(genome &g1, genome &g2);

        // Some utils
        unsigned int count_genomes() {
            unsigned int count = 0;
            for (auto it_species = this->species.begin(); it_species != this->species.end(); it_species++) {

                count += it_species->genomes.size();
            }
            return count;
        }

        /* specie ranking */
        void rank_globally();

        void total_average_fitness();

        /* evolution */
        void cull_species(specie &s, unsigned int cut);

        genome breed_child(specie &s);

        void remove_stale_species();

        void add_to_species(genome &child);

        std::vector<int> compute_spawn();


    public:



        /*************************************************************
         * Defining parameters
         *************************************************************/
        /* pool parameters */
        double max_fitness = -9999.f;


        /* mutation parameters */
        mutation_rate_container mutation_rates;

        /* species parameters */
        speciating_parameter_container speciating_parameters;

        /* neural network parameters */
        network_info_container network_info;

        /* default Genome info */
        defaultGenome default_Genome;

        /* Pointer to best genome */

        std::string session_path;


        /*************************************************************
         * Best Genome
         *************************************************************/
         unsigned int best_key;
         double best_fitness;
         unsigned int best_nodeCnt;
         unsigned int best_connCnt;

        /*************************************************************
         * Generator for random stuff
         *************************************************************/
        // pool's local random number generator
        std::random_device rd;
        std::mt19937 generator;

        /* species */
        std::vector<specie> species;


        /************************************************************
         * Constructor for Pool
         ************************************************************/
        pool(std::string home_dir, unsigned int input, unsigned int output, bool rec = false);


        /************************************************************
         * Generations stuff
         ************************************************************/
        void new_generation();

        unsigned int generation() { return this->generation_number; }


    }; // End of pool class

} // End of namespace cneat


#endif //CNEAT_TRADER_CNEAT_H
