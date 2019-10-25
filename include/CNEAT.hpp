#ifndef _C_NEAT_HPP_
#define _C_NEAT_HPP_

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
        pool(unsigned int input, unsigned int output, bool rec = false) {
            this->network_info.input_size = input;
            this->network_info.output_size = output;
            this->network_info.recurrent = rec;
            this->innovation_nbr = output;
            this->connection_key = 0;
            this->genome_nbr = 0;


            /***********************
             * Load Settings
             ***********************/
            // Load Default genome
            std::ifstream load_fs;

            {
                load_fs.open("../config/default_genome.json");

                if (!load_fs.is_open()) {
                    throw std::runtime_error("Could not open ../config/default_genome.json");
                }
                cereal::JSONInputArchive genome_archive(load_fs);
                default_Genome.serialize(genome_archive);
            }
            load_fs.close();
            load_fs.clear();

            // Load speciating parameter
            {
                load_fs.open("../config/default_speciating_parameters.json");

                if (!load_fs.is_open()) {
                    throw std::runtime_error("Could not open ../config/default_speciating_parameters.json");
                }
                cereal::JSONInputArchive specie_archive(load_fs);
                speciating_parameters.serialize(specie_archive);
            }
            load_fs.close();
            load_fs.clear();

            // Load Mutation rates
            {
                load_fs.open("../config/default_mutation_rates.json");

                if (!load_fs.is_open()) {
                    throw std::runtime_error("Could not open ../config/default_mutation_rates.json");
                }
                cereal::JSONInputArchive mutation_archive(load_fs);
                mutation_rates.serialize(mutation_archive);

            }


            /**
             * seed the mersenne twister with
             * a random number from our computer
             */
            generator.seed(rd());
            std::chrono::high_resolution_clock::time_point s_creationStart;


            /**
             * Create a basic generation with default genomes
             */
            for (unsigned int i = 0; i < this->speciating_parameters.population; i++) {
                s_creationStart = std::chrono::high_resolution_clock::now();

                genome new_genome(this->network_info, this->mutation_rates, this->GetGenomeNbr());

                std::normal_distribution<> gauss_bias(0.0, this->mutation_rates.bias_mutation_rate);
                std::normal_distribution<> gauss_response(0.0, this->mutation_rates.response_mutation_rate);

                for (size_t us_i = 0; us_i < output; us_i++) {

                    node_gene new_node;
                    new_node.activation_function = 0;
                    new_node.aggregation_function = 0;
                    new_node.bias = gauss_bias(this->generator);
                    new_node.response = gauss_response(this->generator);
                    new_node.key = us_i;
                    new_genome.node_genes.push_back(new_node);
                }

                // Decide how to create the genome
                if (default_Genome.connection_type == "random") {
                    this->create_random(new_genome);
                } else if (default_Genome.connection_type == "direct") {
                    this->create_structural_direct(new_genome);
                } else if (default_Genome.connection_type == "indirect") {
                    this->create_structural_indirect(new_genome);
                } else if (default_Genome.connection_type == "template") {
                    std::ifstream i_fs;
                    i_fs.open("../template/template.genome", std::ios::binary);

                    if (!i_fs.is_open()) {
                        throw std::runtime_error("Error while loading template genome! Could not open file!");
                    }
                    cereal::BinaryInputArchive s_archive(i_fs);
                    this->create_fromArchive(new_genome, s_archive);
                }

                this->add_to_species(new_genome);
                std::cerr << "Added Genome " << i << " with size: N: " << new_genome.node_genes.size() << " C: "
                          << new_genome.connection_genes.size() << "  to species.  Took ";
                std::cerr << std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now() - s_creationStart).count();
                std::cerr << " seconds" << std::endl;
            }

            /**
             * Check if every species has at least this->speciating_parameters.min_survivors members
             */
            for (auto it_specie = this->species.begin(); it_specie != this->species.end(); it_specie++) {
                if (it_specie->genomes.size() < this->speciating_parameters.min_survivors) {
                    genome new_genome = it_specie->genomes[0];
                    this->mutate(new_genome);
                    it_specie->genomes.push_back(new_genome);
                }

            }


            /**
             * Create session path
             */

            std::cerr << "Creating new Session directory" << std::endl;
            unsigned int dir_nbr = 0;
            std::string home_dir = "..";
            std::string path_temp = home_dir + "/save/test_" + std::to_string(dir_nbr);

            while (mkdir(path_temp.c_str(), ACCESSPERMS) != 0) {
                dir_nbr++;
                path_temp = home_dir + "/save/test_" + std::to_string(dir_nbr);
            }

            std::cerr << "Created Session directory: " << path_temp << std::endl;
            std::string genome_path = path_temp + "/genomes";
            if (mkdir(genome_path.c_str(), ACCESSPERMS) != 0) {
                throw std::runtime_error("THIS MOTHERFUCKER CAN'T MAKE A FUCKING GENOMES FOLDER");
            }
            this->session_path = path_temp;
            this->network_info.s_path = path_temp;

        }


        /************************************************************
         * Generations stuff
         ************************************************************/
        void new_generation();

        unsigned int generation() { return this->generation_number; }


    }; // End of pool class



    /*************************************************************************************
     * Innovation stuff
    *************************************************************************************/


    /************************************************************************
     *
     * Get Innovation nbr ( aka node_key )
     *
     * @brief pool::get_innovation_nbr
     * @return
     *
     ************************************************************************/
    unsigned int pool::get_innovation_nbr() {
        this->innovation_nbr++;
        return this->innovation_nbr - 1;
    }


    /************************************************************************
     *
     * Get connection_key
     *
     * @brief pool::get_connection_key
     * @return
     *
     ************************************************************************/
    unsigned int pool::get_connection_key() {
        this->connection_key++;
        return this->connection_key - 1;
    }


    unsigned int pool::GetGenomeNbr() {
        this->genome_nbr++;
        return this->genome_nbr - 1;
    }



    /*************************************************************************************/
    /* mutations */
    /*************************************************************************************/


    /************************************************************************
     *
     * Take 2 parent genomes and cross over them to a new child genome
     *
     * @brief pool::crossover
     * @param g1
     * @param g2
     * @return genome
     *
     ************************************************************************/
    genome pool::crossover(const genome &g1, const genome &g2) {
        // Make sure g1 has the higher fitness, so we will include only disjoint/excess
        // genes from the first genome.
        if (g2.fitness > g1.fitness)
            return crossover(g2, g1);

        // Create new genome
        genome child(this->network_info, this->mutation_rates, this->GetGenomeNbr());
        child.can_be_recurrent = this->network_info.recurrent;

        /**
         *  We begin with the connection genes
         *  We will look for connection genes with the same key
         */

        for (auto it_g1 = g1.connection_genes.begin(); it_g1 != g1.connection_genes.end(); it_g1++) {


            // Search for the key in g2.connection genes
            auto it_g2 = find_key(g2.connection_genes.begin(), g2.connection_genes.end(), it_g1->key);
            if (it_g2 == g1.connection_genes.end()) {
                // If we did not find the same key, just keep the connection of g1
                child.connection_genes.push_back(*it_g1);
            } else {
                // If we found the same key ==> crossover
                std::uniform_real_distribution<double> choice(0.0, 1.0);
                connection_gene new_connection;


                // crossover weight
                if (choice(this->generator) < 0.5) {
                    new_connection.weight = it_g1->weight;
                } else {
                    new_connection.weight = it_g2->weight;
                }

                // crossover enabled/disabled
                if (choice(this->generator) < 0.5) {
                    new_connection.enabled = it_g1->enabled;
                } else {
                    new_connection.enabled = it_g2->enabled;
                }
                // set key and anchor point
                new_connection.from_node = it_g1->from_node;
                new_connection.to_node = it_g1->to_node;
                new_connection.key = it_g1->key;

                // add new_new connection to child
                child.connection_genes.push_back(new_connection);
            }
        }
        /**
         * Now do the same thing for the node genes
         */

        for (auto it_g1 = g1.node_genes.begin(); it_g1 != g1.node_genes.end(); it_g1++) {

            // Search for key in g2.node_genes
            auto it_g2 = find_key(g2.node_genes.begin(), g2.node_genes.end(), it_g1->key);
            if (it_g2 == g2.node_genes.end()) {
                // If we didn't find a match just keep node of g1
                child.node_genes.push_back(*it_g1);
            } else {

                // If we found a match perform crossover
                std::uniform_real_distribution<double> choice(0.0, 1.0);
                node_gene new_node;


                // Crossover activation function
                if (choice(this->generator) < 0.5) {

                    new_node.activation_function = it_g1->activation_function;
                } else {

                    new_node.activation_function = it_g2->activation_function;
                }

                // Crossover aggregation function
                if (choice(this->generator) < 0.5) {
                    new_node.aggregation_function = it_g1->aggregation_function;
                } else {

                    new_node.aggregation_function = it_g2->aggregation_function;
                }

                // Crossover bias
                if (choice(this->generator) < 0.5) {

                    new_node.bias = it_g1->bias;
                } else {

                    new_node.bias = it_g2->bias;
                }

                // Crossover response
                if (choice(this->generator) < 0.5) {
                    new_node.response = it_g1->response;
                } else {

                    new_node.response = it_g2->response;
                }
                // Add key to Node
                new_node.key = it_g1->key;

                // Add new_node to node_genes of child
                child.node_genes.push_back(new_node);
            }
        }

        return child;
    }


    /**
     * Mutate the weights of random connection genes
     *
     * @brief pool::mutate_weight
     * @param g
     *
     ************************************************************************/
    void pool::mutate_weight(genome &g) {
        if (g.connection_genes.empty()) { return; }

        // Define
        std::uniform_int_distribution<unsigned int> choice(0, g.connection_genes.size() - 1);
        std::normal_distribution<> gauss(0.0, this->mutation_rates.weight_mutation_Rate);
        unsigned int conn = choice(this->generator);
        g.connection_genes[conn].weight += gauss(this->generator);
    }


    /************************************************************************
     *
     * Enable or disable a random connection_gene
     *
     * @brief pool::mutate_enable_disable
     * @param g
     *
     ************************************************************************/
    void pool::mutate_enable_disable(genome &g) {
        if (g.connection_genes.empty()) { return; }

        // Defone
        std::uniform_int_distribution<int> choice(0, g.connection_genes.size() - 1);
        int cgene = choice(this->generator);

        // Chance enabled/disabled
        g.connection_genes[cgene].enabled = !g.connection_genes[cgene].enabled;
    }


    /************************************************************************
     *
     * Add new connection to genome
     *
     * @brief pool::mutate_link
     * @param g
     * @param force_bias
     *
     ************************************************************************/
    void pool::mutate_addConnection(genome &g) {
        int from_node_key;
        int to_node_key;

        // Choose random from_node
        std::uniform_int_distribution<int> coin_toss(0, 1);
        if (coin_toss(this->generator) == 1 || g.node_genes.empty() == 0) {
            // Choose from input nodes
            std::uniform_int_distribution<int> choice(0, g.input_pins.size() - 1);
            from_node_key = g.input_pins[choice(this->generator)];
        } else {

            // Choose from normal nodes
            std::uniform_int_distribution<int> choice(0, g.node_genes.size() - 1);
            from_node_key = g.node_genes[choice(this->generator)].key;
        }

        // Choose random to_node
        if (coin_toss(this->generator) == 1 &&
            from_node_key > 0) // Don't allow direct connections from input to output nodes
        {
            // Choose from output nodes
            std::uniform_int_distribution<int> choice(0, g.output_pins.size() - 1);
            to_node_key = g.output_pins[choice(this->generator)];
        } else {

            if (g.node_genes.size() > 0) {
                // Choose from normal nodes
                std::uniform_int_distribution<int> choice(0, g.node_genes.size() - 1);
                to_node_key = g.node_genes[choice(this->generator)].key;
            } else {

                // Choose from output nodes
                std::uniform_int_distribution<int> choice(0, g.output_pins.size() - 1);
                to_node_key = g.output_pins[choice(this->generator)];
            }
        }


        // Don't duplicate connections
        for (auto i = g.connection_genes.begin(); i != g.connection_genes.end(); i++) {

            if (i->from_node == from_node_key && i->to_node == to_node_key) {
                return;
            }
        }
        // Create new connection gene
        std::normal_distribution<> gauss(0.0, this->mutation_rates.weight_mutation_Rate);
        connection_gene new_conn_gene;
        new_conn_gene.enabled = true;
        new_conn_gene.from_node = from_node_key;
        new_conn_gene.to_node = static_cast<unsigned int>(to_node_key);
        new_conn_gene.weight = gauss(this->generator);
        new_conn_gene.key = this->get_connection_key();

        /*
         * If genome should be a feedforward network don't add connection if it would create
         * a cycle in the genome
         */
        if (!g.can_be_recurrent) {
            if (create_cycle(g.connection_genes, new_conn_gene)) {

                return;
            }
        } else if (g.can_be_recurrent && !this->network_info.recurrent) {
            g.can_be_recurrent = false;
        }

        // Add new connection to the connection_genes vector
        g.connection_genes.push_back(new_conn_gene);

    }


    /************************************************************************
     *
     * Returns true if the addition of the 'test' connection would create a cycle,
     * assuming that no cycle already exists in the graph represented by 'connections'.
     *
     * @brief pool::create_cycle
     * @param connections
     * @param test
     * @return
     *
     ************************************************************************/
    bool pool::create_cycle(std::vector<connection_gene> &connections, connection_gene &test) {
        // Cycle to same node
        if (test.from_node == test.to_node) {
            return true;
        }

        std::vector<unsigned int> visited;
        visited.push_back(test.to_node);

        while (true) {
            int num_added = 0;

            for (auto i = connections.begin(); i != connections.end(); i++) {

                if (std::find(visited.begin(), visited.end(), i->from_node) != visited.end() &&
                    std::find(visited.begin(), visited.end(), i->to_node) == visited.end()) {
                    if (i->to_node == test.from_node) {
                        return true;
                    }
                    visited.push_back(i->to_node);
                    num_added++;
                }
            }

            if (num_added == 0) {
                return false;
            }
        }
    }


    /************************************************************************
     *
     * Delete Connection
     * @brief pool::mutate_deleteConnection
     * @param g
     *
     ************************************************************************/
    void pool::mutate_deleteConnection(genome &g) {
        if (g.connection_genes.size() <= 1) { return; }

        // Get vector of all disabled connections
        std::vector<connection_gene *> vec_cons;
        for (auto it_con = g.connection_genes.begin(); it_con != g.connection_genes.end(); it_con++) {
            if (it_con->enabled) {
                vec_cons.push_back(&(*it_con));
            }
        }

        if (vec_cons.empty()) {
            return;
        }

        // Choose random connection gene to delete
        std::uniform_int_distribution<unsigned int> choice(0, vec_cons.size() - 1);
        auto conToDelete = vec_cons[choice(this->generator)];

        auto it_conGenes = g.connection_genes.begin();
        for (; it_conGenes != g.connection_genes.end(); it_conGenes++) {
            if (it_conGenes->key == conToDelete->key) {
                break;
            }
        }
        g.connection_genes.erase(it_conGenes);

    }


    /************************************************************************
     *
     * Create connection between from_key and to_key
     *
     * @brief create_connection
     * @param g
     * @param from_key
     * @param to_key
     *
     ************************************************************************/
    void pool::create_connection(genome &g, int from_key, int to_key) {
        std::normal_distribution<> gauss(0.0, this->mutation_rates.weight_mutation_Rate);

        connection_gene new_conn;
        new_conn.enabled = true;
        new_conn.from_node = from_key;
        new_conn.to_node = static_cast<unsigned int>(to_key);
        new_conn.weight = gauss(this->generator);
        new_conn.key = this->get_connection_key();

        g.connection_genes.push_back(new_conn);

    }


    /************************************************************************
     *
     * Change aggreagtion function of random node
     *
     * @brief pool::mutate_aggregation_function
     * @param g
     *
     ************************************************************************/
    void pool::mutate_aggregation_function(genome &g) {
        if (g.node_genes.size() == 0) { return; }

        // Random choice of genome
        std::uniform_int_distribution<unsigned int> choice(0, g.node_genes.size() - 1);
        std::uniform_int_distribution<unsigned int> agg(0,
                                                        static_cast<unsigned int>(g.mutation_rates.aggregation_choices -
                                                                                  1));

        // Get key voe node_genes vector
        unsigned int node = choice(this->generator);

        // get new aggreagation function key
        int agg_func = agg(this->generator);

        g.node_genes[node].aggregation_function = static_cast<unsigned int>(agg_func);


    }


    /************************************************************************
     *
     * Change activation function of random node
     *
     * @brief pool::mutate_activation_function
     * @param g
     *
     ************************************************************************/
    void pool::mutate_activation_function(genome &g) {
        if (g.node_genes.size() == 0) { return; }

        // Random choice of genome
        std::uniform_int_distribution<unsigned int> choice(0, g.node_genes.size() - 1);
        std::uniform_int_distribution<unsigned int> act(0,
                                                        static_cast<unsigned int>(g.mutation_rates.activation_choices -
                                                                                  1));

        // Get key voe node_genes vector
        unsigned int node = choice(this->generator);

        // get new activation function key
        int act_func = act(this->generator);

        g.node_genes[node].activation_function = static_cast<unsigned int>(act_func);
    }


    /************************************************************************
     *
     * Mutate nodes.
     *
     * @brief pool::mutate_node
     * @param g
     *
     ************************************************************************/
    void pool::mutate_node(genome &g) {
        if (g.node_genes.size() == 0) {
            this->add_node(g);
            return;
        }

        std::uniform_real_distribution<double> coin_flip(0.0, 1.0);
        if (coin_flip(this->generator) < g.mutation_rates.node_add_chance) {

            this->add_node(g);
        } else if (coin_flip(this->generator) < g.mutation_rates.node_delete_chance) {

            this->delete_node(g);
        }


        if (coin_flip(this->generator) < g.mutation_rates.aggregation_mutation_chance) {

            this->mutate_aggregation_function(g);
        } else if (coin_flip(this->generator) < g.mutation_rates.activation_mutation_chance) {

            this->mutate_activation_function(g);
        }

    }


    /************************************************************************
     *
     * Splitt existing connection and add a new node
     * Connect them to the old from and to genomes
     *
     * @brief pool::add_node
     * @param g
     *
     ************************************************************************/
    void pool::add_node(genome &g) {
        if (g.connection_genes.size() > 0) {
            // Choose random connection to split
            std::uniform_int_distribution<int> choice(0, g.connection_genes.size() - 1);

            int splitt_conn_key = choice(this->generator);
            int from_node = g.connection_genes[splitt_conn_key].from_node;
            int to_node = g.connection_genes[splitt_conn_key].to_node;
            double weight = g.connection_genes[splitt_conn_key].weight;

            g.connection_genes[splitt_conn_key].enabled = false;

            // Create new node_gene
            std::normal_distribution<> gauss_bias(0.0, this->mutation_rates.bias_mutation_rate);
            std::normal_distribution<> gauss_response(0.0, this->mutation_rates.response_mutation_rate);
            node_gene new_node;
            new_node.activation_function = 0;
            new_node.aggregation_function = 0;
            new_node.bias = gauss_bias(this->generator);
            new_node.response = gauss_response(this->generator);
            new_node.key = get_innovation_nbr();

            /**
             * Connect
             */
            // Create connections from and to the nodes;
            connection_gene new_con1;
            new_con1.enabled = true;
            new_con1.from_node = from_node;
            new_con1.to_node = new_node.key;
            new_con1.weight = weight;
            new_con1.key = get_connection_key();

            connection_gene new_con2;
            new_con2.from_node = new_node.key;
            new_con2.to_node = static_cast<unsigned int>(to_node);
            new_con2.enabled = true;
            new_con2.weight = weight;
            new_con2.key = get_connection_key();

            // Add genes to the genome
            g.connection_genes.push_back(new_con1);
            g.connection_genes.push_back(new_con2);
            g.node_genes.push_back(new_node);


        } else {

            /**
              * If there are no connections to splitt
              * create a new node and connect them to a random out and input node
              */


            // Create new node_gene
            std::normal_distribution<> gauss_bias(0.0, this->mutation_rates.bias_mutation_rate);
            std::normal_distribution<> gauss_response(0.0, this->mutation_rates.response_mutation_rate);
            node_gene new_node;
            new_node.activation_function = 0;
            new_node.aggregation_function = 0;
            new_node.bias = gauss_bias(this->generator);
            new_node.response = gauss_response(this->generator);
            new_node.key = get_innovation_nbr();

            /**
             * Connect
             */
            std::normal_distribution<> gauss(0.0, this->mutation_rates.weight_mutation_Rate);

            // Choose random input node and add a connection
            std::uniform_int_distribution<int> choice(g.input_pins.back(), -1);
            connection_gene new_con1;
            new_con1.enabled = true;
            new_con1.from_node = g.input_pins[choice(this->generator)];
            new_con1.to_node = new_node.key;
            new_con1.weight = gauss(this->generator);
            new_con1.key = get_connection_key();

            // Choose random output node and add a connection
            std::uniform_int_distribution<int> ochoice(0, g.output_pins.size());
            connection_gene new_con2;
            new_con2.enabled = true;
            new_con2.from_node = new_node.key;
            new_con2.to_node = static_cast<unsigned int>(g.output_pins[ochoice(this->generator)]);
            new_con2.weight = gauss(this->generator);
            new_con2.key = get_connection_key();

            g.connection_genes.push_back(new_con1);
            g.connection_genes.push_back(new_con2);
            g.node_genes.push_back(new_node);

        }
    }


    /************************************************************************
     *
     * Delete a random chosen node from the node_genes vector
     *
     * @brief pool::delete_node
     * @param g
     *
     ************************************************************************/
    void pool::delete_node(genome &g) {
        if (g.node_genes.size() <= 1) { return; }

        // Choose random node_gene and get its iterator
        std::uniform_int_distribution<int> choice(0, g.node_genes.size() - 2);
        auto it = g.node_genes.begin();
        it += choice(this->generator); // get the node element
        unsigned int node_key = it->key;// get the node key

        // Don't allow to delete output nodes
        if (std::find(g.output_pins.begin(), g.output_pins.end(), node_key) != g.output_pins.end()) { return; }

        // Search for connections with this from_node key
        for (auto i = g.connection_genes.begin(); i != g.connection_genes.end();) {
            if (i->from_node == node_key) {
                g.connection_genes.erase(i);
            } else {
                i++;
            }
        }


        // Search for connections with this to_node key
        for (auto i = g.connection_genes.begin(); i != g.connection_genes.end();) {
            if (i->to_node == node_key) {
                g.connection_genes.erase(i);
            } else {
                i++;
            }
        }
        // Delete node from node_genes vector
        g.node_genes.erase(it);
    }


    /************************************************************************
     *
     * Mutate response value of random node
     *
     * @brief pool::mutate_response
     * @param g
     *
     ************************************************************************/
    void pool::mutate_response(genome &g) {
        if (g.node_genes.size() == 0) { return; }

        // Choose random node
        std::uniform_int_distribution<unsigned int> choice(0, g.node_genes.size() - 1);
        std::normal_distribution<> gauss(0.0, this->mutation_rates.response_mutation_rate);
        unsigned int node = choice(this->generator);

        g.node_genes[node].response += gauss(this->generator);

    }


    /************************************************************************
     *
     * Mutate bias value of random node
     *
     * @brief pool::mutate_bias
     * @param g
     *
     ************************************************************************/
    void pool::mutate_bias(genome &g) {
        if (g.node_genes.size() == 0) { return; }

        // Choose random node
        std::uniform_int_distribution<unsigned int> choice(0, g.node_genes.size() - 1);
        std::normal_distribution<> gauss(0.0, this->mutation_rates.bias_mutation_rate);
        unsigned int node = choice(this->generator);

        g.node_genes[node].bias += g.node_genes[node].bias * mutation_rates.bias_mutation_rate;

    }


    /************************************************************************
     *
     * Mutate genome g
     *
     * @brief pool::mutate
     * @param g
     *
     ************************************************************************/
    void pool::mutate(genome &g) {
        std::uniform_real_distribution<double> mutate_or_not_mutate(0.0, 1.0);

        // Mutate weight
        if (mutate_or_not_mutate(this->generator) < g.mutation_rates.weight_mutate_chance)
            this->mutate_weight(g);

        // Mutate add connection
        if (g.mutation_rates.connection_add_chance > mutate_or_not_mutate(this->generator)) {
            this->mutate_addConnection(g);
        }

        // Mutate delete connection
        if (g.mutation_rates.connection_delete_chance > mutate_or_not_mutate(this->generator)) {
            this->mutate_deleteConnection(g);
        }

        // Mutate bias
        if (g.mutation_rates.bias_mutation_chance > mutate_or_not_mutate(this->generator)) {
            this->mutate_bias(g);
        }

        // Mutate response
        if (g.mutation_rates.response_mutation_chance > mutate_or_not_mutate(this->generator)) {
            this->mutate_response(g);
        }

        // Mutate enable of gene
        if (mutation_rates.enable_mutation_chance > mutate_or_not_mutate(this->generator)) {
            this->mutate_enable_disable(g);
        }

        // Mutate node
        this->mutate_node(g);


    }


    /************************************************************************
     *
     * Calculate contirbution to genetic distance by disjoint nodes and genes
     *
     * @brief pool::disjoint
     * @param g1
     * @param g2
     * @return
     *
     ************************************************************************/
    double pool::distance(genome &g1, genome &g2) {
        double node_distance = 0.0;
        double connection_distance = 0.0;
        unsigned int disjoint_node = 0;
        unsigned int disjoint_conncetion = 0;


        // Calculate which node_genes of g1 are not in g2
        for (auto i = g1.node_genes.begin(); i != g1.node_genes.end(); i++) {

            auto ii = g2.node_genes.begin();

            // Search for similar node_genes
            for (; ii != g2.node_genes.end(); ii++) {
                if (i->key == ii->key) { break; }

            }



            /*
             * If we did find a node_gene with the same key calculate the genetic distance
             * if not increment disjoint nodes
             */
            if (ii == g2.node_genes.end()) {
                disjoint_node++;
            } else {
                if (i->activation_function != ii->activation_function) {
                    node_distance += 1.0 * this->speciating_parameters.delta_weights;
                }
                if (i->aggregation_function != ii->aggregation_function) {
                    node_distance += 1.0 * this->speciating_parameters.delta_weights;
                }
            }
        }


        // Calculate which node_genes of g2 are not in g1
        for (auto i = g2.node_genes.begin(); i != g2.node_genes.end(); i++) {
            auto ii = g1.node_genes.begin();
            for (; ii != g1.node_genes.end(); ii++) {
                if (i->key != ii->key) {
                    break;
                }
            }
            if (ii == g1.node_genes.end()) {
                disjoint_node++;
            }
        }

        unsigned int max_nodes = std::max(g1.node_genes.size(), g2.node_genes.size());

        // Now calculate node distance
        node_distance = (node_distance + (disjoint_node * this->speciating_parameters.delta_disjoint)) / max_nodes;


        /**
         * Calculate connection distance
         */

        // Calculate wich connection_gens of g1 are not in g2
        for (auto i = g1.connection_genes.begin(); i != g1.connection_genes.end(); i++) {
            // find same key or not if not ii == g2.connection_genes.end()
            auto ii = g2.connection_genes.begin();
            for (; ii != g2.connection_genes.end(); ii++) {
                if (ii->key == i->key) {
                    break;
                }
            }
            if (ii == g2.connection_genes.end()) {
                disjoint_conncetion++;
            } else {
                if (i->enabled != ii->enabled) {
                    connection_distance += 1.0 * this->speciating_parameters.delta_weights;
                }
                if (i->weight != ii->weight) {
                    connection_distance += 1.0 * this->speciating_parameters.delta_weights;
                }
            }
        }

        // Calculate wich connection_genes of g2 are not in g1
        for (auto i = g2.connection_genes.begin(); i != g2.connection_genes.end(); i++) {
            auto ii = g1.connection_genes.begin();
            for (; ii != g1.connection_genes.end(); ii++) {
                if (i->key == ii->key) {
                    break;
                }
            }

            if (ii == g1.connection_genes.end()) {
                disjoint_conncetion++;
            }
        }
        unsigned int max_conn = std::max(g1.connection_genes.size(), g2.connection_genes.size());

        // Calculate connection_distance
        connection_distance =
                (connection_distance + (disjoint_conncetion * this->speciating_parameters.delta_disjoint)) / max_conn;

        return (connection_distance + node_distance) < this->speciating_parameters.delta_threshold;
    }


    /************************************************************************
     * Rank all genomes and report current Max Fitness;
     *
     * @brief pool::rank_globally
     *
     ************************************************************************/
    void pool::rank_globally() {
        std::vector<genome *> global;
        for (auto s = this->species.begin(); s != this->species.end(); s++) {
            for (size_t i = 0; i < (*s).genomes.size(); i++) {
                global.push_back(&((*s).genomes[i]));

                if (s->genomes[i].input_pins.empty()) {
                    // Populate input and ouptut pins
                    int ipin = -1;
                    for (unsigned int us_i = 0; us_i < this->network_info.input_size; us_i++) {
                        s->genomes[i].input_pins.push_back(ipin);
                        ipin--;
                    }
                }
            }
        }

        // Sort genomes
        std::sort(global.begin(), global.end(), [](genome *&a, genome *&b) -> bool {
            return a->fitness > b->fitness; // was a->fitness < b->fitness
        });

        // Report size of best genome
        std::cerr << "Best Genome nbr " << global[0]->key << " with size: N: " << global[0]->node_genes.size() << " C: "
                  << global[0]->connection_genes.size();
        std::cerr << " with fitness: " << global[0]->fitness << std::endl;
        if (global[0]->fitness > this->max_fitness) {
            this->max_fitness = global[0]->fitness;
            this->last_change = this->generation_number;

            // Write best genome to file
            {
                std::ofstream fs;
                std::string s_filename =
                        this->session_path + "/genomes/bestGen_" + std::to_string(this->generation_number) + ".genome";
                fs.open(s_filename, std::ios::binary);
                cereal::BinaryOutputArchive c_archive(fs);
                global[0]->serialize(c_archive);
            }
            std::cerr << " Serialized best one" << std::endl;
        }
        std::cerr << "Last change of fitness in generation: " << this->last_change << std::endl;
    }


    /************************************************************************
     *
     * Compute average fitness of all species
     *
     * @brief pool::total_average_fitness
     * @return
     *
     ************************************************************************/
    void pool::total_average_fitness() {

        std::vector<double> vec_fitness;
        for (auto specie : this->species) {

            for (auto genome : specie.genomes) {

                vec_fitness.push_back(genome.fitness);
            }
        }

        double max_fitness = *std::max_element(vec_fitness.begin(), vec_fitness.end());
        double min_fitness = *std::min_element(vec_fitness.begin(), vec_fitness.end());

        double fitness_range = std::max(1.0, max_fitness - min_fitness);

        for (auto it_specie = this->species.begin(); it_specie != this->species.end(); it_specie++) {

            double mfs = 0.0;
            for (auto genome : it_specie->genomes) {
                mfs += genome.fitness;
            }
            mfs = mfs / it_specie->genomes.size();

            double db_fitnessmf = (mfs - min_fitness) / fitness_range;
            it_specie->average_fitness = (mfs - min_fitness) / fitness_range;

        }
    }


    /***********************************************************************
     *
     * Sort species for fitness and cut down to cut
     *
     * @brief pool::cull_species
     * @param s
     * @param cut
     *
     ************************************************************************/
    void pool::cull_species(specie &s, unsigned int cut) {

        std::sort(s.genomes.begin(), s.genomes.end(),
                  [](genome &a, genome &b) { return a.fitness > b.fitness; });

        unsigned int remaining = cut;
        if (cut < this->speciating_parameters.min_survivors) {
            remaining = static_cast<unsigned int>(this->speciating_parameters.min_survivors);
        }

        if (cut == 0) {
            remaining = 1;
        }

        // letting him make more and more babies (until someone in
        // specie beat him or he becomes weaker during mutations
        while (s.genomes.size() > remaining)
            s.genomes.pop_back();


    }


    /************************************************************************
     *
     * Take random geneomes of species and reproduce
     * either sexually via crossover
     * or asexually
     *
     * @brief pool::breed_child
     * @param s
     * @return
     *
     ************************************************************************/
    genome pool::breed_child(specie &s) {
        // Create new child genome
        genome child(this->network_info, this->mutation_rates, this->GetGenomeNbr());

        child.can_be_recurrent = this->network_info.recurrent;

        //randomizing stuff
        std::uniform_real_distribution<double> distributor(0.0, 1.0);
        std::uniform_int_distribution<unsigned int> choose_genome(0, s.genomes.size() - 1);

        /*
         * If this is true do a crossover of 2 random genomes in the species
         */
        if (distributor(this->generator) < this->mutation_rates.crossover_chance) {
            unsigned int g1id, g2id;
            genome &g1 = s.genomes[g1id = choose_genome(this->generator)];
            genome &g2 = s.genomes[g2id = choose_genome(this->generator)];


            if (g1id == g2id) { // Asexual reproduction
                genome &g = s.genomes[g1id];
                child = g;
            } else { // sexual reproduction
                child = this->crossover(g1, g2);
            }
        } else {   // Asexual reproduction of random genome
            genome &g = s.genomes[choose_genome(this->generator)];
            child = g;
        }

        // Now mutate and return Child genome
        this->mutate(child);
        return child;
    }


    /************************************************************************
     *
     * Check if Species has improved over the last generation
     * If not: Incement staleness and check if it needs to be removed
     *
     * @brief pool::remove_stale_species
     *
     ************************************************************************/
    void pool::remove_stale_species() {

        for (auto it_s = this->species.begin(); it_s != this->species.end();) {
            // Increment staleness
            it_s->staleness++;

            for (auto it_g = it_s->genomes.begin(); it_g != it_s->genomes.end(); it_g++) {
                if (it_g->fitness > it_s->top_fitness) {
                    it_s->top_fitness = it_g->fitness;
                    it_s->staleness = 0;
                }
            }

            if (it_s->staleness > this->speciating_parameters.stale_species
                && this->species.size() > 1
                && it_s->top_fitness < this->max_fitness) {

                std::cerr << "Species with top fitness " << it_s->top_fitness << " is stale" << std::endl;
                this->species.erase(it_s);

            } else {
                it_s++;
            }
        }
    }


    /************************************************************************
     *
     * check if child genome belongs to a species if not => create a new species
     *
     * @brief pool::add_to_species
     * @param child
     *
     ************************************************************************/
    void pool::add_to_species(genome &child) {
        auto s = this->species.begin();
        std::uniform_int_distribution<unsigned int> choice;

        // Check if child-genome by genetic distance belongs to a species
        while (s != this->species.end()) {
            choice = std::uniform_int_distribution<unsigned int>(0, s->genomes.size() - 1);
            if (this->distance((*s).genomes[choice(this->generator)], child)) {
                (*s).genomes.push_back(child);
                break;
            }
            ++s;
        }

        /*********************************************************
         * If this is true ==> child-genome doesn't belong
         * to any species
         * Generate new species then and add it to species-vector
        **********************************************************/
        if (s == this->species.end()) {
            specie new_specie;
            new_specie.genomes.push_back(child);
            this->species.push_back(new_specie);
        }

    }


    std::vector<int> pool::compute_spawn() {

        double f64_sumAfs = 0.0;
        for (auto specie : this->species) {

            f64_sumAfs += specie.average_fitness;
        }
        f64_sumAfs = f64_sumAfs / this->species.size();


        std::vector<int> spawn_amounts;
        for (auto specie : this->species) {

            double f64_s = 0;
            if (f64_sumAfs > 0) {

                double f64_x = (specie.average_fitness / f64_sumAfs) * this->speciating_parameters.population;
                f64_s = std::max(static_cast<double>(this->speciating_parameters.min_survivors), f64_x);
            } else {
                f64_s = this->speciating_parameters.min_survivors;
            }

            double f64_d = (f64_s - static_cast<double>( specie.genomes.size())) * 0.5;
            int f64_c = static_cast<int>( std::round(f64_d));

            int spawn = specie.genomes.size();
            if (std::abs(f64_c) > 0) {
                spawn += f64_c;
            } else if (f64_d > 0) {
                spawn += 1;
            } else if (f64_d < 0) {
                spawn -= 1;
            }

            spawn_amounts.push_back(spawn);
        }


        // Normalize the stuff
        int total_spawn = 0;
        for (auto sp : spawn_amounts) {
            total_spawn += sp;
        }
        double norm = static_cast<double>(this->speciating_parameters.population) / static_cast<double>(total_spawn);

        for (size_t us_spawn = 0; us_spawn < spawn_amounts.size(); us_spawn++) {

            spawn_amounts[us_spawn] = std::max(this->speciating_parameters.min_survivors,
                                               static_cast<int>( std::round(spawn_amounts[us_spawn] * norm)));
        }


        return spawn_amounts;
    }


    /************************************************************************
     *
     * After complete computation of the fitness of each genome
     * We want a new generation made up from the old one
     *
     * @brief pool::new_generation
     ************************************************************************/
    void pool::new_generation() {

        // Ya know ranking for... ya know... competition
        this->rank_globally();

        // Incement staleness and remove stale species
        this->remove_stale_species();

        // Calculate average fitness for every specie
        this->total_average_fitness();

        // Compute Spawnamounts for each species
        std::vector<int> spawn_amounts = this->compute_spawn();



        /**
         *
         *          Breed Children from random genomes in the species
         *
         */
        std::vector<genome> children;

        for (size_t us_spawn = 0; us_spawn < spawn_amounts.size(); us_spawn++) {

            int repro_cutoff = static_cast<int>( std::ceil(
                    this->speciating_parameters.survival_threshhold * this->species[us_spawn].genomes.size()));
            this->cull_species(this->species[us_spawn], repro_cutoff);

            std::uniform_int_distribution<unsigned int> choice(0, this->species[us_spawn].genomes.size() - 1);
            for (size_t us_breed = 0; us_breed < spawn_amounts[us_spawn]; us_breed++) {

                children.push_back(this->breed_child(this->species[us_spawn]));
            }


        }

        std::cerr << "Made " << children.size() << " Children" << std::endl;


        /*********************************************************************************************************************************
         * Now we add the child genomes to the corresponding species
         * We want to create a new species if the genetic distance is too high
         *
         *********************************************************************************************************************************/



        /**
         *
         *         Now add child-genomes to the correspondig species
         *
         */



        // Shuffle the genome so the first species are not privileged
        std::shuffle(children.begin(), children.end(), this->generator);


        auto it_child = children.begin();
        while (this->count_genomes() < this->speciating_parameters.population && it_child != children.end()) {

            this->add_to_species(*it_child);
            it_child++;
        }

        /**
         *
         *          Make sure every species has at least this->speciation_parameters.min_survivors members
         *
         */

        auto it_species = this->species.begin();
        while (it_species != this->species.end()) {

            if (it_species->genomes.empty()) {

                this->species.erase(it_species);
                continue;
            }


            while (it_species->genomes.size() < this->speciating_parameters.min_survivors) {

                genome new_genome = it_species->genomes[0];
                this->mutate(new_genome);
                it_species->genomes.push_back(new_genome);
            }
            it_species++;
        }


        // Increment generation number
        this->generation_number++;
    }


    /************************************************************************
     *
     * Create default genome with random connections
     *
     * @brief pool::create_random
     * @param new_genome
     *
     ************************************************************************/
    void pool::create_random(genome &new_genome) {



        // Add nodes
        for (unsigned int i = 0; i < this->default_Genome.hidden; i++) {

            this->add_node(new_genome);
        }

        // add connections;
        std::uniform_real_distribution<float> choice(0.0, 1.0);
        for (size_t us_i = 0; us_i < new_genome.input_pins.size(); us_i++) {

            for (size_t us_ii = 0; us_ii < new_genome.node_genes.size(); us_ii++) {

                if (choice(this->generator) < this->default_Genome.connect_chance) {

                    this->create_connection(new_genome, new_genome.input_pins[us_i], new_genome.node_genes[us_ii].key);
                }

            }
        }

        for (size_t us_i = 0; us_i < new_genome.node_genes.size(); us_i++) {

            for (size_t us_ii = 0; us_ii < new_genome.output_pins.size(); us_ii++) {

                if (choice(this->generator) < this->default_Genome.connect_chance) {

                    this->create_connection(new_genome, new_genome.node_genes[us_i].key, new_genome.output_pins[us_ii]);
                }
            }
        }

    }


    /************************************************************************
     *
     * Create default genome with predefined indirect structure
     *
     * @brief pool::create_structural
     * @param g
     *
     ************************************************************************/
    void pool::create_structural_indirect(genome &g) {

        // create half as many nodes as there are inputs
        std::normal_distribution<> gauss_bias(0.0, this->mutation_rates.bias_mutation_rate);
        std::normal_distribution<> gauss_response(0.0, this->mutation_rates.response_mutation_rate);
        for (size_t i = 0; i < g.input_pins.size() / 2; i++) {

            node_gene new_node;
            new_node.key = this->get_innovation_nbr();
            new_node.activation_function = 0;
            new_node.aggregation_function = 0;
            new_node.bias = gauss_bias(this->generator);
            new_node.response = gauss_response(this->generator);

            g.node_genes.push_back(new_node);
        }

        // Now 2 input pins with 1 node
        std::normal_distribution<> gauss(0.0, this->mutation_rates.weight_mutation_Rate);
        auto it_input = g.input_pins.begin();
        for (size_t i = g.output_pins.size(); i < g.node_genes.size(); i++) {

            if (it_input == g.input_pins.end()) { break; }


            // Create connection from first node to same node
            connection_gene new_connection;
            new_connection.enabled = true;
            new_connection.weight = gauss(this->generator);
            new_connection.key = this->get_connection_key();
            new_connection.from_node = *it_input;
            new_connection.to_node = g.node_genes[i].key;

            g.connection_genes.push_back(new_connection);

            it_input++;


            // Create connection from second node to same node
            connection_gene new_connection2;
            new_connection2.enabled = true;
            new_connection2.weight = gauss(this->generator);
            new_connection2.key = this->get_connection_key();
            new_connection2.from_node = *it_input;
            new_connection2.to_node = g.node_genes[i].key;

            g.connection_genes.push_back(new_connection2);

            it_input++;

        }


        // Now connect nodes with a probability of 0.5 to the output nodes
        std::uniform_real_distribution<double> flip(0.0, 1.0);
        for (size_t out = 0; out < g.output_pins.size(); out++) {

            for (auto node = g.node_genes.begin(); node != g.node_genes.end(); node++) {

                if (flip(this->generator) < 1.0
                    && std::find(g.output_pins.begin(), g.output_pins.end(), node->key) == g.output_pins.end()) {

                    connection_gene new_connection;
                    new_connection.enabled = true;
                    new_connection.weight = gauss(this->generator);
                    new_connection.key = this->get_connection_key();
                    new_connection.from_node = node->key;
                    new_connection.to_node = static_cast<unsigned int>(g.output_pins[out]);

                    g.connection_genes.push_back(new_connection);
                }
            }
        }

    }


    /************************************************************************
     *
     * Create default genome with predefined direct structure
     *
     * @brief pool::create_structural
     * @param g
     *
     ************************************************************************/
    void pool::create_structural_direct(genome &g) {
        std::normal_distribution<> gauss(0.0, this->mutation_rates.weight_mutation_Rate);

        for (size_t i = 0; i < g.output_pins.size(); i++) {

            for (size_t ii = 0; ii < g.input_pins.size(); ii++) {

                connection_gene new_connection;
                new_connection.enabled = true;
                new_connection.weight = gauss(this->generator);
                new_connection.key = this->get_connection_key();
                new_connection.from_node = g.input_pins[ii];
                new_connection.to_node = static_cast<unsigned int>(g.output_pins[i]);

                g.connection_genes.push_back(new_connection);
            }
        }

        for (size_t i = 0; i < this->default_Genome.hidden; i++) {

            std::normal_distribution<> gauss_bias(0.0, this->mutation_rates.bias_mutation_rate);
            std::normal_distribution<> gauss_response(0.0, this->mutation_rates.response_mutation_rate);
            node_gene new_node;
            new_node.key = this->get_innovation_nbr();
            new_node.activation_function = 0;
            new_node.aggregation_function = 0;
            new_node.bias = gauss_bias(this->generator);
            new_node.response = gauss_response(this->generator);

            g.node_genes.push_back(new_node);

        }

    }


    /************************************************************************
     *
     * Create a genome from an given template genome in an archive
     * Mutate it this->default_Genome.template_mutate times
     *
     * @param new_genome
     * @param s_archive
     *
     ************************************************************************/
    void pool::create_fromArchive(genome &new_genome, cereal::BinaryInputArchive &s_archive) {


        new_genome.serialize(s_archive);

        for (size_t us_i = 0; us_i < this->default_Genome.template_mutate; us_i++) {
            this->mutate(new_genome);
        }
    }


} // end of namespace cneat

#endif

