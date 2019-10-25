//
// Created by alcon on 25.10.19.
//

#ifndef CNEAT_TRADER_CANN_H
#define CNEAT_TRADER_CANN_H


// C / C++
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>

// External
#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/utility.hpp>
#include <ErrorLog.hpp>

// Project
#include <cneat.h>


namespace cann {

    typedef struct {

        int node;
        int aggregation_function;
        int activation_function;
        double response;
        double bias;
        std::vector<std::pair<int, double>> inputs;

        // Serialization
        template<class Archive>
        void serialize(Archive &archive) {

            archive(node,
                    aggregation_function,
                    activation_function,
                    response,
                    bias,
                    inputs);
        }

    } neuron;


    /******************************************************************************************************************************************************************************
     *
     * Neural network class for feed forward networks
     * No recurency, working with layers
     *
     * @brief The feed_forward_network class
     *
     ******************************************************************************************************************************************************************************/
    class feed_forward_network {

    private:
        // Perma stuff
        std::vector<int> input_keys;
        std::vector<int> output_keys;
        std::vector<neuron> node_evals;

        //Changes every activation
        std::unordered_map<int, double> values;

    public:
        /****************************************************
         * Default constructor
         ****************************************************/
        feed_forward_network();

        /****************************************************
         * Default Destructor
         *****************************************************/
        ~feed_forward_network() {
            input_keys.clear();
            output_keys.clear();
            node_evals.clear();
            values.clear();
        }


        /****************************************************
         * Create neuralnet from genome
         ****************************************************/
        std::vector<int> required_for_output(std::vector<int> &input, std::vector<int> &output,
                                             std::vector<cneat::connection_gene> &connections);

        std::vector<std::vector<int>> feed_forward_layers(std::vector<int> &input, std::vector<int> &output,
                                                          std::vector<cneat::connection_gene> &connections);

        void from_genome(cneat::genome &g);


        /****************************************************
         * Evalutae the genome
         ****************************************************/
        void activate(std::vector<double> &inputs, std::vector<double> &outputs);


        /****************************************************
         * Aggregation functions
         ****************************************************/
        inline double agg_sum(std::vector<std::pair<int, double>> &links);

        inline double agg_mean(std::vector<std::pair<int, double>> &links);

        inline double agg_prod(std::vector<std::pair<int, double>> &links);


        /****************************************************
         * Activation functions
         ****************************************************/
        inline double act_sin(double in);

        inline double act_tanh(double in);

        inline double act_sig(double in);

        /****************************************************
         * Utils
         ****************************************************/
        std::vector<cneat::node_gene>::iterator find_node(std::vector<cneat::node_gene> &nodes, int node_key);


        template<class Archive>
        void serialization(Archive &archive) {

            archive(input_keys,
                    output_keys,
                    node_evals,
                    values);
        }


    };

} // End of namesace cann

#endif //CNEAT_TRADER_CANN_H
