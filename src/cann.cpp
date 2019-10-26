//
// Created by alcon on 25.10.19.
//

#include "cann.h"

/****************************************************
 * Constructor
 ****************************************************/
cann::feed_forward_network::feed_forward_network()
{

}

/****************************************************
 *
 * Get iterator to node_gene with node_key - key
 *
 * @brief feed_forward_network::find_neuron
 * @param nodes
 * @param node_key
 * @return
 *
 ****************************************************/
std::vector<cneat::node_gene>::iterator cann::feed_forward_network::find_node(std::vector<cneat::node_gene> &nodes, int key) {

    for (auto node = nodes.begin(); node != nodes.end(); node++)
    {
        if (node->key == key)
        {
            return node;
        }
    }

    return nodes.end();
}


/**************************************************************************************
 * Creation of the network
 **************************************************************************************/

/****************************************************
 *
 * Collect the nodes whose state is required to compute the final network output(s).
 *
 * @brief neuralnet::required_for_output
 * @param input
 * @param output
 * @param connections
 * @return
 *
 ****************************************************/
std::vector<int> cann::feed_forward_network::required_for_output(std::vector<int> &input, std::vector<int> &output, std::vector<cneat::connection_gene> &connections)
{
    std::vector<int> required;
    std::vector<int> s;

    for (auto i = output.begin(); i != output.end(); i++)
    {
        if(std::find(s.begin(), s.end(), *i) == s.end())
        {
            s.push_back((*i));
        }
        if(std::find(required.begin(), required.end(), *i) == required.end())
        {
            required.push_back(*i);
        }
    }

    while (true)
    {
        // Find nodes not in S whose output is consumed by a node in s
        std::vector<int> t;
        for (auto i = connections.begin(); i != connections.end(); i++)
        {
            if (std::find(s.begin(), s.end(), i->from_node) == s.end() && std::find(s.begin(), s.end(), i->to_node) != s.end() && std::find(t.begin(), t.end(), i->from_node) == t.end() )
            {
                t.push_back(i->from_node);
            }
        }

        if (t.empty())
        {
            break;
        }

        std::vector<int> layer_nodes;
        for (auto i = t.begin(); i != t.end(); i++)
        {
            if (std::find(input.begin(), input.end(), *i) == input.end() && std::find(layer_nodes.begin(), layer_nodes.end(), *i) == layer_nodes.end())
            {
                layer_nodes.push_back((*i));
            }
        }

        if (layer_nodes.empty())
        {
            break;
        }

        // add to required and s
        for(auto layer_node : layer_nodes)
        {
            if(std::find(required.begin(), required.end(), layer_node) == required.end())
            {
                required.push_back(layer_node);
            }
        }
        for(auto it_t : t)
        {
            if(std::find(s.begin(), s.end(), it_t) == s.end())
            {
                s.push_back(it_t);
            }
        }

        //required.insert(required.end(), layer_nodes.begin(), layer_nodes.end());
        //s.insert(s.end(), t.begin(), t.end());
    }

    return required;
}

/****************************************************
 *
 * Collect the layers whose members can be evaluated in parallel in a feed-forward network.
 * Returns a list of layers, with each layer consisting of a set of node identifiers.
 * Note that the returned layers do not contain nodes whose output is ultimately
 * never used to compute the final network output.
 *
 * @brief feed_forward_network::feed_forward_layers
 * @param input
 * @param output
 * @param connections
 *
 ****************************************************/
std::vector<std::vector<int>> cann::feed_forward_network::feed_forward_layers(std::vector<int> &input, std::vector<int> &output,
                                          std::vector<cneat::connection_gene> &connections)
{
    std::vector<int> required = this->required_for_output(input, output, connections);
    std::vector<std::vector<int>> layers;
    std::vector<int> s;

    // append inputs to s
    for (auto i : input)
    {
        s.push_back(i);
    }

    while (true)
    {
        std::vector<unsigned int> c;
        for (auto it_conn : connections)
        {
            if (std::find(s.begin(), s.end(), it_conn.from_node) != s.end()
                && std::find(s.begin(), s.end(), it_conn.to_node) == s.end()
                && std::find(c.begin(), c.end(), it_conn.to_node) == c.end())
            {
                c.push_back(it_conn.to_node);
            }
        }


        std::vector<int> t;
        for (auto it_c : c)
        {
            std::vector<bool> vec_all;
            for (auto it_conn : connections)
            {
                if (it_conn.to_node == it_c)
                {
                    vec_all.push_back(std::find(s.begin(), s.end(), it_conn.from_node) != s.end());
                }
            }

            if (std::find(required.begin(), required.end(), it_c) != required.end()
                && std::find(vec_all.begin(), vec_all.end(), false) == vec_all.end()) {

                if (std::find(t.begin(), t.end(), it_c) == t.end())
                {
                    t.push_back(it_c);
                }
            }
        }

        if (t.empty())
        {
            break;
        }

        layers.push_back(t);
        for(auto it_t : t)
        {
            if(std::find(s.begin(), s.end(), it_t) == s.end())
            {
                s.push_back(it_t);
            }
        }
        // s.insert(s.end(), t.begin(), t.end());
    }


    return layers;

}


/****************************************************
 *
 * Receives a genome and returns its phenotype (a FeedForwardNetwork).
 *
 * @brief feed_forward_network::from_genome
 * @param g
 *
 ****************************************************/
void cann::feed_forward_network::from_genome(cneat::genome &g)
{
    if (g.can_be_recurrent)
    {
        std::cerr << "CAN_BE_RECURRENT: " << g.can_be_recurrent;
        ErrorLog::LogError("Genomes are recurrent in a FeedForwardNetwork!",
                           g.network_info.s_path + "/ErrorLog.dat");
    }

    // delete the current settings
    this->input_keys.clear();
    this->output_keys.clear();
    this->values.clear();
    this->node_evals.clear();

    if (g.input_pins.empty())
    {
        ErrorLog::LogError("INPUT PINS OF GENOME EMPTY", "/home/AnnErrorLog.dat");
    }

    this->input_keys = g.input_pins;
    this->output_keys = g.output_pins;

    // Gather expressed connections
    std::vector<cneat::connection_gene> connections;
    for (auto it_connection : g.connection_genes)
    {
        if (it_connection.enabled)
        {
            connections.push_back((it_connection));
        }
    }

    std::vector<std::vector<int>> layers = this->feed_forward_layers(g.input_pins, g.output_pins,g.connection_genes);

    for (auto layer : layers)
    {
        for (auto node : layer)
        {
            neuron new_neuron;
            for (auto conn : connections)
            {
                if (conn.to_node == node)
                {
                    new_neuron.inputs.push_back(std::make_pair(conn.from_node, conn.weight));
                }

            }
            auto it_node = this->find_node(g.node_genes, node);

            if (it_node != g.node_genes.end())
            {
                new_neuron.activation_function = it_node->activation_function;
                new_neuron.aggregation_function = it_node->aggregation_function;
                new_neuron.bias = it_node->bias;
                new_neuron.response = it_node->response;
                new_neuron.node = it_node->key;

                // Add to node_eval vector
                node_evals.push_back(new_neuron);
            }
        }
    }

    //Copy input / Output keys
    input_keys = g.input_pins;
    output_keys = g.output_pins;

    // pre-populate values
    for (auto input : input_keys)
    {
        values[input] = 0.0;
    }

    for (auto output : output_keys)
    {
        values[output] = 0.0;
    }
}

void cann::feed_forward_network::activate(std::vector<double> &inputs, std::vector<double> &outputs)
{
    // Check if inputs.size == input_keys.size
    if (inputs.size() != this->input_keys.size())
    {
        ErrorLog::LogError("Inputs.size() != input_keys.size()", "/home/AnnErrorLog.dat");
        throw std::runtime_error("Inputs.size() != input_keys.size()");
    }

    //Define
    double s;

    // Set input values
    size_t us_inputsSize = inputs.size();
    for (size_t i = 0; i < us_inputsSize; i++)
    {
        //values[ input_keys[i] ] = inputs[i];
        values.find(input_keys[i])->second = inputs[i];
    }

    // Computation loop
    for (auto it_neuron : node_evals)
    {
        // Aggregation function
        switch (it_neuron.aggregation_function)
        {
            case 0:
                s = agg_sum(it_neuron.inputs);
                break;

            case 1:
                s = agg_prod(it_neuron.inputs);
                break;

            case 2:
                s = agg_mean(it_neuron.inputs);
                break;
            default:
                s = agg_sum(it_neuron.inputs);
                break;
        }

        // Activation function
        switch (it_neuron.activation_function)
        {
            case 0:
                this->values[it_neuron.node] = act_sig(s * it_neuron.response + it_neuron.bias);
                break;

            case 1:
                this->values[it_neuron.node] = act_tanh(s * it_neuron.response + it_neuron.bias);
                break;

            case 2:
                this->values[it_neuron.node] = act_sin(s * it_neuron.response + it_neuron.bias);
                break;
            default:
                this->values[it_neuron.node] = act_sig(s * it_neuron.response + it_neuron.bias);
                break;
        }
    }

    // Push output values to output vector
    size_t us_outputSize = outputs.size();
    for (size_t us_it = 0; us_it < us_outputSize; us_it++)
    {
        outputs[us_it] = this->values[us_it];
    }
}


/********************************************************************************************************
 * Aggregationfunctions
 ********************************************************************************************************/

/****************************************************
 *
 * SUM
 *
 * @brief feed_forward_network::agg_sum
 * @param in
 * @return
 *
 *****************************************************/
double cann::feed_forward_network::agg_sum(std::vector<std::pair<int, double>> &links)
{
    double ret = 0.0;
    for (auto it_link : links)
    {
        ret += values[it_link.first] * it_link.second;
    }

    return ret;
}

/****************************************************
 *
 * PRRODUCT
 *
 * @brief feed_forward_network::agg_prod
 * @param in
 * @return
 *
 ****************************************************/
double cann::feed_forward_network::agg_prod(std::vector<std::pair<int, double>> &links)
{
    double ret = 0.0;
    for (auto it_link : links)
    {
        ret *= values[it_link.first] * it_link.second;
    }

    return ret;

}

/****************************************************
 *
 * MEAN
 *
 * @brief feed_forward_network::agg_mean
 * @param in
 * @return
 *
 ****************************************************/
double cann::feed_forward_network::agg_mean(std::vector<std::pair<int, double>> &links)
{
    double ret = 0.0;
    for (auto it_link : links)
    {
        ret += values[it_link.first] * it_link.second;
    }
    ret = ret / links.size();

    return ret;

}


/********************************************************************************************************
 * Activation functions
 ********************************************************************************************************/

/****************************************************
 *
 * Sigmoid function
 *
 * @brief feed_forward_network::act_sig
 * @param in
 * @return
 *
 ****************************************************/
double cann::feed_forward_network::act_sig(double in)
{
    return 1 / (1 + exp(-1 * in));
}

/****************************************************
 *
 * Sinus function
 *
 * @brief feed_forward_network::act_sin
 * @param in
 * @return
 *
 ****************************************************/
double cann::feed_forward_network::act_sin(double in)
{
    return sin(in);
}

/****************************************************
 *
 * Tangens hyperbolicus function
 *
 * @brief feed_forward_network::act_tanh
 * @param in
 * @return
 *
 ****************************************************/
double cann::feed_forward_network::act_tanh(double in)
{
    return tanh(in);
}