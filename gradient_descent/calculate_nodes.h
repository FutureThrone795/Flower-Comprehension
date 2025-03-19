double activation_function(double initial_value, int is_output_layer)
{
    return tanh(initial_value);

    if (is_output_layer)
    {
        
    } 
    else
    {
        return (initial_value > 0.0) ? initial_value : initial_value * 0.2;
    }
}

double activation_function_derivative(double initial_value, int is_output_layer)
{
    return 1.0 - pow(tanh(initial_value), 2.0);

    if (is_output_layer || 1)
    {
        
    } 
    else
    {
        return (initial_value > 0.0) ? 1.0 : 0.2;
    }
}

void compute_node(struct Node_Layer *node_layer, size_t output_index, int is_output_layer)
{
    double value_sum = 0.0;
    double current_weight;
    double current_input;

    for (size_t input_index = 0; input_index < node_layer->input_count; input_index++)
    {
        current_weight = node_layer->input_weights[(output_index * node_layer->input_count) + input_index];
        current_input = node_layer->inputs[input_index];

        value_sum += (current_weight * current_input);
    }
    
    value_sum += node_layer->activation_biases[output_index];

    (node_layer->raw_linear_outputs)[output_index] = value_sum;
    (node_layer->outputs)[output_index] = activation_function(value_sum, is_output_layer);
}

void compute_node_layer(struct Node_Layer *node_layer, int is_output_layer)
{
    for (size_t output_index = 0; output_index < node_layer->output_count; output_index++)
    {
        compute_node(node_layer, output_index, is_output_layer);
    }
}

void compute_node_network(struct Node_Network *node_network)
{
    for (size_t node_layer_index = 0; node_layer_index < node_network->node_layer_count; node_layer_index++)
    {
        compute_node_layer(&(node_network->node_layers[node_layer_index]), node_layer_index == (node_network->node_layer_count - 1));
    }
}