struct Node_Network
{
    uint8_t node_layer_count;

    unsigned long long first_node_layer_input_count;
    unsigned long long* node_layer_output_counts;

    struct Node_Layer *node_layers;
};

void initialize_node_network(struct Node_Network *node_network, float* input, uint8_t node_layer_count, unsigned long long first_node_layer_input_count, unsigned long long* node_layer_output_counts)
{
    node_network->node_layer_count = node_layer_count;
    node_network->first_node_layer_input_count = first_node_layer_input_count;
    node_network->node_layer_output_counts = node_layer_output_counts;

    node_network->node_layers = (struct Node_Layer *)malloc(sizeof(struct Node_Layer) * node_layer_count);
    if (node_network->node_layers == NULL)
    {
        printf("Failed to allocate memory for node network node layers");
        exit(EXIT_FAILURE);
    }

    initialize_node_layer(&(node_network->node_layers[0]), first_node_layer_input_count, node_layer_output_counts[0]);

    for (uint8_t node_layer_index = 1; node_layer_index < node_layer_count; node_layer_index++)
    {
        initialize_node_layer(&(node_network->node_layers[node_layer_index]), node_layer_output_counts[node_layer_index - 1], node_layer_output_counts[node_layer_index]);
    }
}

void randomize_node_network_weights_and_biases(struct Node_Network *node_network, float maximum_weight_value, float maximum_bias_value)
{
    for (uint8_t node_layer_index = 0; node_layer_index < node_network->node_layer_count; node_layer_index++)
    {
        propogate_random_node_layer_weights(&(node_network->node_layers[node_layer_index]), maximum_weight_value);
        propogate_random_node_layer_biases(&(node_network->node_layers[node_layer_index]), maximum_bias_value);
    }
}