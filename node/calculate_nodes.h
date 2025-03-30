float activation_function(float initial_value, uint8_t is_output_layer)
{
    return tanh(initial_value);
}

float activation_function_derivative(float initial_value, uint8_t is_output_layer)
{
    return 1.0f - pow(tanh(initial_value), 2.0f);
}

void compute_node_layer(struct Node_Layer *node_layer, struct Node_Layer_Data_Partition *node_layer_data_partition, uint8_t is_output_layer)
{
    if (node_layer_data_partition->is_allocated_for_gradient_descent)
    {
        multiply_matrices(node_layer_data_partition->raw_linear_outputs, node_layer->input_weights, node_layer->output_count, node_layer->input_count, node_layer_data_partition->inputs, node_layer->input_count, 1);
        apply_activation_function_to_matrix(node_layer_data_partition->outputs, node_layer_data_partition->raw_linear_outputs, 1, node_layer->output_count, &activation_function, is_output_layer);
    }
    else
    {
        multiply_matrices(node_layer_data_partition->outputs, node_layer->input_weights, node_layer->output_count, node_layer->input_count, node_layer_data_partition->inputs, node_layer->input_count, 1);
        apply_activation_function_to_matrix(node_layer_data_partition->outputs, node_layer_data_partition->outputs, 1, node_layer->output_count, &activation_function, is_output_layer);
    }
}

void compute_node_network(struct Node_Network *node_network, struct Node_Network_Data_Partition *node_network_data_partition)
{
    for (uint8_t node_layer_index = 0; node_layer_index < node_network->node_layer_count; node_layer_index++)
    {
        compute_node_layer(&(node_network->node_layers[node_layer_index]), &(node_network_data_partition->node_layer_data_partitions[node_layer_index]), node_layer_index == (node_network->node_layer_count - 1));
    }
}