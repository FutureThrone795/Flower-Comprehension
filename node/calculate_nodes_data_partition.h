/*

These structs store the parts of a network that are required to calculate their outputs and then store their outputs, so that node calculate operations can be threaded

It also handles values for gradient descent that can't be threaded; i.e. output derivatives and raw linear outputs. 
These are only defined if "is allocated_for_gradient_descent" is true- don't edit this value.

*/

struct Node_Layer_Data_Partition
{
    float *inputs;
    float *outputs;

    uint8_t is_allocated_for_gradient_descent;
    float *output_derivatives;
    float *raw_linear_outputs;
};

struct Node_Network_Data_Partition
{
    uint8_t is_allocated_for_gradient_descent;
    struct Node_Layer_Data_Partition *node_layer_data_partitions;
};

void initialize_node_network_data_partition(struct Node_Network *node_network, struct Node_Network_Data_Partition *node_network_data_partition)
{
    for (uint8_t node_layer_index = 1; node_layer_index < node_network->node_layer_count; node_layer_index++)
    {
        node_network_data_partition->node_layer_data_partitions[node_layer_index].inputs = node_network_data_partition->node_layer_data_partitions[node_layer_index - 1].outputs;
    }
}

void allocate_node_layer_data_partition(struct Node_Layer_Data_Partition *node_layer_data_partition, unsigned long long output_count, uint8_t should_allocate_for_gradient_descent)
{
    node_layer_data_partition->is_allocated_for_gradient_descent = should_allocate_for_gradient_descent;

    node_layer_data_partition->outputs = (float *)malloc(sizeof(float) * output_count);
    if (node_layer_data_partition->outputs == NULL)
    {
        printf("Failed to allocate memory for node layer data partition outputs\n");
        exit(EXIT_FAILURE);
    }

    if(should_allocate_for_gradient_descent)
    {
        node_layer_data_partition->raw_linear_outputs = (float *)malloc(sizeof(float) * output_count);
        if (node_layer_data_partition->raw_linear_outputs == NULL)
        {
            printf("Failed to allocate memory for node layer data partition raw linear outputs\n");
            exit(EXIT_FAILURE);
        }

        node_layer_data_partition->output_derivatives = (float *)malloc(sizeof(float) * output_count);
        if (node_layer_data_partition->output_derivatives == NULL)
        {
            printf("Failed to allocate memory for node layer data partition output derivatives\n");
            exit(EXIT_FAILURE);
        }
    }
}

void allocate_node_network_data_partition(struct Node_Network *node_network, struct Node_Network_Data_Partition *node_network_data_partition, uint8_t should_allocate_for_gradient_descent)
{
    node_network_data_partition->is_allocated_for_gradient_descent = should_allocate_for_gradient_descent;

    node_network_data_partition->node_layer_data_partitions = (struct Node_Layer_Data_Partition *)malloc(sizeof(struct Node_Layer_Data_Partition) * node_network->node_layer_count);
    if (node_network_data_partition->node_layer_data_partitions == NULL)
    {
        printf("Failed to allocate memory for node network data partition node layer data partitions\n");
        exit(EXIT_FAILURE);
    }

    for (uint8_t node_layer_data_partition_index = 0; node_layer_data_partition_index < node_network->node_layer_count; node_layer_data_partition_index++)
    {
        allocate_node_layer_data_partition(&(node_network_data_partition->node_layer_data_partitions[node_layer_data_partition_index]), node_network->node_layer_output_counts[node_layer_data_partition_index], should_allocate_for_gradient_descent);
    }
}