void save_progress_image(float *adjoined_progress_image_data, struct Node_Network *node_network, float **image_data, uint8_t batch_size, size_t image_width, size_t image_height, size_t image_channels, size_t cycle_index)
{
    char name[64];

    for (int image_data_index = 0; image_data_index < batch_size; image_data_index++)
    {
        node_network->node_layers[0].inputs = image_data[image_data_index];
        compute_node_network(node_network);
        add_to_adjoined_image(adjoined_progress_image_data, image_data[image_data_index], node_network->node_layers[node_network->node_layer_count - 1].outputs, image_width, image_height, image_channels, image_data_index * image_width * image_height * image_channels * 2);
        sprintf_s(name, 64, "node_networks/node_network_image_%i.png", image_data_index);
        if (DEBUG_SHOULD_SAVE_NODE_NETWORK_IMAGE)
        {
            save_node_network_as_image(node_network, name);
        }
    }

    sprintf_s(name, 60, "progress/progress_image_%i.png", cycle_index);
    save_image(name, image_width * 2, image_height * batch_size, image_channels, adjoined_progress_image_data);
    save_image("progress_image.png", image_width * 2, image_height * batch_size, image_channels, adjoined_progress_image_data);

    printf("Saved Progress Image\n");
}

void print_aggregate_batch_accuracy(struct Node_Network *node_network, float **image_data, size_t batch_size, size_t image_size)
{
    static float prev_aggregate_batch_accuracy = 0.0;
    float aggregate_batch_accuracy = calculate_aggregate_batch_data_difference_squared(node_network, image_data, batch_size, image_size);

    printf("Gradient Descent Step Completed, Aggregate Batch Accuracy: %f ", aggregate_batch_accuracy);

    if (prev_aggregate_batch_accuracy != 0.0)
    {
        if (aggregate_batch_accuracy > prev_aggregate_batch_accuracy)
        {
            printf("(+)");
        }
        else
        {
            printf("(-)");
        }
    }

    printf("\n");

    prev_aggregate_batch_accuracy = aggregate_batch_accuracy;
}