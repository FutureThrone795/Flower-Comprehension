void save_progress_image(float *adjoined_progress_image_data, struct Node_Network *node_network, struct Node_Network_Data_Partition *node_network_data_partition, float **image_data, unsigned long long batch_size, unsigned long long image_width, unsigned long long image_height, unsigned long long image_channels, unsigned long long cycle_index)
{
    char name[64];

    for (int image_data_index = 0; image_data_index < batch_size; image_data_index++)
    {
        node_network_data_partition->node_layer_data_partitions[0].inputs = image_data[image_data_index];
        compute_node_network(node_network, node_network_data_partition);
        add_to_adjoined_image(adjoined_progress_image_data, image_data[image_data_index], node_network_data_partition->node_layer_data_partitions[node_network->node_layer_count - 1].outputs, image_width, image_height, image_channels, image_data_index * image_width * image_height * image_channels * 2);
        snprintf(name, 64, "node_networks/node_network_image_%i.png", image_data_index);
    }

    snprintf(name, 60, "progress/progress_image_%llu.png", cycle_index);
    save_image(name, image_width * 2, image_height * batch_size, image_channels, adjoined_progress_image_data);
    save_image("progress_image.png", image_width * 2, image_height * batch_size, image_channels, adjoined_progress_image_data);

    printf("Saved Progress Image\n");
}

void print_aggregate_batch_accuracy(float aggregate_batch_accuracy)
{
    static float prev_aggregate_batch_accuracy = 0.0;

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