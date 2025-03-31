
float calculate_data_difference_squared(float *data_1, float *data_2, uint64_t data_size)
{
	float data_difference = 0;

	for (uint64_t data_index = 0; data_index < data_size; data_index++)
	{
		data_difference += pow(data_1[data_index] - data_2[data_index], 2.0);
	}

	return data_difference;
}

float calculate_aggregate_batch_data_difference_squared(struct Node_Network *node_network, struct Node_Network_Data_Partition *node_network_data_partition, float **batch_data, uint64_t batch_size, uint64_t individual_data_size)
{
	float aggregate_batch_accuracy = 0.0f;

	for (int data_index = 0; data_index < batch_size; data_index++)
	{
		node_network_data_partition->node_layer_data_partitions[0].inputs = batch_data[data_index];
		compute_node_network(node_network, node_network_data_partition);
		aggregate_batch_accuracy += calculate_data_difference_squared(batch_data[data_index], node_network_data_partition->node_layer_data_partitions[node_network->node_layer_count - 1].outputs, individual_data_size);
	}

	return (aggregate_batch_accuracy / batch_size);
}