
float calculate_data_difference_squared(float *data_1, float *data_2, size_t data_size)
{
	float data_difference = 0;

	for (size_t data_index = 0; data_index < data_size; data_index++)
	{
		data_difference += pow(data_1[data_index] - data_2[data_index], 2.0);
	}

	return data_difference;
}

float calculate_aggregate_batch_data_difference_squared(struct Node_Network *node_network, float **batch_data, size_t batch_size, size_t individual_data_size)
{
	float aggregate_batch_accuracy = 0.0f;

	for (int data_index = 0; data_index < batch_size; data_index++)
	{
		node_network->node_layers[0].inputs = batch_data[data_index];
		compute_node_network(node_network);
		aggregate_batch_accuracy += calculate_data_difference_squared(batch_data[data_index], node_network->node_layers[node_network->node_layer_count - 1].outputs, individual_data_size);
	}

	return (aggregate_batch_accuracy / batch_size);
}