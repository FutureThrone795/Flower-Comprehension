struct Node_Layer
{
    unsigned long long input_count;
    unsigned long long output_count;

    float *input_weights;
    float *activation_biases;
};

void initialize_node_layer(struct Node_Layer *node_layer, unsigned long long input_count, unsigned long long output_count)
{
    node_layer->input_count = input_count;
    node_layer->output_count = output_count;

    node_layer->input_weights = (float *)malloc(sizeof(float) * input_count * output_count);
    if (node_layer->input_weights == NULL)
    {
        printf("Failed to allocate memory for node layer input weights");
        exit(EXIT_FAILURE);
    }

    node_layer->activation_biases = (float *)malloc(sizeof(float) * output_count);
    if (node_layer->activation_biases == NULL)
    {
        printf("Failed to allocate memory for node layer activation biases");
        exit(EXIT_FAILURE);
    }
}

float random_value(float minimum_value, float maximum_value)
{
    return ((maximum_value * 2.0f * rand()) / (float)RAND_MAX) - maximum_value;
}

float symmetric_random_value(float maximum_value)
{
    return random_value(-maximum_value, maximum_value);
}

void propogate_random_node_layer_weights(struct Node_Layer *node_layer, float multiplier)
{
    for (unsigned long long weight_index = 0; weight_index < node_layer->input_count * node_layer->output_count; weight_index++)
    {
        //node_layer->input_weights[weight_index] = multiplier * symmetric_random_value(1.0) * symmetric_random_value(1.0);
        node_layer->input_weights[weight_index] = multiplier * symmetric_random_value(sqrt(6.0f / (node_layer->input_count + sqrt(node_layer->output_count))));
    }
}

void propogate_random_node_layer_biases(struct Node_Layer *node_layer, float multiplier)
{
    for (unsigned long long bias_index = 0; bias_index < node_layer->output_count; bias_index++)
    {
        node_layer->activation_biases[bias_index] = multiplier * symmetric_random_value(1.0f) * symmetric_random_value(1.0f);
    }
}