struct Node_Layer
{
    size_t input_count;
    size_t output_count;

    double *inputs;
    double *input_weights;
    double *activation_biases;
    double *raw_linear_outputs;
    double *outputs;
};

void initialize_node_layer(struct Node_Layer *node_layer, size_t input_count, size_t output_count)
{
    node_layer->input_count = input_count;
    node_layer->output_count = output_count;

    node_layer->input_weights = (double *)malloc(sizeof(double) * input_count * output_count);
    if (node_layer->input_weights == NULL)
    {
        printf("Failed to allocate memory for node layer input weights");
        exit(EXIT_FAILURE);
    }

    node_layer->activation_biases = (double *)malloc(sizeof(double) * output_count);
    if (node_layer->activation_biases == NULL)
    {
        printf("Failed to allocate memory for node layer activation biases");
        exit(EXIT_FAILURE);
    }

    node_layer->raw_linear_outputs = (double *)malloc(sizeof(double) * output_count);
    if (node_layer->raw_linear_outputs == NULL)
    {
        printf("Failed to allocate memory for node layer raw linear outputs");
        exit(EXIT_FAILURE);
    }

    node_layer->outputs = (double *)malloc(sizeof(double) * output_count);
    if (node_layer->input_weights == NULL)
    {
        printf("Failed to allocate memory for node layer output");
        exit(EXIT_FAILURE);
    }
}

double random_value(double minimum_value, double maximum_value)
{
    return ((maximum_value * 2.0 * rand()) / (double)RAND_MAX) - maximum_value;
}

double symmetric_random_value(double maximum_value)
{
    return random_value(-maximum_value, maximum_value);
}

void propogate_random_node_layer_weights(struct Node_Layer *node_layer, double multiplier)
{
    for (size_t weight_index = 0; weight_index < node_layer->input_count * node_layer->output_count; weight_index++)
    {
        //node_layer->input_weights[weight_index] = multiplier * symmetric_random_value(1.0) * symmetric_random_value(1.0);
        node_layer->input_weights[weight_index] = multiplier * symmetric_random_value(sqrt(6.0 / (node_layer->input_count + sqrt(node_layer->output_count))));
    }
}

void propogate_random_node_layer_biases(struct Node_Layer *node_layer, double multiplier)
{
    for (size_t bias_index = 0; bias_index < node_layer->output_count; bias_index++)
    {
        node_layer->activation_biases[bias_index] = multiplier * symmetric_random_value(1.0) * symmetric_random_value(1.0);
    }
}