struct Gradient_Descent_Derivatives
{
    float **input_weight_derivatives;
    float **activation_bias_derivatives;
};

void allocate_gradient_descent_derivatives(struct Node_Network *node_network, struct Gradient_Descent_Derivatives *gradient_descent_derivatives)
{
    float ***input_weight_derivatives = &(gradient_descent_derivatives->input_weight_derivatives);
    float ***activation_bias_derivatives = &(gradient_descent_derivatives->activation_bias_derivatives);

    *input_weight_derivatives = (float **)malloc(sizeof(float *) * node_network->node_layer_count);
    if (*input_weight_derivatives == NULL)
    {
        printf("Failed to allocate memory for gradient descent input weight derivatives\n");
        exit(EXIT_FAILURE);
    }

    *activation_bias_derivatives = (float **)malloc(sizeof(float *) * node_network->node_layer_count);
    if (*activation_bias_derivatives == NULL)
    {
        printf("Failed to allocate memory for gradient descent activation bias derivatives\n");
        exit(EXIT_FAILURE);
    }

    for (int node_layer_index = 0; node_layer_index < node_network->node_layer_count; node_layer_index++)
    {
        struct Node_Layer *node_layer = &(node_network->node_layers[node_layer_index]);

        (*input_weight_derivatives)[node_layer_index] = (float *)malloc(sizeof(float) * node_layer->input_count * node_layer->output_count);
        if ((*input_weight_derivatives)[node_layer_index] == NULL)
        {
            printf("Failed to allocate memory for gradient descent input weight derivatives for node layer %i\n", node_layer_index);
            exit(EXIT_FAILURE);
        }

        (*activation_bias_derivatives)[node_layer_index] = (float *)malloc(sizeof(float) * node_layer->output_count);
        if ((*activation_bias_derivatives)[node_layer_index] == NULL)
        {
            printf("Failed to allocate memory for gradient descent activation bias derivatives for node layer %i\n", node_layer_index);
            exit(EXIT_FAILURE);
        }
    }
}

void reset_gradient_descent_derivatives(struct Node_Network *node_network, struct Gradient_Descent_Derivatives *gradient_descent_derivatives)
{
    for (uint8_t node_layer_index = 0; node_layer_index < node_network->node_layer_count; node_layer_index++)
    {
        struct Node_Layer *node_layer = &(node_network->node_layers[node_layer_index]);

        for (uint64_t output_index = 0; output_index < node_layer->output_count; output_index++)
        {
            gradient_descent_derivatives->activation_bias_derivatives[node_layer_index][output_index] = 0.0f;

            for (uint64_t input_index = 0; input_index < node_layer->input_count; input_index++)
            {
                uint64_t weight_index = (output_index * node_layer->input_count) + input_index;

                gradient_descent_derivatives->input_weight_derivatives[node_layer_index][weight_index] = 0.0f;
            }
        }
    }
}

void multiply_gradient_descent_bias_and_weight_derivatives(struct Node_Network *node_network, struct Gradient_Descent_Derivatives *gradient_descent_derivatives, float factor)
{
    for (uint8_t node_layer_index = 0; node_layer_index < node_network->node_layer_count; node_layer_index++)
    {
        struct Node_Layer *node_layer = &(node_network->node_layers[node_layer_index]);

        for (uint64_t output_index = 0; output_index < node_layer->output_count; output_index++)
        {
            gradient_descent_derivatives->activation_bias_derivatives[node_layer_index][output_index] *= factor;

            for (uint64_t input_index = 0; input_index < node_layer->input_count; input_index++)
            {
                uint64_t weight_index = (output_index * node_layer->input_count) + input_index;

                gradient_descent_derivatives->input_weight_derivatives[node_layer_index][weight_index] *= factor;
            }
        }
    }
}

void divide_gradient_descent_bias_and_weight_derivatives(struct Node_Network *node_network, struct Gradient_Descent_Derivatives *gradient_descent_derivatives, float divisor)
{
    multiply_gradient_descent_bias_and_weight_derivatives(node_network, gradient_descent_derivatives, 1.0f / divisor);
}

void deallocate_gradient_descent_derivatives(struct Node_Network *node_network, struct Gradient_Descent_Derivatives *gradient_descent_derivatives)
{
    float ***input_weight_derivatives = &(gradient_descent_derivatives->input_weight_derivatives);
    float ***activation_bias_derivatives = &(gradient_descent_derivatives->activation_bias_derivatives);
    
    for (uint8_t node_layer_index = 0; node_layer_index < node_network->node_layer_count; node_layer_index++)
    {
        free((*input_weight_derivatives)[node_layer_index]);
        free((*activation_bias_derivatives)[node_layer_index]);
    }

    free(*input_weight_derivatives);
    free(*activation_bias_derivatives);
}