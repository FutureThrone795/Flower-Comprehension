struct Gradient_Descent_Derivatives
{
    double **input_weight_derivatives;
    double **output_derivatives;
    double **activation_bias_derivatives;
};

void allocate_gradient_descent_derivatives(struct Node_Network *node_network, struct Gradient_Descent_Derivatives *gradient_descent_derivatives)
{
    double ***input_weight_derivatives = &(gradient_descent_derivatives->input_weight_derivatives);
    double ***output_derivatives = &(gradient_descent_derivatives->output_derivatives);
    double ***activation_bias_derivatives = &(gradient_descent_derivatives->activation_bias_derivatives);

    *input_weight_derivatives = (double **)malloc(sizeof(double *) * node_network->node_layer_count);
    if (*input_weight_derivatives == NULL)
    {
        printf("Failed to allocate memory for gradient descent input weight derivatives\n");
        exit(EXIT_FAILURE);
    }

    *output_derivatives = (double **)malloc(sizeof(double *) * node_network->node_layer_count);
    if (*output_derivatives == NULL)
    {
        printf("Failed to allocate memory for gradient descent activation output derivatives\n");
        exit(EXIT_FAILURE);
    }

    *activation_bias_derivatives = (double **)malloc(sizeof(double *) * node_network->node_layer_count);
    if (*activation_bias_derivatives == NULL)
    {
        printf("Failed to allocate memory for gradient descent activation bias derivatives\n");
        exit(EXIT_FAILURE);
    }

    for (int node_layer_index = 0; node_layer_index < node_network->node_layer_count; node_layer_index++)
    {
        struct Node_Layer *node_layer = &(node_network->node_layers[node_layer_index]);

        (*input_weight_derivatives)[node_layer_index] = (double *)malloc(sizeof(double) * node_layer->input_count * node_layer->output_count);
        if ((*input_weight_derivatives)[node_layer_index] == NULL)
        {
            printf("Failed to allocate memory for gradient descent input weight derivatives for node layer %i\n", node_layer_index);
            exit(EXIT_FAILURE);
        }

        (*output_derivatives)[node_layer_index] = (double *)malloc(sizeof(double) * node_layer->output_count);
        if ((*output_derivatives)[node_layer_index] == NULL)
        {
            printf("Failed to allocate memory for gradient descent output derivatives for node layer %i\n", node_layer_index);
            exit(EXIT_FAILURE);
        }

        (*activation_bias_derivatives)[node_layer_index] = (double *)malloc(sizeof(double) * node_layer->output_count);
        if ((*activation_bias_derivatives)[node_layer_index] == NULL)
        {
            printf("Failed to allocate memory for gradient descent activation bias derivatives for node layer %i\n", node_layer_index);
            exit(EXIT_FAILURE);
        }
    }
}

void reset_gradient_descent_derivatives(struct Node_Network *node_network, struct Gradient_Descent_Derivatives *gradient_descent_derivatives)
{
    for (size_t node_layer_index = 0; node_layer_index < node_network->node_layer_count; node_layer_index++)
    {
        struct Node_Layer *node_layer = &(node_network->node_layers[node_layer_index]);

        for (size_t output_index = 0; output_index < node_layer->output_count; output_index++)
        {
            gradient_descent_derivatives->output_derivatives[node_layer_index][output_index] = 0.0;
            gradient_descent_derivatives->activation_bias_derivatives[node_layer_index][output_index] = 0.0;

            for (size_t input_index = 0; input_index < node_layer->input_count; input_index++)
            {
                size_t weight_index = (output_index * node_layer->input_count) + input_index;

                gradient_descent_derivatives->input_weight_derivatives[node_layer_index][weight_index] = 0.0;
            }
        }
    }
}

void multiply_gradient_descent_bias_and_weight_derivatives(struct Node_Network *node_network, struct Gradient_Descent_Derivatives *gradient_descent_derivatives, double factor)
{
    for (size_t node_layer_index = 0; node_layer_index < node_network->node_layer_count; node_layer_index++)
    {
        struct Node_Layer *node_layer = &(node_network->node_layers[node_layer_index]);

        for (size_t output_index = 0; output_index < node_layer->output_count; output_index++)
        {
            gradient_descent_derivatives->activation_bias_derivatives[node_layer_index][output_index] *= factor;

            for (size_t input_index = 0; input_index < node_layer->input_count; input_index++)
            {
                size_t weight_index = (output_index * node_layer->input_count) + input_index;

                gradient_descent_derivatives->input_weight_derivatives[node_layer_index][weight_index] *= factor;
            }
        }
    }
}

void divide_gradient_descent_bias_and_weight_derivatives(struct Node_Network *node_network, struct Gradient_Descent_Derivatives *gradient_descent_derivatives, double divisor)
{
    multiply_gradient_descent_bias_and_weight_derivatives(node_network, gradient_descent_derivatives, 1.0 / divisor);
}

void deallocate_gradient_descent_derivatives(struct Node_Network *node_network, struct Gradient_Descent_Derivatives *gradient_descent_derivatives)
{
    double ***input_weight_derivatives = &(gradient_descent_derivatives->input_weight_derivatives);
    double ***output_derivatives = &(gradient_descent_derivatives->output_derivatives);
    double ***activation_bias_derivatives = &(gradient_descent_derivatives->activation_bias_derivatives);
    
    for (size_t node_layer_index = 0; node_layer_index < node_network->node_layer_count; node_layer_index++)
    {
        free((*input_weight_derivatives)[node_layer_index]);
        free((*output_derivatives)[node_layer_index]);
        free((*activation_bias_derivatives)[node_layer_index]);
    }

    free(*input_weight_derivatives);
    free(*output_derivatives);
    free(*activation_bias_derivatives);
}