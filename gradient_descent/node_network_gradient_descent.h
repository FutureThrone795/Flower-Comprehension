#include "gradient_descent_derivative.h"

#define DESCENT_DERIVATIVE_BIAS_STRENGTH_FACTOR 0.03
#define DESCENT_DERIVATIVE_WEIGHT_STRENGTH_FACTOR 0.03

#define SHOULD_NORMALIZE_NODE_LAYER_BIAS_STEP_SIZE 0
#define SHOULD_NORMALIZE_NODE_LAYER_WEIGHT_STEP_SIZE 0

#define SHOULD_DIVIDE_NODE_LAYER_BY_DEPTH 1

void node_layer_derivatives(struct Gradient_Descent_Derivatives *gradient_descent_derivatives, struct Node_Network *node_network, size_t node_layer_index, double* target_output, size_t batch_index)
{
    struct Node_Layer *node_layer = &(node_network->node_layers[node_layer_index]);

    double **input_weight_derivatives = gradient_descent_derivatives->input_weight_derivatives;
    double **output_derivatives = gradient_descent_derivatives->output_derivatives;
    double **activation_bias_derivatives = gradient_descent_derivatives->activation_bias_derivatives;

    int is_output_layer = node_layer_index == (node_network->node_layer_count - 1);

    for (size_t output_index = 0; output_index < node_layer->output_count; output_index++)
    {
        double current_output_derivative;
        if (node_layer_index == (node_network->node_layer_count - 1))
        {
            current_output_derivative = 2.0 * (node_layer->outputs[output_index] - target_output[output_index]);
        }
        else
        {
            struct Node_Layer *next_node_layer = &(node_network->node_layers[node_layer_index + 1]);
            double total_weight = 0.0;

            for (size_t next_node_layer_output_index = 0; next_node_layer_output_index < next_node_layer->output_count; next_node_layer_output_index++)
            {
                total_weight += (next_node_layer->input_weights[(next_node_layer_output_index * next_node_layer->input_count) + output_index] * output_derivatives[node_layer_index + 1][next_node_layer_output_index] * activation_function_derivative(next_node_layer->raw_linear_outputs[next_node_layer_output_index], 0)); //FIX IS_OUTPUT_LAYER
            }

            total_weight /= next_node_layer->output_count;
            current_output_derivative = total_weight;
        }
        output_derivatives[node_layer_index][output_index] = current_output_derivative;

        double current_activation_bias_derivative;
        current_activation_bias_derivative = activation_function_derivative(node_layer->raw_linear_outputs[output_index], is_output_layer) * current_output_derivative;

        activation_bias_derivatives[node_layer_index][output_index] += current_activation_bias_derivative;

        for (size_t input_index = 0; input_index < node_layer->input_count; input_index++)
        {
            double current_input_weight_derivative;
            current_input_weight_derivative = activation_function_derivative(node_layer->raw_linear_outputs[output_index], is_output_layer) * node_layer->inputs[input_index] * current_output_derivative;

            input_weight_derivatives[node_layer_index][(output_index * node_layer->input_count) + input_index] += current_input_weight_derivative;
        }
    }
}

void node_network_gradient_descent(struct Node_Network *node_network, struct Gradient_Descent_Derivatives *gradient_descent_derivatives, double** target_batch_outputs, size_t target_batch_count)
{   
    reset_gradient_descent_derivatives(node_network, gradient_descent_derivatives);

    for (size_t batch_index = 0; batch_index < target_batch_count; batch_index++)
    {
        double* target_outputs = target_batch_outputs[batch_index];
        node_network->node_layers[0].inputs = target_outputs;
        compute_node_network(node_network);

        for (size_t node_layer_index = 0; node_layer_index < node_network->node_layer_count; node_layer_index++)
        {
            size_t actual_node_layer_index = (node_network->node_layer_count - 1) - node_layer_index;
            node_layer_derivatives(gradient_descent_derivatives, node_network, actual_node_layer_index, target_outputs, batch_index);
        }

        if (DEBUG_SHOULD_SHOW_GRADIENT_BATCH_INDEX)
        {
            printf("Gradient Descent Completed, Batch Index %llu/%llu (%.2f %%)\n", batch_index + 1, target_batch_count, 100.0 * ((double)batch_index + 1.0) / (double)target_batch_count);
        }
    }

    divide_gradient_descent_bias_and_weight_derivatives(node_network, gradient_descent_derivatives, (double)target_batch_count);

    double **input_weight_derivatives = gradient_descent_derivatives->input_weight_derivatives;
    double **output_derivatives = gradient_descent_derivatives->output_derivatives;
    double **activation_bias_derivatives = gradient_descent_derivatives->activation_bias_derivatives;

    for (size_t node_layer_index = 0; node_layer_index < node_network->node_layer_count; node_layer_index++)
    {
        struct Node_Layer *node_layer = &(node_network->node_layers[node_layer_index]);
        double layer_activation_bias_divisor = 1.0;
        double layer_weights_divisor = 1.0;

        if (SHOULD_NORMALIZE_NODE_LAYER_BIAS_STEP_SIZE || SHOULD_NORMALIZE_NODE_LAYER_WEIGHT_STEP_SIZE)
        {
            for (size_t pre_apply_output_index = 0; pre_apply_output_index < node_layer->output_count; pre_apply_output_index++)
            {
                if (SHOULD_NORMALIZE_NODE_LAYER_BIAS_STEP_SIZE)
                {
                    double abs_activation_bias_value = fabs(activation_bias_derivatives[node_layer_index][pre_apply_output_index]);
                    if (layer_activation_bias_divisor < abs_activation_bias_value)
                    {
                        layer_activation_bias_divisor = abs_activation_bias_value;
                    }
                }

                if (SHOULD_NORMALIZE_NODE_LAYER_WEIGHT_STEP_SIZE)
                {
                    for (size_t pre_apply_input_index = 0; pre_apply_input_index < node_layer->input_count; pre_apply_input_index++)
                    {
                        size_t weight_index = (pre_apply_output_index * node_layer->input_count) + pre_apply_input_index;

                        double abs_weight_value = fabs(input_weight_derivatives[node_layer_index][weight_index]);
                        if (layer_weights_divisor < abs_weight_value)
                        {
                            layer_weights_divisor = abs_weight_value;
                        }
                    }
                }
            }
            
            if (layer_activation_bias_divisor == 0.0)
            {
                layer_activation_bias_divisor = 1.0;
            }
            if (layer_weights_divisor == 0.0)
            {
                layer_weights_divisor = 1.0;
            }
        }

        if (SHOULD_DIVIDE_NODE_LAYER_BY_DEPTH)
        {
            layer_activation_bias_divisor *= (node_layer_index + 1);
            layer_weights_divisor *= (node_layer_index + 1);
        }

        for (size_t output_index = 0; output_index < node_layer->output_count; output_index++)
        {
            node_layer->activation_biases[output_index] -= (activation_bias_derivatives[node_layer_index][output_index] / layer_activation_bias_divisor) * DESCENT_DERIVATIVE_BIAS_STRENGTH_FACTOR;

            for (size_t input_index = 0; input_index < node_layer->input_count; input_index++)
            {
                size_t weight_index = (output_index * node_layer->input_count) + input_index;

                node_layer->input_weights[weight_index] -= (input_weight_derivatives[node_layer_index][weight_index] / layer_weights_divisor) * DESCENT_DERIVATIVE_WEIGHT_STRENGTH_FACTOR;
            }
        }
    }
}