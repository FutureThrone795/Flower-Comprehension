#include "gradient_descent_derivative.h"

void node_layer_derivatives(struct Node_Network *node_network, struct Node_Network_Data_Partition *node_network_data_partition, struct Gradient_Descent_Derivatives *gradient_descent_derivatives, unsigned long long node_layer_index, float* target_output)
{
    struct Node_Layer *node_layer = &(node_network->node_layers[node_layer_index]);
    struct Node_Layer_Data_Partition *node_layer_data_partition = &(node_network_data_partition->node_layer_data_partitions[node_layer_index]);

    float **input_weight_derivatives = gradient_descent_derivatives->input_weight_derivatives;
    float **activation_bias_derivatives = gradient_descent_derivatives->activation_bias_derivatives;

    int is_output_layer = node_layer_index == (node_network->node_layer_count - 1);

    for (unsigned long long output_index = 0; output_index < node_layer->output_count; output_index++)
    {
        float current_output_derivative;
        if (node_layer_index == (node_network->node_layer_count - 1))
        {
            current_output_derivative = 2.0f * (node_layer_data_partition->outputs[output_index] - target_output[output_index]);
        }
        else
        {
            struct Node_Layer *next_node_layer = &(node_network->node_layers[node_layer_index + 1]);
            float total_weight = 0.0f;

            for (unsigned long long next_node_layer_output_index = 0; next_node_layer_output_index < next_node_layer->output_count; next_node_layer_output_index++)
            {
                struct Node_Layer_Data_Partition *next_node_layer_data_partition = &(node_network_data_partition->node_layer_data_partitions[node_layer_index + 1]);
                total_weight += (next_node_layer->input_weights[(next_node_layer_output_index * next_node_layer->input_count) + output_index] * next_node_layer_data_partition->output_derivatives[next_node_layer_output_index] * activation_function_derivative(next_node_layer_data_partition->raw_linear_outputs[next_node_layer_output_index], 0)); //FIX IS_OUTPUT_LAYER
            }

            total_weight /= next_node_layer->output_count;
            current_output_derivative = total_weight;
        }
        node_layer_data_partition->output_derivatives[output_index] = current_output_derivative;

        float current_activation_bias_derivative;
        current_activation_bias_derivative = activation_function_derivative(node_layer_data_partition->raw_linear_outputs[output_index], is_output_layer) * current_output_derivative;

        activation_bias_derivatives[node_layer_index][output_index] = current_activation_bias_derivative;

        for (unsigned long long input_index = 0; input_index < node_layer->input_count; input_index++)
        {
            float current_input_weight_derivative;
            current_input_weight_derivative = activation_function_derivative(node_layer_data_partition->raw_linear_outputs[output_index], is_output_layer) * node_layer_data_partition->inputs[input_index] * current_output_derivative;

            input_weight_derivatives[node_layer_index][(output_index * node_layer->input_count) + input_index] = current_input_weight_derivative;
        }
    }
}

void node_network_derivatives(struct Node_Network *node_network, struct Node_Network_Data_Partition *node_network_data_partition, struct Gradient_Descent_Derivatives *gradient_descent_derivatives, float* target_output)
{
    for (uint8_t node_layer_index = 0; node_layer_index < node_network->node_layer_count; node_layer_index++)
    {
        uint8_t actual_node_layer_index = (node_network->node_layer_count - 1) - node_layer_index;
        node_layer_derivatives(node_network, node_network_data_partition, gradient_descent_derivatives, actual_node_layer_index, target_output);
    }
}