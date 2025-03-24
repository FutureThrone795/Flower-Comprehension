#include "stdio.h"
#include "stdint.h"
#include "time.h"
#define _USE_MATH_DEFINES
#include "math.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define DEBUG_SHOULD_SHOW_GRADIENT_BATCH_INDEX 0
#define DEBUG_SHOULD_SHOW_ACCURACY_ON_GRADIENT_DESCENT_COMPLETION 1
#define DEBUG_SHOULD_SAVE_NODE_NETWORK_IMAGE 0

#define IMAGE_SAVE_FREQUENCY 1
#define SHOULD_PROMPT_BEFORE_DESCENT_CYCLE 0

#define NODE_NETWORK_DATA_SAVE_FREQUENCY 1

#ifndef _WIN32
    #if SHOULD_PROMPT_BEFORE_DESCENT_CYCLE != 0
        #error Descent Cycle Prompt Only Works For Windows Machines
    #endif
#endif

#include "node/node_layer.h"
#include "node/node_network.h"
#include "node/target_image_utilities.h"

#include "gradient_descent/calculate_nodes.h"
#include "math_utilities/compare_data.h"
#include "math_utilities/matrix_math.h"

#include "gradient_descent/node_network_gradient_descent.h"

#include "file_utilities/file_utilities.h"

#define MAXIMUM_GRADIENT_DESCENT_CYCLES -1
//Set to -1 for infinite cycles

int main(int argc, char **argv)
{
    char node_network_data_file_name[100];
    if (argc == 1)
    {
        printf("File name for node network data not provided. Using default...\n");
        sprintf(node_network_data_file_name, "node_network_data_files/default.bin");
    }
    else
    {
        sprintf_s(node_network_data_file_name, 100, "node_network_data_files/%s.bin", argv[1]);
    }

    srand(time(0));

    const size_t IMAGE_HEIGHT = 150;
    const size_t IMAGE_WIDTH = 150;
    const size_t IMAGE_CHANNELS = 3;

    const size_t IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS; 
    const size_t BATCH_SIZE = 3;

    #define NODE_LAYER_COUNT 6
    const size_t FIRST_NODE_LAYER_INPUT_COUNT = IMAGE_SIZE;
    size_t node_layer_output_count[NODE_LAYER_COUNT] = { 1024, 512, 128, 512, 1024, IMAGE_SIZE };

    double **image_data = (double **)malloc(sizeof(double *) * BATCH_SIZE);
    for (int image_data_index = 0; image_data_index < BATCH_SIZE; image_data_index++)
    {
        image_data[image_data_index] = (double *)malloc(sizeof(double) * IMAGE_SIZE);
        if (image_data[image_data_index] == NULL)
        {
            printf("Failed to allocate memory for image batch data for image #%i\n", image_data_index);
            exit(EXIT_FAILURE);
        }
    }

    double *adjoined_progress_image_data = (double *)malloc((sizeof(double) * IMAGE_SIZE * 2 * BATCH_SIZE));
    if (adjoined_progress_image_data == NULL)
    {
        printf("Failed to allocate memory for adjoined progress image data\n");
        exit(EXIT_FAILURE);
    }

    struct Node_Network node_network;
    initialize_node_network(&node_network, *image_data, NODE_LAYER_COUNT, FIRST_NODE_LAYER_INPUT_COUNT, node_layer_output_count);

    int does_load_from_file_fail = load_node_network_data_from_file(node_network_data_file_name, &node_network, NODE_LAYER_COUNT, FIRST_NODE_LAYER_INPUT_COUNT, node_layer_output_count);
    if (does_load_from_file_fail)
    {
        randomize_node_network_weights_and_biases(&node_network, 1.0, 1.0);
    }

    struct Gradient_Descent_Derivatives gradient_descent_derivatives;
    allocate_gradient_descent_derivatives(&node_network, &gradient_descent_derivatives);

    char name[64];

    for (int i = 0; i < MAXIMUM_GRADIENT_DESCENT_CYCLES || MAXIMUM_GRADIENT_DESCENT_CYCLES == -1; i++)
    {
        srand(12);
        for (int image_data_index = 0; image_data_index < BATCH_SIZE; image_data_index++)
        {
            load_random_image(image_data[image_data_index], IMAGE_SIZE);
        }

        if (i % IMAGE_SAVE_FREQUENCY == 0)
        {
            for (int image_data_index = 0; image_data_index < BATCH_SIZE; image_data_index++)
            {
                node_network.node_layers[0].inputs = image_data[image_data_index];
                compute_node_network(&node_network);
                add_to_adjoined_image(adjoined_progress_image_data, image_data[image_data_index], node_network.node_layers[node_network.node_layer_count - 1].outputs, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, image_data_index * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS * 2);
                sprintf_s(name, 64, "node_networks/node_network_image_%i.png", image_data_index);
                if (DEBUG_SHOULD_SAVE_NODE_NETWORK_IMAGE)
                {
                    save_node_network_as_image(&node_network, name);
                }
            }

            sprintf_s(name, 60, "progress/progress_image_%i.png", i);
            save_image(name, IMAGE_WIDTH * 2, IMAGE_HEIGHT * BATCH_SIZE, IMAGE_CHANNELS, adjoined_progress_image_data);
            save_image("progress_image.png", IMAGE_WIDTH * 2, IMAGE_HEIGHT * BATCH_SIZE, IMAGE_CHANNELS, adjoined_progress_image_data);

            printf("Saved Progress Image\n");
        }

        if (i % NODE_NETWORK_DATA_SAVE_FREQUENCY == 0)
        {
            save_node_network_data_to_file(node_network_data_file_name, &node_network, NODE_LAYER_COUNT, FIRST_NODE_LAYER_INPUT_COUNT, node_layer_output_count);
        }

        node_network_gradient_descent(&node_network, &gradient_descent_derivatives, image_data, BATCH_SIZE);

        if (DEBUG_SHOULD_SHOW_ACCURACY_ON_GRADIENT_DESCENT_COMPLETION)
        {
            static double prev_aggregate_batch_accuracy = 0.0;
            double aggregate_batch_accuracy = calculate_aggregate_batch_data_difference_squared(&node_network, image_data, BATCH_SIZE, IMAGE_SIZE);

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

        if (SHOULD_PROMPT_BEFORE_DESCENT_CYCLE)
        {
            system("pause");
        }
    }

    compute_node_network(&node_network);
    save_image("final_image.png", 150, 150, 3, node_network.node_layers[node_network.node_layer_count - 1].outputs);

    deallocate_gradient_descent_derivatives(&node_network, &gradient_descent_derivatives);

    return 0;
}