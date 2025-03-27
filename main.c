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
#include "node/calculate_nodes_data_partition.h"

#include "node/target_image_utilities.h"
#include "node/calculate_nodes.h"

#include "math_utilities/compare_data.h"
#include "math_utilities/matrix_math.h"

#include "gradient_descent/node_network_gradient_descent.h"
#include "gradient_descent/gradient_descent_debug.h"

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
    const uint8_t BATCH_SIZE = 25;

    #define NODE_LAYER_COUNT 6
    const size_t FIRST_NODE_LAYER_INPUT_COUNT = IMAGE_SIZE;
    size_t node_layer_output_count[NODE_LAYER_COUNT] = { 1024, 512, 128, 512, 1024, IMAGE_SIZE };

    float **image_data = (float **)malloc(sizeof(float *) * BATCH_SIZE);
    for (int image_data_index = 0; image_data_index < BATCH_SIZE; image_data_index++)
    {
        image_data[image_data_index] = (float *)malloc(sizeof(float) * IMAGE_SIZE);
        if (image_data[image_data_index] == NULL)
        {
            printf("Failed to allocate memory for image batch data for image #%i\n", image_data_index);
            exit(EXIT_FAILURE);
        }
    }

    float *adjoined_progress_image_data = (float *)malloc((sizeof(float) * IMAGE_SIZE * 2 * BATCH_SIZE));
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

    struct Node_Network_Data_Partition node_network_data_partition;
    allocate_node_network_data_partition(&node_network, &node_network_data_partition, 1);

    for (int cycle_index = 0; cycle_index < MAXIMUM_GRADIENT_DESCENT_CYCLES || MAXIMUM_GRADIENT_DESCENT_CYCLES == -1; cycle_index++)
    {
        for (int image_data_index = 0; image_data_index < BATCH_SIZE; image_data_index++)
        {
            load_random_image(image_data[image_data_index], IMAGE_SIZE);
        }

        if (cycle_index % IMAGE_SAVE_FREQUENCY == 0)
        {
            save_progress_image(adjoined_progress_image_data, &node_network, image_data, BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, cycle_index);
        }

        if (cycle_index % NODE_NETWORK_DATA_SAVE_FREQUENCY == 0)
        {
            save_node_network_data_to_file(node_network_data_file_name, &node_network, NODE_LAYER_COUNT, FIRST_NODE_LAYER_INPUT_COUNT, node_layer_output_count);
        }

        node_network_gradient_descent(&node_network, &gradient_descent_derivatives, image_data, BATCH_SIZE);

        if (DEBUG_SHOULD_SHOW_ACCURACY_ON_GRADIENT_DESCENT_COMPLETION)
        {
            print_aggregate_batch_accuracy(&node_network, image_data, BATCH_SIZE, IMAGE_SIZE);
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