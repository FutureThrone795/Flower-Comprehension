#include "stdio.h"
#include "stdint.h"
#include "time.h"
#define _USE_MATH_DEFINES
#include "math.h"
#include "pthread.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "math_utilities/matrix_math.h"

#include "node/node_layer.h"
#include "node/node_network.h"
#include "node/calculate_nodes_data_partition.h"

#include "node/target_image_utilities.h"
#include "node/calculate_nodes.h"

#include "math_utilities/compare_data.h"

#define DEBUG_SHOULD_SHOW_GRADIENT_BATCH_INDEX 0
#define THREAD_COUNT 128

#include "gradient_descent/node_network_gradient_descent.h"
#include "gradient_descent/gradient_descent_handler.h"
#include "gradient_descent/gradient_descent_debug.h"

#include "file_utilities/file_utilities.h"

#define DEBUG_SHOULD_SHOW_ACCURACY_ON_GRADIENT_DESCENT_COMPLETION 1
#define DEBUG_SHOULD_SAVE_NODE_NETWORK_IMAGE 0

#define SHOULD_PRINT_AGGREGATE_BATCH_ACCURACY 0

#define DESCENT_CYCLE_COMPLETION_PRINT_FREQUENCY 25

#define IMAGE_SAVE_FREQUENCY 0
#define SHOULD_PROMPT_BEFORE_DESCENT_CYCLE 0

#define NODE_NETWORK_DATA_SAVE_FREQUENCY 25

#ifndef _WIN32
    #if SHOULD_PROMPT_BEFORE_DESCENT_CYCLE != 0
        #error Descent Cycle Prompt Only Works For Windows Machines
    #endif
#endif

#define MAXIMUM_GRADIENT_DESCENT_CYCLES -1
//Set to -1 for infinite cycles

#define BATCH_SIZE 128
#define NODE_LAYER_COUNT 6

int main(int argc, char **argv)
{
    char node_network_data_file_name[100];
    char aggregate_batch_accuracy_tracking_file_name[100];
    if (argc == 1)
    {
        printf("File name for node network data not provided. Using default...\n");
        sprintf(node_network_data_file_name, "node_network_data_files/default.bin");
        sprintf(aggregate_batch_accuracy_tracking_file_name, "aggregate_batch_accuracy_tracking_files/default.txt");
    }
    else
    {
        snprintf(node_network_data_file_name, 100, "node_network_data_files/%s.bin", argv[1]);
        snprintf(aggregate_batch_accuracy_tracking_file_name, 100, "aggregate_batch_accuracy_tracking_files/%s.txt", argv[1]);
    }

    srand(time(0));

    const unsigned long long IMAGE_HEIGHT = 150;
    const unsigned long long IMAGE_WIDTH = 150;
    const unsigned long long IMAGE_CHANNELS = 3;

    const unsigned long long IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS; 

    const unsigned long long FIRST_NODE_LAYER_INPUT_COUNT = IMAGE_SIZE;
    unsigned long long node_layer_output_count[NODE_LAYER_COUNT] = { 1024, 512, 256, 512, 1024, IMAGE_SIZE };

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

    unsigned long long cycle_index;
    int does_load_from_file_fail = load_node_network_data_from_file(node_network_data_file_name, &node_network, &cycle_index, NODE_LAYER_COUNT, FIRST_NODE_LAYER_INPUT_COUNT, node_layer_output_count);
    if (does_load_from_file_fail)
    {
        cycle_index = 0;
        randomize_node_network_weights_and_biases(&node_network, 1.0, 1.0);
    }

    struct Gradient_Descent_Derivatives *gradient_descent_derivatives;
    struct Node_Network_Data_Partition *node_network_data_partition;
    allocate_and_initialize_gradient_descent_values(&node_network, &gradient_descent_derivatives, &node_network_data_partition);

    struct Node_Network_Data_Partition misc_node_network_data_partition; //For calculating differences, saving images, etc. 
    allocate_node_network_data_partition(&node_network, &misc_node_network_data_partition, 0); //Does not support gradient descent
    initialize_node_network_data_partition(&node_network, &misc_node_network_data_partition);

    float aggregate_batch_accuracy;

    for (; cycle_index < MAXIMUM_GRADIENT_DESCENT_CYCLES || MAXIMUM_GRADIENT_DESCENT_CYCLES == -1; cycle_index++)
    {
        for (int image_data_index = 0; image_data_index < BATCH_SIZE; image_data_index++)
        {
            load_random_image(image_data[image_data_index], IMAGE_SIZE);
        }

        if (IMAGE_SAVE_FREQUENCY != 0 && cycle_index % IMAGE_SAVE_FREQUENCY == 0)
        {
            save_progress_image(adjoined_progress_image_data, &node_network, &misc_node_network_data_partition, image_data, BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, cycle_index);
        }

        if (cycle_index % NODE_NETWORK_DATA_SAVE_FREQUENCY == 0)
        {
            save_node_network_data_to_file(node_network_data_file_name, &node_network, cycle_index, NODE_LAYER_COUNT, FIRST_NODE_LAYER_INPUT_COUNT, node_layer_output_count);
        }

        gradient_descent_cycle(&node_network, gradient_descent_derivatives, node_network_data_partition, image_data, BATCH_SIZE, &aggregate_batch_accuracy);

        if (SHOULD_PRINT_AGGREGATE_BATCH_ACCURACY)
        {
            print_aggregate_batch_accuracy(aggregate_batch_accuracy);
        }

        save_aggregate_batch_accuracy(aggregate_batch_accuracy_tracking_file_name, cycle_index, aggregate_batch_accuracy);

        if (cycle_index % DESCENT_CYCLE_COMPLETION_PRINT_FREQUENCY == 0)
        {
            printf("Gradient descent cycle index %llu completed\n", cycle_index);
        }

        if (SHOULD_PROMPT_BEFORE_DESCENT_CYCLE)
        {
            system("pause");
        }
    }

    deallocate_gradient_descent_derivatives(&node_network, gradient_descent_derivatives);

    pthread_exit(NULL);
    return 0;
}