////////////////////////////////////////////////////////// FILE SPECIFICATION
/*

MAIN HEADER 
|- char[] "FutureThrone795 Node_Network Data File"
|- unsigned long long version
|- unsigned long long cycle_index
|- char[] "https://github.com/FutureThrone795"
|- uint8_t node_layer_count
|- unsigned long long first_node_layer_input_count
|- unsigned long long[node_layer_count] node_layer_output_counts
|- char[] "End Header"
\- NODE_NETWORK_DATA

NODE_NETWORK_DATA
|- NODE_LAYER_DATA
|- NODE_LAYER_DATA
|- ...
\- NODE_LAYER_DATA [node_layer_count]

NODE_LAYER_DATA
|- float[node_layer_input_count * node_layer_output_count] input_weights
\- float[node_layer_output_count] activation_biases

*/

const char file_description[] = "FutureThrone795 Node_Network Data File";
const char github_link[] = "https://github.com/FutureThrone795";
const char end_header[] = "End Header";

#define VERSION 3

void save_aggregate_batch_accuracy(const char *file_name, unsigned long long cycle_index, float aggregate_batch_accuracy)
{
    FILE *file_pointer;
    file_pointer = fopen(file_name, "a");
    
    if (cycle_index == 0)
    {
        fprintf(file_pointer, "Cycle Index\t\tAggregate Batch Accuracy");
    }

    fprintf(file_pointer, "\n%llu\t\t\t\t%f", cycle_index, aggregate_batch_accuracy);

    fclose(file_pointer);
}

unsigned long long calculate_node_network_data_float_count(uint8_t node_layer_count, unsigned long long first_node_layer_input_count, unsigned long long *node_layer_output_counts)
{
    unsigned long long float_count = 0;

    float_count += first_node_layer_input_count * node_layer_output_counts[0];
    float_count += node_layer_output_counts[0];

    for (uint8_t node_layer_index = 1; node_layer_index < node_layer_count; node_layer_index++)
    {
        float_count += node_layer_output_counts[node_layer_index - 1] * node_layer_output_counts[node_layer_index];
        float_count += node_layer_output_counts[node_layer_index];
    }

    return float_count;
}

void seek_and_read_file_header_also_allocate_node_layer_input_counts_and_brew_me_coffee(FILE *file_pointer, unsigned long long *version_dest, unsigned long long *cycle_index_dest, uint8_t *node_layer_count_dest, unsigned long long *first_node_layer_input_count_dest, unsigned long long **node_layer_output_counts_dest)
{
    fseeko(file_pointer, sizeof(file_description), SEEK_CUR);

    fread(version_dest, sizeof(unsigned long long), 1, file_pointer);
    fread(cycle_index_dest, sizeof(unsigned long long), 1, file_pointer);

    fseeko(file_pointer, sizeof(github_link), SEEK_CUR);

    fread(node_layer_count_dest, sizeof(uint8_t), 1, file_pointer);
    fread(first_node_layer_input_count_dest, sizeof(unsigned long long), 1, file_pointer);

    (*node_layer_output_counts_dest) = (unsigned long long *)malloc(sizeof(unsigned long long) * (*node_layer_count_dest));
    if (node_layer_output_counts_dest == NULL)
    {
        printf("Failed to allocate memory for node layer output counts\n");
        fclose(file_pointer);
        exit(EXIT_FAILURE);
    }

    fread((*node_layer_output_counts_dest), sizeof(unsigned long long), *node_layer_count_dest, file_pointer);

    fseeko(file_pointer, sizeof(end_header), SEEK_CUR);
}

void write_file_header(FILE *file_pointer, unsigned long long version, unsigned long long cycle_index, uint8_t node_layer_count, unsigned long long first_node_layer_input_count, unsigned long long *node_layer_output_counts)
{
    fwrite(file_description, sizeof(file_description), 1, file_pointer);

    fwrite(&version, sizeof(unsigned long long), 1, file_pointer);
    fwrite(&cycle_index, sizeof(unsigned long long), 1, file_pointer);

    fwrite(github_link, sizeof(github_link), 1, file_pointer);

    fwrite(&node_layer_count, sizeof(uint8_t), 1, file_pointer);
    fwrite(&first_node_layer_input_count, sizeof(unsigned long long), 1, file_pointer);
    fwrite(node_layer_output_counts, sizeof(unsigned long long), node_layer_count, file_pointer);

    fwrite(end_header, sizeof(end_header), 1, file_pointer);
}

int load_node_network_data_from_file(const char *file_name, struct Node_Network *node_network, unsigned long long *cycle_index, uint8_t node_layer_count, unsigned long long first_node_layer_input_count, unsigned long long *node_layer_output_counts)
{
    FILE *file_pointer;
    file_pointer = fopen(file_name, "rb");
    if (file_pointer == NULL)
    {
        printf("File named %s does not exist. Continuing with randomly-generated node values\n", file_name);
        return 1;
    }

    uint8_t file_node_layer_count;
    unsigned long long file_version, file_first_node_layer_input_count;
    unsigned long long *file_node_layer_output_counts = 0;

    seek_and_read_file_header_also_allocate_node_layer_input_counts_and_brew_me_coffee(file_pointer, &file_version, cycle_index, &file_node_layer_count, &file_first_node_layer_input_count, &file_node_layer_output_counts);
    printf("Loaded header data from %s:\nVersion: %llu, Node layer count: %u\n", file_name, file_version, (unsigned)file_node_layer_count);

    if (file_version != VERSION)
    {
        printf("Node network data file is from a deprecated version\n");
        printf("Current version: %i\n", VERSION);
        fclose(file_pointer);
        exit(EXIT_FAILURE);
    }

    if (file_node_layer_count != node_layer_count || file_first_node_layer_input_count != first_node_layer_input_count)
    {
        printf("Node network from file and destination node network do not conform\n");
        fclose(file_pointer);
        exit(EXIT_FAILURE);
    }

    int do_output_counts_conform = 1;

    printf("Node layer output counts:\n");
    for (unsigned long long file_node_layer_output_count_index = 0; file_node_layer_output_count_index < file_node_layer_count; file_node_layer_output_count_index++)
    {
        printf("%llu ", file_node_layer_output_counts[file_node_layer_output_count_index]);
        if (file_node_layer_output_counts[file_node_layer_output_count_index] != node_layer_output_counts[file_node_layer_output_count_index])
        {
            printf("-> Does not conform with destination node network output of %llu at index %llu\n", node_layer_output_counts[file_node_layer_output_count_index], file_node_layer_output_count_index);
            do_output_counts_conform = 0;
        }
    }
    printf("\n");

    if (!do_output_counts_conform)
    {
        printf("Node network output counts from file and destination node network do not conform\n");
        fclose(file_pointer);
        exit(EXIT_FAILURE);
    }
    
    free(file_node_layer_output_counts);

    unsigned long long node_network_data_float_count = calculate_node_network_data_float_count(node_layer_count, first_node_layer_input_count, node_layer_output_counts);
    float *file_node_network_data_buffer = (float *)malloc(sizeof(float) * node_network_data_float_count);
    if (file_node_network_data_buffer == NULL)
    {
        printf("Failed to allocate memory for file node network data buffer");
        fclose(file_pointer);
        exit(EXIT_FAILURE);
    }

    fread(file_node_network_data_buffer, sizeof(float), node_network_data_float_count, file_pointer);

    unsigned long long buffer_cursor = 0;
    for (uint8_t node_layer_index = 0; node_layer_index < node_layer_count; node_layer_index++)
    {
        struct Node_Layer *node_layer = &(node_network->node_layers[node_layer_index]);

        for (unsigned long long input_weight_index = 0; input_weight_index < node_layer->output_count * node_layer->input_count; input_weight_index++)
        {
            node_layer->input_weights[input_weight_index] = file_node_network_data_buffer[buffer_cursor];
            buffer_cursor++;
        }

        for (unsigned long long activation_bias_index = 0; activation_bias_index < node_layer->output_count; activation_bias_index++)
        {
            node_layer->activation_biases[activation_bias_index] = file_node_network_data_buffer[buffer_cursor];
            buffer_cursor++;
        }
    }

    printf("Loaded node network data from %s\n", file_name);

    free(file_node_network_data_buffer);
    fclose(file_pointer);
    return 0;
}

void save_node_network_data_to_file(const char *file_name, struct Node_Network *node_network, unsigned long long cycle_index, uint8_t node_layer_count, unsigned long long first_node_layer_input_count, unsigned long long *node_layer_output_counts)
{
    FILE *file_pointer;
    file_pointer = fopen(file_name, "wb");

    unsigned long long node_network_data_float_count = calculate_node_network_data_float_count(node_layer_count, first_node_layer_input_count, node_layer_output_counts);
    float *file_node_network_data_buffer = (float *)malloc(sizeof(float) * node_network_data_float_count);
    if (file_node_network_data_buffer == NULL)
    {
        printf("Failed to allocate memory for file node network data buffer\n");
        fclose(file_pointer);
        exit(EXIT_FAILURE);
    }

    write_file_header(file_pointer, VERSION, cycle_index, node_layer_count, first_node_layer_input_count, node_layer_output_counts);
    
    unsigned long long buffer_cursor = 0;
    for (uint8_t node_layer_index = 0; node_layer_index < node_layer_count; node_layer_index++)
    {
        struct Node_Layer *node_layer = &(node_network->node_layers[node_layer_index]);

        for (unsigned long long input_weight_index = 0; input_weight_index < node_layer->output_count * node_layer->input_count; input_weight_index++)
        {
            file_node_network_data_buffer[buffer_cursor] = node_layer->input_weights[input_weight_index];
            buffer_cursor++;
        }

        for (unsigned long long activation_bias_index = 0; activation_bias_index < node_layer->output_count; activation_bias_index++)
        {
            file_node_network_data_buffer[buffer_cursor] = node_layer->activation_biases[activation_bias_index];
            buffer_cursor++;
        }
    }

    fwrite(file_node_network_data_buffer, sizeof(float), node_network_data_float_count, file_pointer);

    printf("Saved node network data to %s\n", file_name);
    free(file_node_network_data_buffer);
    fclose(file_pointer);
}