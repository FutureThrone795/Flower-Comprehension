#define DESCENT_DERIVATIVE_BIAS_STRENGTH_FACTOR 0.005
#define DESCENT_DERIVATIVE_WEIGHT_STRENGTH_FACTOR 0.005

#define SHOULD_DIVIDE_NODE_LAYER_BY_DEPTH 0

void allocate_and_initialize_gradient_descent_values(struct Node_Network *node_network, struct Gradient_Descent_Derivatives **gradient_descent_derivatives, struct Node_Network_Data_Partition **node_network_data_partition)
{
    *gradient_descent_derivatives = (struct Gradient_Descent_Derivatives *)malloc(THREAD_COUNT * sizeof(struct Gradient_Descent_Derivatives));
    *node_network_data_partition = (struct Node_Network_Data_Partition *)malloc(THREAD_COUNT * sizeof(struct Node_Network_Data_Partition));

    for (uint8_t thread_index = 0; thread_index < THREAD_COUNT; thread_index++)
    {
        allocate_gradient_descent_derivatives(node_network, &((*gradient_descent_derivatives)[thread_index]));
        allocate_node_network_data_partition(node_network, &((*node_network_data_partition)[thread_index]), 1);
        initialize_node_network_data_partition(node_network, &((*node_network_data_partition)[thread_index]));
    }
}

struct Gradient_Descent_Master_Handler_Data
{
    float **target_batch_data;
    float *aggregate_batch_accuracy;
    unsigned long long batch_size;
    struct Gradient_Descent_Derivatives *master_gradient_descent_derivatives;
    struct Node_Network *node_network;
    uint8_t *has_computed_batch_indexes;

    pthread_mutex_t* master_gradient_descent_derivatives_mutex;
    pthread_mutex_t* has_computed_batch_index_mutex;
};

struct Gradient_Descent_Worker_Thread_Data
{
    struct Gradient_Descent_Master_Handler_Data* gradient_descent_master_handler_data;
    uint8_t thread_index;
    struct Gradient_Descent_Derivatives *gradient_descent_derivatives;
    struct Node_Network_Data_Partition *node_network_data_partition;
};

void initialize_master_handler_data(struct Gradient_Descent_Master_Handler_Data *gradient_descent_master_handler_data, struct Node_Network *node_network, struct Gradient_Descent_Derivatives *master_gradient_descent_derivatives, float **target_batch_data, unsigned long long batch_size, pthread_mutex_t* master_gradient_descent_derivatives_mutex, pthread_mutex_t* has_computed_batch_index_mutex, float *aggregate_batch_accuracy)
{
    allocate_gradient_descent_derivatives(node_network, master_gradient_descent_derivatives);
    reset_gradient_descent_derivatives(node_network, master_gradient_descent_derivatives);

    gradient_descent_master_handler_data->has_computed_batch_index_mutex = has_computed_batch_index_mutex;
    gradient_descent_master_handler_data->master_gradient_descent_derivatives_mutex = master_gradient_descent_derivatives_mutex;

    gradient_descent_master_handler_data->target_batch_data = target_batch_data;
    *aggregate_batch_accuracy = 0.0f;
    gradient_descent_master_handler_data->aggregate_batch_accuracy = aggregate_batch_accuracy;
    gradient_descent_master_handler_data->master_gradient_descent_derivatives = master_gradient_descent_derivatives;
    gradient_descent_master_handler_data->node_network = node_network;

    gradient_descent_master_handler_data->batch_size = batch_size;

    gradient_descent_master_handler_data->has_computed_batch_indexes = (uint8_t *)malloc(sizeof(uint8_t) * batch_size);
    if (gradient_descent_master_handler_data->has_computed_batch_indexes == NULL)
    {
        printf("Unable to allocate memory for has computed batch indexes for master gradient descent handler\n");
        exit(EXIT_FAILURE);
    }
}

void destroy_master_handler(struct Gradient_Descent_Master_Handler_Data *gradient_descent_master_handler_data)
{
    free(gradient_descent_master_handler_data->has_computed_batch_indexes);
    deallocate_gradient_descent_derivatives(gradient_descent_master_handler_data->node_network, gradient_descent_master_handler_data->master_gradient_descent_derivatives);
}



void *gradient_descent_worker_thread(void *thread_arg)
{
    struct Gradient_Descent_Worker_Thread_Data *gradient_descent_worker_thread_data = (struct Gradient_Descent_Worker_Thread_Data *)thread_arg;
    struct Gradient_Descent_Master_Handler_Data *gradient_descent_master_handler_data = gradient_descent_worker_thread_data->gradient_descent_master_handler_data;

    struct Node_Network *node_network = gradient_descent_master_handler_data->node_network;
    struct Gradient_Descent_Derivatives *gradient_descent_derivatives = gradient_descent_worker_thread_data->gradient_descent_derivatives;
    struct Node_Network_Data_Partition *node_network_data_partition = gradient_descent_worker_thread_data->node_network_data_partition;

    struct Gradient_Descent_Derivatives *master_gradient_descent_derivatives = gradient_descent_master_handler_data->master_gradient_descent_derivatives;

    uint8_t thread_index = gradient_descent_worker_thread_data->thread_index;

    //printf("Init %llu\n", thread_index);

    for(uint8_t batch_index = 0; batch_index < gradient_descent_master_handler_data->batch_size; batch_index++)
    {
        pthread_mutex_lock(gradient_descent_master_handler_data->has_computed_batch_index_mutex);
        if (gradient_descent_master_handler_data->has_computed_batch_indexes[batch_index] == 1)
        {
            pthread_mutex_unlock(gradient_descent_master_handler_data->has_computed_batch_index_mutex);
            continue;
        }
        gradient_descent_master_handler_data->has_computed_batch_indexes[batch_index] = 1;
        pthread_mutex_unlock(gradient_descent_master_handler_data->has_computed_batch_index_mutex);

        node_network_data_partition->node_layer_data_partitions[0].inputs = gradient_descent_master_handler_data->target_batch_data[batch_index];
        compute_node_network(node_network, node_network_data_partition);

        (*gradient_descent_master_handler_data->aggregate_batch_accuracy) += calculate_data_difference_squared(node_network_data_partition->node_layer_data_partitions[node_network->node_layer_count - 1].outputs, gradient_descent_master_handler_data->target_batch_data[batch_index], node_network->node_layers[node_network->node_layer_count - 1].output_count) / gradient_descent_master_handler_data->batch_size;

        node_network_derivatives(node_network, node_network_data_partition, gradient_descent_derivatives, gradient_descent_master_handler_data->target_batch_data[batch_index]);

        pthread_mutex_lock(gradient_descent_master_handler_data->master_gradient_descent_derivatives_mutex);
        for (uint8_t node_layer_index = 0; node_layer_index < node_network->node_layer_count; node_layer_index++)
        {
            struct Node_Layer *node_layer = &(node_network->node_layers[node_layer_index]);
            add_to_matrix_with_second_term_coefficient(master_gradient_descent_derivatives->activation_bias_derivatives[node_layer_index], gradient_descent_derivatives->activation_bias_derivatives[node_layer_index], 1, node_layer->output_count, 1.0f);
            add_to_matrix_with_second_term_coefficient(master_gradient_descent_derivatives->input_weight_derivatives[node_layer_index], gradient_descent_derivatives->input_weight_derivatives[node_layer_index], node_layer->output_count, node_layer->input_count, 1.0f);
        }
        pthread_mutex_unlock(gradient_descent_master_handler_data->master_gradient_descent_derivatives_mutex);

        //printf("Gradient descent batch index %i completed by worker thread %i\n", batch_index, thread_index);
    }

    pthread_exit(NULL);
    return 0;
}


void gradient_descent_cycle(struct Node_Network *node_network, struct Gradient_Descent_Derivatives *gradient_descent_derivatives, struct Node_Network_Data_Partition *node_network_data_partitions, float **target_batch_data, unsigned long long batch_size, float *aggregate_batch_accuracy)
{
    pthread_mutex_t master_gradient_descent_derivatives_mutex;
    pthread_mutex_t has_computed_batch_index_mutex;
    pthread_mutex_init(&master_gradient_descent_derivatives_mutex, NULL);
    pthread_mutex_init(&has_computed_batch_index_mutex, NULL);

    struct Gradient_Descent_Derivatives master_gradient_descent_derivatives;

    struct Gradient_Descent_Worker_Thread_Data gradient_descent_worker_thread_data[THREAD_COUNT];

    struct Gradient_Descent_Master_Handler_Data gradient_descent_master_handler_data = {0};
    initialize_master_handler_data(&gradient_descent_master_handler_data, node_network, &master_gradient_descent_derivatives, target_batch_data, batch_size, &master_gradient_descent_derivatives_mutex, &has_computed_batch_index_mutex, aggregate_batch_accuracy);

    for (uint8_t thread_data_index = 0; thread_data_index < THREAD_COUNT; thread_data_index++)
    {
        gradient_descent_worker_thread_data[thread_data_index].gradient_descent_master_handler_data = &gradient_descent_master_handler_data;
    }

    pthread_t threads[THREAD_COUNT];
    int return_code;
    
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for(uint8_t thread_index = 0; thread_index < THREAD_COUNT; thread_index++)
    {
        gradient_descent_worker_thread_data[thread_index].thread_index = thread_index;
        gradient_descent_worker_thread_data[thread_index].gradient_descent_derivatives = &(gradient_descent_derivatives[thread_index]);
        gradient_descent_worker_thread_data[thread_index].node_network_data_partition = &(node_network_data_partitions[thread_index]);
        return_code = pthread_create(&threads[thread_index], &attr, gradient_descent_worker_thread, (void *)&(gradient_descent_worker_thread_data[thread_index]));
        if (return_code) {
            printf("ERROR; return code from pthread_create() is %d\n", return_code);
            exit(EXIT_FAILURE);
        }
    }
    
    void *status;

    pthread_attr_destroy(&attr);
    for(unsigned long long thread_index = 0; thread_index < THREAD_COUNT; thread_index++) {
        return_code = pthread_join(threads[thread_index], &status);
        if (return_code) {
            printf("ERROR; return code from pthread_join() is %d\n", return_code);
            exit(EXIT_FAILURE);
        }
        //printf("Completed join with thread %ld having a status of %llu\n", thread_index, (unsigned long long)status);
    }

    for (uint8_t node_layer_index = 0; node_layer_index < node_network->node_layer_count; node_layer_index++)
    {
        float layer_activation_bias_factor = 1.0f;
        float layer_weights_factor = 1.0f;

        if (SHOULD_DIVIDE_NODE_LAYER_BY_DEPTH)
        {
            layer_activation_bias_factor /= node_layer_index + 1;
            layer_weights_factor /= node_layer_index + 1;
        }

        struct Node_Layer *node_layer = &(node_network->node_layers[node_layer_index]);

        add_to_matrix_with_second_term_coefficient(node_layer->activation_biases, master_gradient_descent_derivatives.activation_bias_derivatives[node_layer_index], 1, node_layer->output_count, -1.0f * DESCENT_DERIVATIVE_BIAS_STRENGTH_FACTOR * layer_activation_bias_factor / (float)batch_size);
        add_to_matrix_with_second_term_coefficient(node_layer->input_weights, master_gradient_descent_derivatives.input_weight_derivatives[node_layer_index], node_layer->output_count, node_layer->input_count, -1.0f * DESCENT_DERIVATIVE_WEIGHT_STRENGTH_FACTOR * layer_weights_factor / (float)batch_size);
    }

    destroy_master_handler(&gradient_descent_master_handler_data);

    pthread_mutex_destroy(&master_gradient_descent_derivatives_mutex);
    pthread_mutex_destroy(&has_computed_batch_index_mutex);
}