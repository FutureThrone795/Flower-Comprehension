float array_dot_product_step(float* data_1, unsigned long long step_1, float* data_2, unsigned long long step_2, unsigned long long data_length)
{
    float dot_product_sum = 0.0f;

    for (unsigned long long data_index = 0; data_index < data_length; data_index++)
    {
        dot_product_sum += data_1[data_index * step_1] * data_2[data_index * step_2];
    }

    return dot_product_sum;
}

float array_dot_product(float* data_1, float* data_2, unsigned long long data_length)
{
    return array_dot_product_step(data_1, 1, data_2, 1, data_length);
}

void apply_activation_function_to_matrix(float* output, float* data, unsigned long long rows, unsigned long long columns, float(*func)(float, uint8_t), uint8_t is_output_layer)
{
    for (unsigned long long output_row_index = 0; output_row_index < rows; output_row_index++)
    {
        for (unsigned long long output_column_index = 0; output_column_index < columns; output_column_index++)
        {
            unsigned long long output_index = (output_row_index * columns) + output_column_index;
            float dat2 = data[output_index];
            float dat = func(dat2, is_output_layer);
            output[output_index] = dat;
        }
    }
}

void multiply_matrices(float* output, float* data_1, unsigned long long rows_1, unsigned long long columns_1, float* data_2, unsigned long long rows_2, unsigned long long columns_2)
{
    if (columns_1 != rows_2)
    {
        printf("Attempted to multiply matrices that do not conform for matrix multiplication.\n");
        printf("Matrix 1: %llux%llu\n", rows_1, columns_1);
        printf("Matrix 2: %llux%llu\n", rows_2, columns_2);
        exit(EXIT_FAILURE);
    }

    for (unsigned long long output_row_index = 0; output_row_index < rows_1; output_row_index++)
    {
        for (unsigned long long output_column_index = 0; output_column_index < columns_2; output_column_index++)
        {
            unsigned long long output_index = (output_row_index * columns_2) + output_column_index;
            output[output_index] = array_dot_product_step(&(data_1[output_row_index * columns_1]), 1, &(data_2[output_column_index]), columns_2, rows_2);
        }
    }
}

void add_matrices_with_second_term_coefficient(float* output, float* data_1, unsigned long long rows_1, unsigned long long columns_1, float* data_2, unsigned long long rows_2, unsigned long long columns_2, float second_term_coefficient)
{
    if (rows_1 != rows_2 || columns_1 != columns_2)
    {
        printf("Attempted to add matrices that do not conform for matrix addition.\n");
        printf("Matrix 1: %llux%llu", rows_1, columns_1);
        printf("Matrix 2: %llux%llu", rows_2, columns_2);
        exit(EXIT_FAILURE);
    }

    unsigned long long data_size = rows_1 * columns_1;

    for (unsigned long long data_index = 0; data_index < data_size; data_index++)
    {
        output[data_index] = data_1[data_index] + (second_term_coefficient * data_2[data_index]);
    }
}

void add_matrices(float* output, float* data_1, unsigned long long rows_1, unsigned long long columns_1, float* data_2, unsigned long long rows_2, unsigned long long columns_2)
{
    add_matrices_with_second_term_coefficient(output, data_1, rows_1, columns_1, data_2, rows_2, columns_2, 1.0f);
}

void add_to_matrix_with_second_term_coefficient(float* output, float* data, unsigned long long rows, unsigned long long columns, float second_term_coefficient)
{
    add_matrices_with_second_term_coefficient(output, output, rows, columns, data, rows, columns, second_term_coefficient);
}

void subtract_matrices(float* output, float* data_1, unsigned long long rows_1, unsigned long long columns_1, float* data_2, unsigned long long rows_2, unsigned long long columns_2)
{
    add_matrices_with_second_term_coefficient(output, data_1, rows_1, columns_1, data_2, rows_2, columns_2, -1.0f);
}

void print_matrix(float *data, unsigned long long rows, unsigned long long columns)
{
    printf("\n");

    for (unsigned long long row_index = 0; row_index < rows; row_index++)
    {
        printf("[ ");

        for (unsigned long long column_index = 0; column_index < columns; column_index++)
        {
            unsigned long long data_index = column_index + row_index * columns;

            printf("%f", data[data_index]);

            if (column_index < columns - 1)
            {
                printf(", ");
            }
        }

        if (row_index < rows - 1)
        {
            printf("],\n");
        }
        else
        {
            printf("]\n");
        }
    }

    printf("\n");
}

/*
void debug_matrix(float *data, unsigned long long rows, unsigned long long columns)
{
    for(unsigned long long i = 0; i < rows*columns; i++) 
    { 
        if (isnanf(data[i]))
        {
            print_matrix(data, rows, columns);
            return;
        }
    }
}
*/