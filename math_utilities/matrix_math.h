double array_dot_product_step(double* data_1, size_t step_1, double* data_2, size_t step_2, size_t data_length)
{
    double dot_product_sum = 0.0;

    for (size_t data_index = 0; data_index < data_length; data_index++)
    {
        dot_product_sum += data_1[data_index * step_1] * data_2[data_index * step_2];
    }

    return dot_product_sum;
}

double array_dot_product(double* data_1, double* data_2, size_t data_length)
{
    return array_dot_product_step(data_1, 1, data_2, 1, data_length);
}

void multiply_matrices(double* output, double* data_1, size_t rows_1, size_t columns_1, double* data_2, size_t rows_2, size_t columns_2)
{
    if (columns_1 != rows_2)
    {
        printf("Attempted to multiply matrices that do not conform for matrix multiplication.\n");
        printf("Matrix 1: %llux%llu", rows_1, columns_1);
        printf("Matrix 2: %llux%llu", rows_2, columns_2);
        exit(EXIT_FAILURE);
    }

    for (size_t output_row_index = 0; output_row_index < columns_2; output_row_index++)
    {
        for (size_t output_column_index = 0; output_column_index < rows_1; output_column_index++)
        {
            size_t output_index = (output_row_index * columns_2) + output_column_index;
            output[output_index] = array_dot_product_step(&(data_1[output_row_index * columns_1]), 1, &(data_2[output_column_index]), columns_2, rows_2);
        }
    }
}

void add_matrices_with_second_term_coefficient(double* output, double* data_1, size_t rows_1, size_t columns_1, double* data_2, size_t rows_2, size_t columns_2, double second_term_coefficient)
{
    if (rows_1 != rows_2 || columns_1 != columns_2)
    {
        printf("Attempted to add matrices that do not conform for matrix addition.\n");
        printf("Matrix 1: %llux%llu", rows_1, columns_1);
        printf("Matrix 2: %llux%llu", rows_2, columns_2);
        exit(EXIT_FAILURE);
    }

    size_t data_size = rows_1 * columns_1;

    for (size_t data_index = 0; data_index < data_size; data_index++)
    {
        output[data_index] = data_1[data_index] + (second_term_coefficient * data_2[data_index]);
    }
}

void add_matrices(double* output, double* data_1, size_t rows_1, size_t columns_1, double* data_2, size_t rows_2, size_t columns_2)
{
    add_matrices_with_second_term_coefficient(output, data_1, rows_1, columns_1, data_2, rows_2, columns_2, 1.0);
}

void subtract_matrices(double* output, double* data_1, size_t rows_1, size_t columns_1, double* data_2, size_t rows_2, size_t columns_2)
{
    add_matrices_with_second_term_coefficient(output, data_1, rows_1, columns_1, data_2, rows_2, columns_2, -1.0);
}

void print_matrix(double *data, size_t rows, size_t columns)
{
    printf("\n");

    for (size_t row_index = 0; row_index < rows; row_index++)
    {
        printf("[ ");

        for (size_t column_index = 0; column_index < columns; column_index++)
        {
            size_t data_index = column_index + row_index * columns;

            printf("%.2f", data[data_index]);

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