void load_random_image(float *image_data_buffer, size_t image_size)
{
	int width, height, channels;
	char image_name[64];
	sprintf_s(image_name, 64, "cropped_102flowers/cropped_%04i.png", (rand() + rand() ^ rand())%8186);

	uint8_t *requested_image = stbi_load(image_name, &width, &height, &channels, 0);
    if (requested_image == NULL) 
    {
        printf("Image \"%s\" failed to load.\n", image_name);
        exit(EXIT_FAILURE);
    }

	if (width * height * channels != image_size)
	{
		printf("Supplied image size (%llu) does not match loaded image size (%llu).\n", image_size, (size_t)width * height * channels);
        exit(EXIT_FAILURE);
	}

	for (size_t image_data_index = 0; image_data_index < image_size; image_data_index++)
	{
		image_data_buffer[image_data_index] = 2.0f * (float)requested_image[image_data_index] / UINT8_MAX - 1.0f;
	}

	stbi_image_free(requested_image);
}

void save_image(const char* name, size_t width, size_t height, int channels, float* data)
{
	size_t image_size = width * height * channels;

	uint8_t *modified_image = (uint8_t *)malloc(image_size);
	if (modified_image == NULL)
	{
		printf("Failed to allocate memory for modified image.\n");
		exit(EXIT_FAILURE);
	}

	for (size_t image_data_index = 0; image_data_index < image_size; image_data_index++)
	{
		if (data[image_data_index] <= -1.0f)
		{
			modified_image[image_data_index] = 0;
			continue;
		}
		if (data[image_data_index] >= 1.0f)
		{
			modified_image[image_data_index] = UINT8_MAX;
			continue;
		}
		modified_image[image_data_index] = (uint8_t)(UINT8_MAX * (data[image_data_index] + 1.0f) / 2.0f);
	}

	stbi_write_png(name, width, height, channels, modified_image, width * channels);

	free(modified_image);
}

void adjoin_images(float *image_data_buffer, float *image_data_1, float*image_data_2, size_t single_image_size)
{
	for (int image_data_index_1 = 0; image_data_index_1 < single_image_size; image_data_index_1++)
	{
		image_data_buffer[image_data_index_1] = image_data_1[image_data_index_1];
	}

	for (int image_data_index_2 = 0; image_data_index_2 < single_image_size; image_data_index_2++)
	{
		image_data_buffer[image_data_index_2 + single_image_size] = image_data_2[image_data_index_2];
	}
}

void add_to_adjoined_image(float *image_data_buffer, float *image_data_1, float *image_data_2, size_t single_image_width, size_t single_image_height, size_t single_image_channels, size_t start_index)
{
	size_t single_image_size = single_image_width * single_image_height * single_image_channels;
	size_t stride_bytes = single_image_width * single_image_channels;

	for (int image_data_index = 0; image_data_index < single_image_size * 2; image_data_index++)
	{
		size_t single_image_data_index = image_data_index % (stride_bytes) + (image_data_index / (stride_bytes * 2)) * stride_bytes;

		if (image_data_index % (stride_bytes * 2) < stride_bytes)
		{
			image_data_buffer[start_index + image_data_index] = image_data_1[single_image_data_index];
		}
		else
		{
			image_data_buffer[start_index + image_data_index] = image_data_2[single_image_data_index];
		}
	}
}

/*
void save_node_network_as_image(struct Node_Network *node_network, char *name)
{
	size_t image_width = (size_t)ceil(sqrt(node_network->node_layers[0].input_count / 3));
	size_t image_height = (size_t)ceil(sqrt(node_network->node_layers[0].input_count / 3));

	for (size_t node_layer_index = 0; node_layer_index < node_network->node_layer_count; node_layer_index++)
	{
		size_t individual_image_side;

		individual_image_side = (size_t)ceil(sqrt(node_network->node_layers[node_layer_index].output_count / 3));
		image_height += individual_image_side;

		if (image_width < individual_image_side)
		{
			image_width = individual_image_side;
		}
	}

	float *image_data = (float *)malloc(sizeof(float) * image_width * image_height * 3);
	if (image_data == NULL)
	{
		printf("Failed to allocate memory for node network visualization image\n");
		exit(EXIT_FAILURE);
	}

	for (size_t image_data_index = 0; image_data_index < image_width * image_height * 3; image_data_index++)
	{
		image_data[image_data_index] = -2.0f;
	}

	size_t next_layer_image_start_index = 0;

	{
		size_t input_layer_image_side = (size_t)ceil(sqrt(node_network->node_layers[0].input_count / 3));
		for (size_t image_data_index = 0; image_data_index < node_network->node_layers[0].input_count; image_data_index++)
		{
			image_data[(image_data_index / input_layer_image_side) * image_width + (image_data_index % input_layer_image_side)] = node_network->node_layers[0].inputs[image_data_index];
		}
		next_layer_image_start_index = image_width * 3 * input_layer_image_side;
	}

	for (size_t node_layer_index = 0; node_layer_index < node_network->node_layer_count; node_layer_index++)
	{
		size_t individual_image_side = (size_t)sqrt(node_network->node_layers[node_layer_index].output_count / 3);
		for (size_t image_data_index = 0; image_data_index < node_network->node_layers[node_layer_index].output_count; image_data_index++)
		{
			//image_data[next_layer_image_start_index + image_data_index] = node_network->node_layers[0].inputs[(image_data_index / image_width) * individual_image_side + (image_data_index % (image_width))];
			image_data[next_layer_image_start_index + (image_data_index / individual_image_side) * image_width + (image_data_index % individual_image_side)] = node_network->node_layers[node_layer_index].raw_linear_outputs[image_data_index];
		}
		next_layer_image_start_index += image_width * 3 * individual_image_side;
	}

	save_image(name, image_width, image_height, 3, image_data);
}
*/