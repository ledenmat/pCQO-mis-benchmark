#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <filesystem>

#include <string>

// SATLIB settings
// #define LEARNING_RATE 0.0003
// #define MOMENTUM 0.875
// #define NUM_ITERATIONS 7500
// #define NUM_ITERATIONS_PER_BATCH 30
// #define GAMMA 900
// #define GAMMA_PRIME 1
// #define BATCH_SIZE 256
// #define STD 2.25

// ER settings
// #define LEARNING_RATE 0.000009
// #define MOMENTUM 0.9
// #define NUM_ITERATIONS 225000
// #define NUM_ITERATIONS_PER_BATCH 450
// #define GAMMA 350
// #define GAMMA_PRIME 7
// #define BATCH_SIZE 256
// #define STD 2.25

// Declare settings as global variables
// Read in the first argument as the file path
std::string FILE_PATH;
float LEARNING_RATE;
float MOMENTUM;
int NUM_ITERATIONS;
int NUM_ITERATIONS_PER_BATCH;
int GAMMA;
int GAMMA_PRIME;
int BATCH_SIZE;
float STD;
int OUTPUT_INTERVAL;
std::string INITIALIZATION_VECTOR;

// Function to parse user input and set the global variables
void parse_user_input(int argc, const char *argv[])
{
    if (argc > 10)
    {
        FILE_PATH = argv[1];
        LEARNING_RATE = std::stof(argv[2]);
        MOMENTUM = std::stof(argv[3]);
        NUM_ITERATIONS = std::stoi(argv[4]);
        NUM_ITERATIONS_PER_BATCH = std::stoi(argv[5]);
        GAMMA = std::stoi(argv[6]);
        GAMMA_PRIME = std::stoi(argv[7]);
        BATCH_SIZE = std::stoi(argv[8]);
        STD = std::stof(argv[9]);
        OUTPUT_INTERVAL = std::stoi(argv[10]);
    } else {
        std::cerr << "Usage: " << argv[0] << " <file_path> <learning_rate> <momentum> <num_iterations> <num_iterations_per_batch> <gamma> <gamma_prime> <batch_size> <std> <output_interval>" << std::endl;
        exit(1);
    }
    if (argc > 11)
    {
        INITIALIZATION_VECTOR = argv[11];
    }
}

torch::TensorOptions default_tensor_options = torch::TensorOptions().dtype(torch::kFloat16);
torch::TensorOptions default_tensor_options_gpu = default_tensor_options.device(torch::kCUDA);

class InitializationSampler
{
private:
    torch::Tensor mean_vector;
    float stdev;

public:
    InitializationSampler(torch::Tensor mean_vector, float stdev)
    {
        this->mean_vector = mean_vector;
        this->stdev = stdev;
    }
    torch::Tensor sample(torch::Tensor matrix)
    {
        torch::Tensor sample = torch::normal_out(matrix, mean_vector, STD, std::nullopt);
        return sample;
    }
    torch::Tensor sample_previous(torch::Tensor matrix)
    {
        mean_vector = matrix.clone();
        torch::Tensor sample = torch::normal_out(matrix, mean_vector, STD, std::nullopt);
        return sample;
    }
};

class Optimizer
{
public:
    float gamma;
    float gamma_prime;
    float learning_rate;
    float momentum;
    torch::Tensor velocity;
    Optimizer(float learning_rate, float momentum, float gamma, float gamma_prime, int graph_order)
    {
        this->learning_rate = learning_rate;
        this->momentum = momentum;
        this->velocity = torch::zeros({BATCH_SIZE, graph_order}, default_tensor_options_gpu);
        this->gamma = gamma;
        this->gamma_prime = gamma_prime;
    }
    torch::Tensor compute_gradient(torch::Tensor &adjacency_matrix, torch::Tensor &adjacency_matrix_comp, torch::Tensor &X)
    {
        torch::Tensor first = torch::matmul(X, adjacency_matrix);
        torch::Tensor second = torch::matmul(X, adjacency_matrix_comp);
        torch::Tensor gradient = -1 + first.mul(gamma) - second.mul(gamma_prime);
        return gradient;
    }
    torch::Tensor velocity_update(torch::Tensor &X, torch::Tensor &gradient)
    {
        torch::Tensor momentum_term = velocity.mul(momentum);
        torch::Tensor learning_rate_term = gradient.mul(learning_rate);
        this->velocity = momentum_term.add(learning_rate_term);
        return X.sub(this->velocity);
    }
};

// Write a function that reads a graph from a file in DIMACS format and returns a cpp vector
torch::Tensor read_graph_from_file(const std::string &file_path)
{
    std::ifstream file(file_path);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file");
    }

    std::string header_line;

    while (std::getline(file, header_line))
    {
        if (header_line[0] == 'p')
        {
            break;
        }
    }

    // Header_line should look like this: "p edge 200 9876", we need to grab the second two numbers

    std::istringstream header_stream(header_line);
    std::string p, format;
    signed long number_of_nodes, number_of_edges;
    // std::cout << header_line << std::endl;
    header_stream >> p >> format >> number_of_nodes >> number_of_edges;
    // std::cout << "Number of nodes: " << number_of_nodes << std::endl;

    // torch::Tensor graph_entries = torch::zeros({number_of_nodes*number_of_nodes});
    std::vector<float> graph_entries(number_of_nodes * number_of_nodes, 0.0);

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream edge_stream(line);
        char a;
        size_t node_one, node_two;
        edge_stream >> a >> node_one >> node_two;
        graph_entries[(node_one - 1) * number_of_nodes + (node_two - 1)] = 1.0;
        graph_entries[(node_two - 1) * number_of_nodes + (node_one - 1)] = 1.0;
    }

    // Convert the vector to a tensor
    file.close();
    return torch::from_blob(graph_entries.data(), {number_of_nodes, number_of_nodes}).to(default_tensor_options_gpu);
}

torch::Tensor generate_complement_graph(const torch::Tensor &graph)
{
    return torch::ones({graph.sizes()[0], graph.sizes()[1]}, default_tensor_options_gpu) - graph - torch::eye(graph.sizes()[0], default_tensor_options_gpu);
}

int main(int argc, const char *argv[])
{
    int sum_max = 0;
    int count = 0;

    parse_user_input(argc, argv);

    std::cout << FILE_PATH << "-" << GAMMA << "-" << LEARNING_RATE << "-" << GAMMA_PRIME << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    torch::manual_seed(113);

    torch::Tensor adjacency_matrix = read_graph_from_file(FILE_PATH);
    long number_of_nodes = adjacency_matrix.sizes()[0];
    // std::cout << "Number of nodes: " << number_of_nodes << std::endl;

    torch::Tensor adjacency_matrix_comp = generate_complement_graph(adjacency_matrix);

    // Compute the degree of each node in the graph
    torch::Tensor degrees = adjacency_matrix.sum(0);

    // Compute the max degree of the graph
    float max_degree = degrees.max().item<float>();
    // std::cout << "Max degree: " << degrees.max().item<float>() << std::endl;

    // Check if the initialization directory is provided and not empty
    torch::Tensor mean_vector;
    if (!INITIALIZATION_VECTOR.empty())
    {
        std::istringstream iss(INITIALIZATION_VECTOR);
        std::vector<float> mean_vector_data;
        std::string value;

        while (std::getline(iss, value, ' '))
        {
            mean_vector_data.push_back((value == "1") ? 1.0f : 0.0f);
        }

        if (mean_vector_data.size() != number_of_nodes)
        {
            throw std::runtime_error("Initialization vector size does not match the number of nodes in the graph");
        }

        mean_vector = torch::from_blob(mean_vector_data.data(), {number_of_nodes}, default_tensor_options).clone();
        mean_vector = mean_vector.unsqueeze(0).expand({BATCH_SIZE, -1}).to(torch::kCUDA);
    }
    else
    {
        // Create the mean vector if no initialization directory is provided
        mean_vector = torch::zeros({number_of_nodes}, default_tensor_options);
        for (int i = 0; i < number_of_nodes; i++)
        {
            mean_vector[i] = 1.0 - (degrees[i] / (max_degree));
        }
        mean_vector = mean_vector.unsqueeze(0).expand({BATCH_SIZE, -1}).to(torch::kCUDA);
    }


    InitializationSampler sampler = InitializationSampler(mean_vector, STD);

    // Create initilzation matrix and sample from a normal distribution with mean_vector and std 0.01
    torch::Tensor X = sampler.sample(torch::zeros({BATCH_SIZE, number_of_nodes}, default_tensor_options_gpu));

    // //std::cout << "Initialization matrix: " << X << std::endl;

    Optimizer optimizer = Optimizer(LEARNING_RATE, MOMENTUM, GAMMA, GAMMA_PRIME, number_of_nodes);

    int max = 0;
    torch::Tensor max_vector;
    //std::cout << "Starting optimization" << std::endl;

    torch::Tensor ones_vector = torch::ones({number_of_nodes}, default_tensor_options_gpu);
    torch::Tensor update = number_of_nodes * adjacency_matrix - adjacency_matrix_comp;

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++)
    {
        torch::Tensor gradient = optimizer.compute_gradient(adjacency_matrix, adjacency_matrix_comp, X);
        X = optimizer.velocity_update(X, gradient);

        // Clamp the initialization matrix to be between 0 and 1
        X = X.clamp(0, 1);

        if ((iteration + 1) % NUM_ITERATIONS_PER_BATCH == 0)
        {
            torch::Tensor masks = X.gt(0.5).to(torch::kFloat16);
            // Iterate over the batch dimension of the masks tensor
            // torch::Tensor sums = masks.sum(1);

            for (int i = 0; i < masks.sizes()[0]; i++)
            {
                // if (sums[i].item<float>() > 0 && masks[i].t().matmul(adjacency_matrix).matmul(masks[i]).item<float>() == 0)
                // {
                    torch::Tensor binarized_update = masks[i] - 0.1 * (-ones_vector + masks[i].matmul(update));
                    binarized_update.clamp_(0, 1);
                    if (torch::equal(binarized_update, masks[i]))
                    {
                        if (masks[i].sum().item<float>() > max)
                        {
                            max = masks[i].sum().item<float>();
                            max_vector = masks[i].clone();
                        }
                    }
                // }
            }

            if (iteration + 1 == NUM_ITERATIONS_PER_BATCH || ((iteration + 1) / NUM_ITERATIONS_PER_BATCH) % OUTPUT_INTERVAL == 0 || iteration + 1 == NUM_ITERATIONS)
            {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds = end - start;
                std::cout << (iteration + 1) / NUM_ITERATIONS_PER_BATCH << std::endl;
                std::cout << max << std::endl;
                std::cout << elapsed_seconds.count() << std::endl;

            }
            if (iteration + 1 != NUM_ITERATIONS)
            {
                X = sampler.sample_previous(X);
            } else {
                // Print the max vector on one line
                for (int i = 0; i < max_vector.sizes()[0]; i++)
                {
                    std::cout << max_vector[i].item<float>() << (i == max_vector.sizes()[0] - 1 ? "\n" : " ");
                }
            }
        }
    }
}