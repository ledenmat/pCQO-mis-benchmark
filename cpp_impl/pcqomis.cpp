#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <filesystem>


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

torch::TensorOptions default_tensor_options = torch::TensorOptions().dtype(torch::kFloat16);
torch::TensorOptions default_tensor_options_gpu = default_tensor_options.device(torch::kCUDA);

class InitializationSampler
{
public:
    torch::Tensor mean_vector;
    float std;
    InitializationSampler(torch::Tensor mean_vector, float std)
    {
        this->mean_vector = mean_vector;
        this->std = std;
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
    //std::cout << header_line << std::endl;
    header_stream >> p >> format >> number_of_nodes >> number_of_edges;
    //std::cout << "Number of nodes: " << number_of_nodes << std::endl;

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

// Write a function that benchmarks the performance of the above function
void benchmark_read_graph_from_file(const std::string &file_path)
{
    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor tensor = read_graph_from_file(file_path);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    //std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    // list tensor size
    //std::cout << "Tensor size: " << tensor.sizes() << std::endl;
}

// Write a function that benchmarks the performance of the above function
void benchmark_generate_complement_graph(const torch::Tensor &graph)
{
    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor complement_graph = generate_complement_graph(graph);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    //std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    // list tensor size
    //std::cout << "Tensor size: " << complement_graph.sizes() << std::endl;
}

int main(int argc, const char *argv[])
{
    // Read in the first arugment as the file path
    std::string directory_path = argv[1];
    int sum_max = 0;
    int count = 0;

    for (const auto &entry : std::filesystem::directory_iterator(directory_path))
    {
        std::string file_path = entry.path().string();
        std::cout << entry.path().filename().string() << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        torch::manual_seed(113);

        torch::Tensor adjacency_matrix = read_graph_from_file(file_path);
        long number_of_nodes = adjacency_matrix.sizes()[0];
        //std::cout << "Number of nodes: " << number_of_nodes << std::endl;

        torch::Tensor adjacency_matrix_comp = generate_complement_graph(adjacency_matrix);

        // Compute the degree of each node in the graph
        torch::Tensor degrees = adjacency_matrix.sum(0);

        // Compute the max degree of the graph
        float max_degree = degrees.max().item<float>();
        //std::cout << "Max degree: " << degrees.max().item<float>() << std::endl;

        // Create a mean vector
        torch::Tensor mean_vector = torch::zeros({number_of_nodes}, default_tensor_options);

        for (int i = 0; i < number_of_nodes; i++)
        {
            mean_vector[i] = 1.0 - (degrees[i] / (max_degree));
        }

        mean_vector = mean_vector.unsqueeze(0).expand({BATCH_SIZE, -1}).to(torch::kCUDA);

        //std::cout << "Mean vector: " << mean_vector.sizes() << std::endl;

        InitializationSampler sampler = InitializationSampler(mean_vector, STD);

        // Create initilzation matrix and sample from a normal distribution with mean_vector and std 0.01
        torch::Tensor X = sampler.sample(torch::zeros({BATCH_SIZE, number_of_nodes}, default_tensor_options_gpu));

        // //std::cout << "Initialization matrix: " << X << std::endl;

        Optimizer optimizer = Optimizer(LEARNING_RATE, MOMENTUM, GAMMA, GAMMA_PRIME, number_of_nodes);

        int max = 0;
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
                            }
                        }
                    // }
                }

                if (iteration + 1 == NUM_ITERATIONS_PER_BATCH || ((iteration + 1) / NUM_ITERATIONS_PER_BATCH) % 10 == 0)
                {
                    auto end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_seconds = end - start;
                    std::cout << (iteration + 1) / NUM_ITERATIONS_PER_BATCH << std::endl;
                    std::cout << max << std::endl;
                    std::cout << elapsed_seconds.count() << std::endl;
                }
                X = sampler.sample_previous(X);
            }
        }
        // std::cout << "Max: " << max << std::endl;
        // sum_max += max;
        // count++;
    }
    // std::cout << "Average: " << sum_max / count << std::endl;
}