# Differentiable Quadratic Optimization For The Maximum Independent Set Problem

# pCQO-MIS v1 #

## ICML 2025

## Description

This repository contains the code for the (pCQO-MIS) method. The goal is to provide tools and implementations for the experiments conducted in the submission.

- [pCQO-MIS v1](#pcqo-mis-v1)
  - [Description](#description)
  - [pCQO-MIS](#pcqo-mis)
    - [Application Setup](#application-setup)
    - [Running the Application](#running-the-application)
  - [Benchmark Setup](#benchmark-setup)
    - [Prerequisites](#prerequisites)
    - [Setup and Installation](#setup-and-installation)
    - [Configuration](#configuration)
      - [Graph Import](#graph-import)
      - [Solver Configuration](#solver-configuration)
      - [Warning](#warning)
    - [Running the Script](#running-the-script)
      - [Checkpoints and Final Results](#checkpoints-and-final-results)
    - [Output](#output)
  - [Basic Hyper-Parameters Fine-Tuning](#basic-hyper-parameters-fine-tuning)
  - [Notes](#notes)

## pCQO-MIS

### Application Setup

1. Install [LibTorch](https://pytorch.org/get-started/locally/).
2. Clone the repository and navigate to the `./cpp_impl` directory.
3. Run the following CMake command:  
   ```bash
   cmake -DCMAKE_PREFIX_PATH={path to libtorch} ..
   ```
4. Build the program using:  
   ```bash
   cmake --build . --config Release
   ```
   The executable `pcqo_mis` will be available in the `./external` directory.

### Running the Application

To run the `pcqo_mis` application, follow these steps:

1. **Build the Application**  
   Ensure the application is built as described in the [Application Setup](#application-setup) section. The executable `pcqo_mis` should be available in the `./external` directory.

2. **Prepare Input Data**  
   The application requires a graph file in DIMACS format as input. Ensure the graph file is accessible and correctly formatted.

3. **Run the Application**  
   Use the following command to execute the application:

   ```bash
   ./external/pcqo_mis <file_path> [<learning_rate> <momentum> <num_iterations> <num_iterations_per_batch> <gamma> <gamma_prime> <batch_size> <std> <output_interval>] [initialization_vector]
   ```

   - `<file_path>`: Path to the DIMACS graph file (required).
   - All other parameters are optional. If not provided, a grid search will be performed to attempt to find suitable values. **Note:** This grid search may result in suboptimal parameters that do not yield ideal results. It is strongly recommended to provide these parameters for better performance.

4. **Example Command**  
   For example, to run the application with a graph file `graph.dimacs` and specific parameters:

   ```bash
   ./external/pcqo_mis graph.dimacs 0.0003 0.875 7500 30 900 1 256 2.25 10
   ```
   For more information about the DIMACS format, see utils/NetworkX_to_DIMACS.ipynb

5. **Output**  
   - The application prints intermediate results at intervals specified by `<output_interval>`.
   - At the end of execution, the maximum independent set found is printed as a binary vector.

6. **Notes**  
   - Ensure the graph file is correctly formatted and accessible.
   - Adjust parameters as needed for different graph sizes or optimization requirements.

## Benchmark Setup

### Prerequisites

- Python 3.10
- [pCQO-MIS Setup](#application-setup)
- Required libraries: `pandas`, `torch`, `gurobipy`, `ortools`

### Setup and Installation

1. Clone the repository and navigate to the repository's root directory.
2. Create a virtual environment using `venv`:
   ```bash
   python3 -m venv .venv
   ```
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
4. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
5. (Optional) If using Gurobi, obtain a license and install it on the machine.
6. (Optional) If using ReduMIS, clone the [KaMIS project](https://github.com/KarlsruheMIS/KaMIS), build the ReduMIS program, and place it in the `external` folder.
7. Retrieve datasets from the `/graphs` folder used in the original experiments.
8. Run the benchmarking suite:
   ```bash
   python benchmark.py
   ```

### Configuration

#### Graph Import

Specify the directories containing the graph data by editing the `graph_directories` list in `benchmark.py`. For example:

```python
graph_directories = [
    "./graphs/satlib/m403",
    "./graphs/satlib/m411",
    "./graphs/satlib/m418",
]
```

#### Solver Configuration

Define the solvers to use in the `solvers` list. Uncomment the solvers and specify their parameters. For example:

```python
solvers = [
    {
        "name": "Gurobi",
        "class": GurobiMIS,
        "params": {"time_limit": 100},
    },
    {
        "name": "CPSAT",
        "class": CPSATMIS,
        "params": {"time_limit": 30},
    },
]
```


#### Warning

When running the `pcqo_mis` application, ensure that all arguments are provided. If any optional arguments are omitted, the application will default to a slower grid search to determine suitable values. This grid search may result in suboptimal parameters and significantly increase runtime. Providing all arguments is strongly recommended for optimal performance.


### Running the Script

Run the script to start the benchmarking process:

```bash
python benchmark.py
```

#### Checkpoints and Final Results

Intermediate results are saved at intervals defined by `SOLUTION_SAVE_INTERVAL`. Final results are saved as CSV files named `zero_to_stage_{current_stage}_of_{total_stages}_total_stages.csv`.

### Output

The script generates a CSV file containing results for each graph and solver, including solution sizes and time taken.

## Basic Hyper-Parameters Fine-Tuning

For new graphs, a basic hyper-parameter search procedure is provided to assist in setting up $T$ and $\alpha$. You can use it by running `pcqo_mis` without all optional parameters. **Note:** This grid search may result in suboptimal parameters that do not yield ideal results. It is strongly recommended to determine these parameters yourself for better performance.

## Notes

- Ensure graph data and solver implementations are correctly set up and accessible.
- Adjust `SOLUTION_SAVE_INTERVAL` to control the frequency of checkpoint saves.
- The benchmarking process may take significant time depending on the number and size of graphs and solvers used.
- For large datasets exceeding local RAM, use the `benchmark_large_graphs.py` script.

