# pCQO-MIS v1 #

## Description ##
This repository houses the code for (pCQO-MIS) method. The goal of this repository is to provide tools and implementations for the experiments performed in the submission.

- [pCQO-MIS v1](#pcqo-mis-v1)
  - [Description](#description)
  - [Prerequisites](#prerequisites)
  - [Setup and Installation](#setup-and-installation)
  - [Configuration](#configuration)
    - [Graph Import](#graph-import)
    - [Solver Configuration](#solver-configuration)
  - [Running the Script](#running-the-script)
    - [Checkpoints and Final Results](#checkpoints-and-final-results)
  - [Customization](#customization)
    - [Initializers](#initializers)
    - [Example: Degree-based Initializer](#example-degree-based-initializer)
  - [Output](#output)
  - [Notes](#notes)


## Prerequisites

- Python 3.10
- Required libraries: `pandas`, `torch`, `gurobipy`, `ortools`

## Setup and Installation

1. Clone the repository and navigate to the repository's root directory.
2. Create a virtual environment using venv
   ```bash
   python3 -m venv .venv
   ```
3. Activate your environment
   ```bash
   source .venv/bin/activate
   ```
4. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
5. (If you want to run Gurobi) Obtain licenses for Gurobi and install that license on the machine you will be running this repository on.
6. (If you want to run ReduMIS) Clone the [KaMIS project](https://github.com/KarlsruheMIS/KaMIS) and build a copy of the ReduMIS program. Place the program in the `external` folder of this repository.
7. Browse the /graphs folder to retrieve the datasets used in the original experiments.
8. Run the benchmarking suite:
   ```bash
   python benchmark.py
   ```


## Configuration

### Graph Import

Specify the directories containing the graph data by uncommenting the appropriate lines in the `graph_directories` list within `benchmark.py`. For example:

```python
graph_directories = [
    "./graphs/satlib/m403",
    "./graphs/satlib/m411",
    "./graphs/satlib/m418",
]
```

### Solver Configuration

Define the solvers you want to use in the `solvers` list. Uncomment the solvers you want to benchmark and specify their parameters. For example:

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

## Running the Script

Run the script to start the benchmarking process:

```bash
python benchmark.py
```

### Checkpoints and Final Results

The script saves intermediate results at regular intervals defined by `SOLUTION_SAVE_INTERVAL`. The final results are saved at the end of the benchmarking process. Output files are saved as CSV files named `zero_to_stage_{current_stage}_of_{total_stages}_total_stages.csv`.

## Customization

### Initializers

The script includes optional initializers for the solvers:

1. **Default initializer**: Uniform distribution.
2. **Degree-based initializer**: Initialize values based on node degrees.

Specify the relevant initializer in the solver's params.

### Example: Degree-based Initializer

```python
mean_vector = []
degrees = dict(graph["data"].degree())

# Find the maximum degree
max_degree = max(degrees.values())

for _, degree in graph["data"].degree():
    degree_init = 1 - degree / max_degree
    mean_vector.append(degree_init)

min_degree_initialization = max(mean_vector)

for i in range(len(mean_vector)):
    mean_vector[i] = mean_vector[i] / min_degree_initialization

solver_instance.value_initializer = lambda _: torch.normal(
    mean=torch.Tensor(mean_vector), std=solver["params"]["std"]
)
```

## Output

The script outputs a CSV file containing the results for each graph and solver, including solution sizes and time taken for each solver.

## Notes

- Ensure the graph data and solver implementations are correctly set up and accessible.
- Adjust the `SOLUTION_SAVE_INTERVAL` as needed to control the frequency of checkpoint saves.
- The benchmarking process may be time-consuming depending on the number and size of graphs, and the solvers used.
- Large datasets that exceed local RAM can be run using the ```benchmark_large_graphs.py``` script.
