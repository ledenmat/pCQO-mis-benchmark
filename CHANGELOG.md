# Changelog

All notable changes to this project will be documented in this file.

## 2023-11-10

### Added

- ReduMIS solver and new project structure.
- Subgraph analysis for the dNN w/subgraph method in the performance analysis notebook.

## 2023-11-3

### Added

- New Solver abstraction. This allows the user to create new MIS solvers and benchmark them in a consistent manner.
- New performance analysis notebook. Using the new solver abstraction, this notebook allows you to run a series of datasets across all solvers and plot their results.

## 2023-10-20

### Added

- Made major edits to all notebooks. Moved functions from notebooks to datalessNN.py
- Added a new notebook and related library for the SAM optimizer (sam.py, demo_with_SAM.ipynb)

## Moved

- Custom Weight Initialization Module (datalessNN_graph_params.py) to datalessNN.py
- Dataless NN Pytorch Module (datalessNN_module.py) to datalessNN.py

## 2023-10-14

### Added

- Custom Pytorch Layers (custom_layers.py)
- Custom Pytorch Layer Constraints (layer_constraints.py)
- Custom Weight Initialization Module (datalessNN_graph_params.py)
- Dataless NN Pytorch Module (datalessNN_module.py)
- Experimental notebook demo'ing the original code from Ismail Alkhouri (@ialkhouri)