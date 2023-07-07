# MFSM_experiment (Multi-fidelity surrogate model experiment)

This repository contains an experimental study on multi-fidelity surrogate models using artificially constructed multi-fidelity data. The artificially constructed multi-fidelity data consists of three types: multi-fidelity data with additive noise, multi-fidelity data with multiplicative noise, and multi-fidelity data with mixed noise (additive and multiplicative).

The experiment utilizes three methods: additive correction, multiplicative correction, and Co-Kriging-based integrated correction. Each method is evaluated on the three different noise types of multi-fidelity data sets. A hypothetical ideal function (ground truth function) is assumed, and its parameters are randomly set for each experiment. The experiments are repeated 100 times, and the average mean squared error (MSE) is calculated.

## File Descriptions

- `100_correction.py`: Running this file will generate the experimental results for 100 random sets of parameters for the ground truth function. It calculates the average MSE for the three correction methods over the 100 experiments. Additionally, it includes a comparison with a high-fidelity data fitting model (HFDM).

- `100_parameters.txt`: This file displays the parameter selections for the ground truth function in each of the 100 experiments. Each line represents a set of parameters.

- `1_of_100_correction.py`: This script showcases the results of one specific experiment out of the 100. It plots the high and low fidelity data points under different noise conditions, along with the three correction methods and HFDM.

- `result.pdf`: This is the vector graphic illustration obtained after running `1_of_100_correction.py`, displaying the results.

Please refer to the individual file descriptions for more details on each file.

## Environment
Python Version: 3.11.4




