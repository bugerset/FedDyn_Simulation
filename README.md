
# FedDyn on CIFAR-10 and MNIST with MobileNet (PyTorch) #

This repository provides a PyTorch implementation of FedDyn (Federated Learning based on Dynamic Regularization) using MobileNet on CIFAR-10 and MNIST datasets. 
It is designed to handle Non-IID data distributions effectively by introducing dynamic regularization to mitigate client drift.

## Project Overview ##
FedDyn optimizes the global objective by dynamically updating a regularizer for each client. This simulation covers:

Data Partitioning: IID and Non-IID (Dirichlet distribution) splits.

State Management: Handling server state h and client states g k​ ,θ k
​
  for dynamic regularization.

Optimized Aggregation: Server-side update logic that specifically excludes BatchNorm parameters from FedDyn regularization to maintain stability.

Hardware Acceleration: Built-in support for Apple Silicon (MPS) and CUDA.

Folder Structure
Plaintext

├── main.py               # Main entry point for the simulation
├── feddyn_experiment.py  # Single-client experiment script
├── data/
│   ├── cifar10.py        # Data loading for CIFAR-10
│   └── partition.py      # IID/Non-IID partitioning logic
├── fl/
│   ├── feddyn.py         # Local training with FedDyn loss
│   └── server.py         # FedDyn server aggregation & h-state update
├── models/
│   └── mobilenet.py      # MobileNet architecture (supports BN/GN)
└── utils/
    ├── device.py         # Device selection (CPU/CUDA/MPS)
    └── eval.py           # Evaluation metrics
Requirements
Python 3.9+

PyTorch (with MPS or CUDA support)

torchvision, numpy

Usage
Run with default settings (FedDyn + CIFAR-10):

Bash

python main.py --train feddyn --dyn-alpha 0.05 --num-clients 10 --client-frac 0.5
For a single-client experiment to verify convergence:

Bash

python feddyn_experiment.py
Core Arguments
Argument	Default	Description
--train	feddyn	Federated learning algorithm to use
--dyn-alpha	0.05	Regularization strength α for FedDyn
--partition	niid	Data split method (iid or niid)
--alpha	0.5	Dirichlet concentration for Non-IID severity
--device	auto	Training device (cpu, cuda, or mps)

Sheets로 내보내기

Implementation Details
FedDyn Loss Function
Each client minimizes a local loss that includes a linear term and a quadratic proximal term:

L 
total
​
 =L 
task
​
 −⟨g 
k
​
 ,θ⟩+ 
2
α
​
 ∥θ−θ 
k
t
​
 ∥ 
2
 
where g 
k
​
  is the local gradient state and θ 
k
t
​
  is the previous global model.

BatchNorm Handling
To ensure model stability, the server updates the global model by applying the FedDyn correction only to learnable parameters (weights and biases), while performing simple averaging for BatchNorm buffers (running_mean, running_var).
