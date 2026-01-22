
# FedDyn on CIFAR-10 and MNIST with MobileNet (PyTorch)

This repository provides a PyTorch implementation of FedDyn (Federated Learning based on Dynamic Regularization) using MobileNet on CIFAR-10 and MNIST datasets. 
It is designed to handle Non-IID data distributions effectively by introducing dynamic regularization to mitigate client drift.

## Project Overview 
FedDyn optimizes the global objective by dynamically updating a regularizer for each client. This simulation covers:

Data Partitioning: IID and Non-IID (Dirichlet distribution) splits.

State Management: Handling server state h and client states $g_k​, θ_k$
​
  for dynamic regularization.

Optimized Aggregation: Server-side update logic that specifically excludes BatchNorm parameters from FedDyn regularization to maintain stability.

Folder Structure
~~~ 
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
~~~

## Requirements

- Python 3.9+ recommended
- PyTorch + torchvision
- numpy

Run with default settings:
```bash
python main.py
```
Example: FedAvg + IID
```bash
python main.py --train fedavg --partition iid
```
Example: FedAvg + Non-IID (Dirichlet)
```bash
python main.py --train fedavg --partition niid --alpha 0.5 --min-size 10
```
Example: FedProx (with mu)
```bash
python main.py --train fedprox --mu 0.1 --partition niid --alpha 0.5
```

## Device Selection

The code supports:
```
	•	--device auto (default): selects CUDA if available, else MPS (Apple Silicon), else CPU
	•	--device cuda
	•	--device mps
	•	--device cpu
```

Example:
```bash
python main.py --device auto
```

## CLI Arguments

Key arguments (from utils/parser.py):
```
	•	Reproducibility / compute
	•	--seed (default: 845)
	•	--device in {auto,cpu,cuda,mps}

	•	Training method
	•	--dyn-alpha (FedDyn alpha, default 0.1)


	•	Dataset
	• --data-set (default cifar10, choices=[cifar10, mnist])
	•	--data-root (default ./data)
	•	--augment (train-time augmentation)
	•	--normalize / --no-normalize
	•	--test-batch-size (default 128)

	•	Federated learning config
	•	--num-clients (default 10)
	•	--client-frac fraction of clients sampled per round (default 0.25)
	•	--local-epochs (default 1)
	•	--batch-size (default 100)
	•	--lr learning rate (default 1e-2)
	•	--rounds communication rounds (default 10)

	•	Data partitioning
	•	--partition in {iid,niid}
	•	--alpha: Dirichlet concentration parameter controlling Non-IID severity.
		    ├── α = 0.1 ~ 0.3: highly skewed label distribution (strong Non-IID)
		  	├──	α = 0.5: moderate Non-IID (default)
		  	└──	α = 0.8 ~ 1.0: closer to IID
	•	--min-size minimum samples per client in non-IID (default 10)
	•	--print-labels / --no-print-labels

	•	Learning rate Scheduler (ReduceOnPlateau)
	•	--lr-factor learning rate * factor (default 0.5)
	•	--lr-patience (default 5)
	•	--min-lr (deafult 1e-6)
	•	--lr-threshold (default 1e-4)
	•	--lr-cooldown (default 0)
```

Notes on Implementation
```
	•	Client training (fl/fedavg.py, fl/fedprox.py, fl/scaffold.py)
	•	Uses SGD with momentum=0.9 and weight decay=5e-4 (In scaffold, no momentum and weight decay)
	•	Returns the local state_dict moved to CPU (for aggregation)
	•	Server aggregation (fl/server.py)
	•	Weighted average of parameters using client dataset sizes
	•	Non-IID partitioning (data/partition.py)
	•	Uses a Dirichlet distribution per class across clients
	•	Includes a safety loop to ensure each client has at least min_size samples
```

## Expected Output

Each round prints evaluation results like:
```bash
=== Evaluate global model 1 Round ===
[01] acc=XX.XX%, loss=Y.YYYYYY
=====================================
```

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
