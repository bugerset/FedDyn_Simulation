
# FedDyn on CIFAR-10 and MNIST with MobileNet (PyTorch)

This repository provides a PyTorch implementation of FedDyn (Federated Learning based on Dynamic Regularization) using MobileNet on CIFAR-10 and MNIST datasets. 
It is designed to handle Non-IID data distributions effectively by introducing dynamic regularization to mitigate client drift.

## Project Overview 
FedDyn optimizes the global objective by dynamically updating a regularizer for each client. This simulation covers:

Data Partitioning: IID and Non-IID (Dirichlet distribution) splits.

State Management: Handling server state **$h$** and client states **$g_k​ , θ_k$** for dynamic regularization.

Optimized Aggregation: Server-side update logic that specifically excludes BatchNorm parameters from FedDyn regularization to maintain stability.

## Recommended Folder Structure

Your `main.py` imports modules like `data.cifar10`, `fl.client`, etc.  
So the easiest way to run without changing code is to organize files like this:
```
├── main.py
├── data/
│   ├── __init__.py
│   ├── cifar10.py
│	├── mnist.py
│   └── partition.py
├── fl/
│   ├── __init__.py
│   ├── feddyn.py
│   └── server.py
├── models/
│   ├── __init__.py
│   └── mobilenet.py
└── utils/
 	├── __init__.py
 	├── device.py
    ├── eval.py
    ├── parser.py
    └── seed.py
```

## Requirements

- Python 3.9+ recommended
- PyTorch + torchvision
- numpy

Run with default settings:
```bash
python main.py
```
Example1: Non-IID
```bash
python main.py --partition niid
```
Example2: Non-IID with control dyn-alpha
```bash
python main.py --train fedavg --partition niid --alpha 0.5 --min-size 10
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
	•   --data-set (default cifar10, choices=[cifar10, mnist])
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
	•	--lr-factor (learning rate * factor, default 0.5)
	•	--lr-patience (default 5)
	•	--min-lr (deafult 1e-6)
	•	--lr-threshold (default 1e-4)
	•	--lr-cooldown (default 0)
```

Notes on Implementation
```
	•	Uses SGD with momentum=0.9 and weight decay=5e-4
	•	Returns the local state_dict moved to CPU
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


BatchNorm Handling
To ensure model stability, the server updates the global model by applying the FedDyn correction only to learnable parameters (weights and biases), while performing simple averaging for BatchNorm buffers (running_mean, running_var).
