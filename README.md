
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
python main.py --partition niid --alpha 0.5 --min-size 10
```
Example3: Change the number and ratio of participating clients + local epoch
```bash
python main.py --num-clients 100 --client-frac 0.2 --local-epochs 5
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
## FedDyn Implementation Notes

### 1) Client-side Update (fl/feddyn.py)

Each client minimizes a dynamically regularized objective to reduce client drift from the global optimum.

**Local objective (per client):**

$$𝝷_k^t = L_{total}(𝝷) - {\<g_k^{t-1}, 𝝷\>} + \frac{\alpha}{2} * |\theta-\theta^{t-1}\|^2$$

- $L_{\text{task}}$: standard cross-entropy loss on local batch $b$.
- $-\langle 𝝷_k^{t}, \theta \rangle$: linear correction term using the client-specific state $h_k^t$.
- $\frac{\alpha}{2}\|\theta-\theta^{t}\|^2$: proximal term keeping the local model close to the global model $\theta^t$.

**Optimizer:** SGD with `momentum=0.9`, `weight_decay=5e-4`.

**Client state update (after local training):**

$$
g_k^{t} = g_k^{t-1} - \alpha(\theta_k^{t}-\theta^{t-1})
$$

where $\theta_k^{t+1}$ is the client model after local training and $\theta^{t}$ is the global model received at the start of round $t$.

⸻

2) Server-side Aggregation (fl/server.py)

The server maintains a global correction state $h$ and updates the global model using a corrected averaging scheme.

(a) Server state $h$ update:
$$h^{t} = h^{t-1} - \alpha \cdot \frac{1}{m}\sum_{k\in P_i}(\theta_k^{t}-\theta^{t-1})$$<br>
	•	$m$: Number of all clients<br>
	•	The server state $$h$$ accumulates the average drift $$(\theta_k^{t}-\theta^{t-1})$$ across every participating clients.

(b) Global model update
For learnable parameters (weights/bias):

$$\\overline{\theta^{t}} = \frac{1}{P}\sum_{k\in P_i}\theta_k^{t}$$

$$\theta^t = \\overline{\theta^{t}} - \frac{1}{\alpha}h^{t}$$

For BatchNorm buffers (e.g., running_mean, running_var, num_batches_tracked):

$$\theta^{t} = \\overline{\theta^{t}}$$

BatchNorm buffers are aggregated by simple averaging (no FedDyn correction).

## Expected Output

Each round prints evaluation results like:
```bash
=== Evaluate global model 1 Round ===
[01] acc=XX.XX%, loss=Y.YYYYYY
=====================================
```
