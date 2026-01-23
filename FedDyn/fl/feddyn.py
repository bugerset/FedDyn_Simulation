import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def init_feddyn_states(global_model, num_clients):
    w_t = {name: p.detach().cpu().clone() for name, p in global_model.state_dict().items()}
    h_t = {name: torch.zeros_like(p) for name, p in w_t.items()}

    theta_k = [{name: p.clone() for name, p in w_t.items()} for _ in range(num_clients)]
    g_k = [{name: torch.zeros_like(p) for name, p in w_t.items()} for _ in range(num_clients)]  
    return h_t, theta_k, g_k

@torch.no_grad()
def get_global_theta_prev(global_model):
    return {name: p.detach().cpu().clone() for name, p in global_model.state_dict().items()}

def local_train_feddyn(global_model, client_dataset, g_k, theta_prev, alpha=0.1, 
                       epochs=1, batch_size=64, lr=1e-3, device="cpu", momentum=0.9, weight_decay=5e-4):
    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    local_model.train()

    loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    g_k_prev = {name: p.detach().to(device) for name, p in g_k.items()}
    theta_k_prev = {name: p.detach().to(device) for name, p in theta_prev.items()}

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = local_model(x)
            loss = criterion(pred, y)

            linear_term = 0.0
            prox_term = 0.0

            for name, p in local_model.named_parameters():
                # Loss is scalar
                linear_term = linear_term + (g_k_prev[name] * p).sum()
                prox_term = prox_term + ((p - theta_k_prev[name]) ** 2).sum()

            total_loss = loss - linear_term + (0.5 * alpha * prox_term)
            total_loss.backward()
            optimizer.step()

    new_theta_k = {name: p.detach().cpu().clone() for name, p in local_model.state_dict().items()}

    return new_theta_k

@torch.no_grad()
def compute_delta(global_theta_prev, local_theta):
    return {name: (local_theta[name] - global_theta_prev[name]).detach().cpu() for name in global_theta_prev.keys()}

@torch.no_grad()
def update_feddyn_local(g_k_prev, delta, alpha=0.1):
    new_g_k = {}
    for name in delta.keys():
        g = g_k_prev[name] - alpha * delta[name]
        new_g_k[name] = g

    return new_g_k
