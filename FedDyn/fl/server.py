import torch

@torch.no_grad()
def update_h_feddyn(global_theta_prev, deltas, h_prev, alpha, num_clients):
    m = float(num_clients)

    avg_delta = {name: torch.zeros_like(p) for name, p in global_theta_prev.items()}
    
    for delta in deltas:
        for name, p in delta.items():
            avg_delta[name].add_(p)

    h_t = {}
    for name in h_prev.keys():
        h_t[name] = h_prev[name] - (alpha * (avg_delta[name] / m)) 

    return h_t

@torch.no_grad()
def server_train_feddyn(global_model, h, thetas, alpha):
    p = len(thetas)
    learnable_params = [n for n, _ in global_model.named_parameters()]
    
    sum_theta = {name: torch.zeros_like(param) for name, param in thetas[0].items()}
    for theta in thetas:
        for name, param in theta.items():
            sum_theta[name].add_(param)

    new_global_theta = {}
    for name in sum_theta.keys():
        avg_theta = sum_theta[name] / float(p)
        
        # FedDyn Server update (without BN)
        if name in learnable_params:
            new_global_theta[name] = avg_theta - (1 / alpha * h[name])
        else:
            new_global_theta[name] = avg_theta

    return new_global_theta

@torch.no_grad()
def load_state_dict_feddyn(global_model, new_global_theta):
    state = global_model.state_dict()

    for k, v in new_global_theta.items():
        if k in state:
            state[k] = v.to(state[k].device)

    global_model.load_state_dict(state)

    return global_model