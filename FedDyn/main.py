import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from data import mnist
from data import cifar10
from data.partition import IID_partition, NIID_partition, print_label_counts
from fl.feddyn import init_feddyn_states, get_global_theta_prev, local_train_feddyn, compute_delta, update_feddyn_local
from fl.server import update_h_feddyn, server_train_feddyn, load_state_dict_feddyn
from models.mobilenet import MobileNet
from utils.seed import set_seed
from utils.eval import eval
from utils.parser import parse_args
from utils.device import select_device

def main():
    args = parse_args()
    set_seed(args.seed, True)
    rng = np.random.default_rng(args.seed)

    device = select_device(args.device)
    print(f"Device => {device}")

    if args.data_set == "cifar10":
        train_ds, test_ds = cifar10.get_cifar10(root=args.data_root, normalize=args.normalize, augment=args.augment)
        global_model = MobileNet(num_classes=10, in_channel=3).to(device)
    else:
        train_ds, test_ds = mnist.get_mnist(root=args.data_root, normalize=args.normalize, augment=args.augment)
        global_model = MobileNet(num_classes=10, in_channel=1).to(device)


    test_loader = DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False)
    
    lr_holder = nn.Parameter(torch.zeros(1, device=device), requires_grad=True)
    lr_opt = torch.optim.SGD([lr_holder], lr=args.lr)

    scheduler = ReduceLROnPlateau(
        lr_opt,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.lr_threshold,
        cooldown=args.lr_cooldown,
        min_lr=args.min_lr,
    )
    current_lr = lr_opt.param_groups[0]["lr"]

    if args.partition == "niid":
        clients = NIID_partition(
            train_ds, num_clients=args.num_clients, seed=args.seed,
            alpha=args.alpha, min_size=args.min_size
        )
    else:
        clients = IID_partition(train_ds, num_clients=args.num_clients, seed=args.seed)

    if args.print_labels:
        print("\n=== Client label distributions ===")
        print_label_counts(train_ds, clients, num_classes=10)

    m = max(1, int(args.client_frac * args.num_clients))

    server_h, client_thetas, client_gs = init_feddyn_states(global_model, num_clients=args.num_clients)

    for r in range(args.rounds):
        selected = rng.choice(args.num_clients, size=m, replace=False)

        global_theta_prev = get_global_theta_prev(global_model)

        client_state_dicts = []
        theta_k_list = []
        deltas_list = []

        for cid in selected:
            local_state_dict, new_theta_k = local_train_feddyn(
                global_model=global_model,
                client_dataset=clients[cid],
                g_k=client_gs[cid],
                theta_prev=client_thetas[cid],
                alpha=args.dyn_alpha,
                epochs=args.local_epochs,
                batch_size=args.batch_size,
                lr=current_lr,
                device=device,
                momentum=0.9,
                weight_decay=5e-4)
            
            local_delta = compute_delta(global_theta_prev, new_theta_k)

            new_g_k = update_feddyn_local(client_gs[cid], local_delta, alpha=args.dyn_alpha)

            client_gs[cid] = new_g_k
            client_thetas[cid] = new_theta_k

            client_state_dicts.append(local_state_dict)
            theta_k_list.append(new_theta_k)
            deltas_list.append(local_delta)
            
        server_h = update_h_feddyn(global_theta_prev, deltas_list, server_h, alpha=args.dyn_alpha, num_clients=args.num_clients)

        new_global_theta = server_train_feddyn(global_model, server_h, theta_k_list, alpha=args.dyn_alpha)

        global_model = load_state_dict_feddyn(global_model, new_global_theta)

        # eval
        print(f"\n=== Evaluate global model Round {r + 1} ===")
        acc, loss = eval(global_model, test_loader, device=device, verbose=False)
        print(f"[{r+1:02d}] acc={acc*100:.2f}%, loss={loss:.6f}")

        # scheduler
        prev_lr = current_lr
        scheduler.step(loss) 
        current_lr = lr_opt.param_groups[0]["lr"]
        if current_lr != prev_lr:
            print(f"--> LR reduced: {prev_lr:.6g} -> {current_lr:.6g}")

if __name__ == "__main__":
    main()