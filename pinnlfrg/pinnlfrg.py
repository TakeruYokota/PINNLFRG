from parameter_load import parameter_load
from output_tensorboard import output_tensorboard
from nn import NN
import os
from loss import DELoss
from solver import solver
import torch
import random
import numpy as np


def torch_fix_seed(seed=57):
    """
    Fixing the seed for random numbers
    """
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def main():
    p = parameter_load()
    torch_fix_seed(p.rseed)

    # Getting device information
    device = "cpu"
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Using {device} device (#GPU: {n_gpu})")
    for i in range(n_gpu):
        print(f"{i}: {torch.cuda.get_device_name(i)}")

    device_id = device if device == "cpu" else f"{device}:0"
    print(device_id)

    # Model setting
    # Define an NN architecture in nn.py and load here
    #   model = NN_example(...)
    model = NN(p.Nphi, p.m2, p.use_xavier,
               Nlayer=p.Nlayer, Nnode=p.Nnode, std=p.std)
    if device == "cuda":
        model.to(device)

    # Optimizer setting
    optimizer = torch.optim.Adam(model.parameters(), lr=p.lr)

    # Initial iteration number
    istart = 1

    # Check point loading
    cpt = None
    if p.load_cpt == 1:
        if os.path.isfile(p.cptpath_load):
            cpt = torch.load(p.cptpath_load)
            print("Check point loaded (filepath: {})".format(p.cptpath_load))
        else:
            print("Check point unloaded (file {} does not exist)".format(
                p.cptpath_load))

    if cpt is not None:
        model.load_state_dict(cpt['model_state_dict'])
        optimizer.load_state_dict(cpt['opt_state_dict'])
        if device == "cuda":
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to('cuda')
        istart = cpt['iter']

    # Setting of loss function
    loss_fn = DELoss(p.Nphi, p.lend, p.m2, p.g4,
                     p.Ncol, Ncol_pre=p.Ncol_pre,
                     std_col_mul=p.std_col_mul,
                     std_col_mul_pre=p.std_col_mul_pre,
                     device=device_id)

    # Setting for tensorboard
    if p.out_tensorboard == 1:
        ot = output_tensorboard(loss_fn, p.lend, p.p0, p.p1,
                                p.Nphi, p.m2, p.g4, device=device_id)

    # Setting for solver
    s = solver(model, loss_fn, optimizer, ot)

    # Pretraining
    if p.Niter_pre > 0 and p.Ncol_pre > 0:
        s.solve_eq(p.Niter_pre, p.Nprint, p.Nplot, p.Nsave, p.cptpath_pre,
                   1, p.lr_pre, ispre=1)

    # Training for PDE
    if p.Niter > 0:
        s.solve_eq(p.Niter, p.Nprint, p.Nplot, p.Nsave, p.cptpath_save,
                   istart, p.lr, lr_decay=p.lr_decay)


if __name__ == '__main__':
    main()
