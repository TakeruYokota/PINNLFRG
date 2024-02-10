import argparse


def parameter_load():
    ap = argparse.ArgumentParser(
        description='''
        This program calculates the effective action 
        of the zero-dimensional O(N) model with PINN-LFRG method
        (https://doi.org/10.48550/arXiv.2312.16038).

        The action is defined by
        S(vec(phi))=(1/2) m^2 phi^2 + (1/4!) g phi^4.

        The reduced effective action gamma_l = Gamma_l - S_l(phi) - DS_free
        is replaced by an NN. 
        This NN is composed of fully connected layers 
        with softplus activation functions.
        ''')
    ap.add_argument("--lend", type=float, default=5,
                    help='''
                    This sets the IR endpoint of the Wilson RG parameter lend=ln(k_UV / k_IR).

                    (Type: float, Default value: 5)
                    ''')
    ap.add_argument("--p0", type=float, default=0,
                    help='''
                    This sets the range of phi ([p0, p1]) in plots output to tensorboard.

                    (Type: float, Default value: 0)
                    ''')
    ap.add_argument("--p1", type=float, default=1,
                    help='''
                    This sets the range of phi ([p0, p1]) in plots output to tensorboard.

                    (Type: float, Default value: 0)
                    ''')
    ap.add_argument("--std_col_mul", type=float, default=1,
                    help='''
                    This determines the width of the Gaussian distribution
                    for the collocation points |vec(phi)|
                    in the PDE training.
                    The standard deviation of the distribution is given by
                    std_col_mul * (N)^(1/2) / m.
                    
                    (Type: float, Default value: 1)
                    ''')
    ap.add_argument("--std_col_mul_pre", type=float, default=1,
                    help='''
                    This determines the width of the Gaussian distribution
                    for the collocation points |vec(phi)|
                    in the pretraining.
                    The standard of the distribution is given by 
                    std_col_mul_pre * (N)^(1/2) / m.

                    (Type: float, Default value: 1)
                    ''')
    ap.add_argument("--Nphi", type=int, default=1,
                    help='''
                    This sets the number of the components of the scalar field: Nphi=N

                    (Type: int, Default value: 1)                  
                    ''')
    ap.add_argument("--m2", type=float, default=0.01,
                    help='''
                    This sets the mass squared m2.

                    (Type: float, Default value: 0.01)
                    ''')
    ap.add_argument("--g4", type=float, default=0,
                    help='''
                    This sets the coupling g4.

                    (Type: float, Default value: 0)
                    ''')
    ap.add_argument("--Nlayer", type=int, default=3,
                    help='''
                    This sets the number of the hidden layers.

                    (Type: int, Default value: 3)
                    ''')
    ap.add_argument("--Nnode", type=int, default=256,
                    help='''
                    This sets the number of the nodes per hidden layer.

                    (Type: int, Default value: 256)
                    ''')
    ap.add_argument("--std", type=float, default=1e0,
                    help='''
                    This sets the standard deviation of the Gaussian distribution
                    for the initialization of the NN's parameters.

                    (Type: float, Default value: 1)
                    ''')
    ap.add_argument("--use_xavier", type=int, default=1,
                    help='''
                    If this is set to 1, the Xavier initialization 
                    is used for NN.

                    (Type: int, Default value: 1)
                    ''')
    ap.add_argument("--rseed", type=int, default=57,
                    help='''
                    This sets the seed for random numbers.

                    (Type: int, Default value: 57)
                    ''')
    ap.add_argument("--Ncol", type=int, default=1000,
                    help='''
                    This sets the number of the collocation points
                    to evaluate the PDE loss.

                    (Type: int, Default value: 1000)
                    ''')
    ap.add_argument("--Ncol_pre", type=int, default=0,
                    help='''
                    This sets the number of the collocation points
                    to evaluate the loss for pretraining.

                    If this is set to 0, the pretraining is skipped.

                    (Type: int, Default value: 0)
                    ''')
    ap.add_argument("--Niter", type=int, default=10000,
                    help='''
                    This sets the number of the iteration to stop the training.

                    (Type: int, Default value: 10000)
                    ''')
    ap.add_argument("--Niter_pre", type=int, default=0,
                    help='''
                    This sets the number of the iteration to stop the pretraining.

                    If this is set to 0, the pretraining is skipped.

                    (Type: int, Default value: 0)
                    ''')
    ap.add_argument("--Nprint", type=int, default=100,
                    help='''
                    This sets the interval of the iteration for
                    outputting messages to the standard output and
                    outputting numerical data to tensorboard.

                    (Type: int, Default value: 100)
                    ''')
    ap.add_argument("--Nplot", type=int, default=1000,
                    help='''
                    This sets the interval of the iteration for outputting plots
                    to tensorboard.

                    (Type: int, Default value: 1000)
                    ''')
    ap.add_argument("--Nsave", type=int, default=1000,
                    help='''
                    This sets the interval of the iteration for outputting the check point.

                    (Type: int, Default value: 1000)
                    ''')
    ap.add_argument("--lr", type=float, default=1e-3,
                    help='''
                    This sets the learning rate.

                    (Type: float, Default value: 1e-3)
                    ''')
    ap.add_argument("--lr_pre", type=float, default=1e-3,
                    help='''
                    This sets the learning rate for pretraining.

                    (Type: float, Default value: 1e-3)
                    ''')
    ap.add_argument("--lr_decay", type=float, default=1,
                    help='''
                    This sets the exponential decay rate for the learning rate.

                    This is not applied to pretraining.

                    (Type: float, Default value: 1)
                    ''')
    ap.add_argument("--load_cpt", type=int, default=0,
                    help='''
                    If this is set to 1, the check point 
                    indicated by the path cptpath_load is loaded.

                    (Type: int, Default value: 0)
                    ''')
    ap.add_argument("--cptpath_load", type=str, default='./out.cpt',
                    help='''
                    This sets the path to the check point to load.

                    (Type: str, Default value: ./out.cpt)
                    ''')
    ap.add_argument("--cptpath_save", type=str, default='./out.cpt',
                    help='''
                    This sets the path to the check point to save.

                    (Type: str, Default value: ./out.cpt)
                    ''')
    ap.add_argument("--cptpath_pre", type=str, default='./out.cpt',
                    help='''
                    This sets the path for saving the check point 
                    after pretraining.

                    (Type: str, Default value: ./out.cpt)
                    ''')
    ap.add_argument("--out_tensorboard", type=int, default=0,
                    help='''
                    If this is set to 1,
                    the results are output to tensorboard.

                    (Type: str, Default value: 0)
                    ''')

    return ap.parse_args()
