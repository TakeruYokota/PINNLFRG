# PINN-LFRG for 0D O(N) model

Pytorch implementation of PINN-LFRG solver for the zero-dimensional O(N) model: [arXiv:2312.16038](https://doi.org/10.48550/arXiv.2312.16038).

The renormalization-group-induced effective action of the effective action of 0D O(N) model is represented by an NN. The physics-informed neural network (PINN) is used to derive the solution of the Wetterich equation.

## Usage
To start the training, execute the Python script:
```
python3 pinnlfrg/pinnlfrg.py
```
Parameters in the calculation are given as optional arguments. Please take a look at the help option.
```
python3 pinnlfrg/pinnlfrg.py -h
```
In the default setting, this code generates the checkpoint file `out.cpt`, which contains the resultant learning parameters and log files for `tensorboard`.

The script `run/plot_all.py` visualizes the results of the effective action and the self-energy from the checkpoint file `out.cpt`.

## Quickstart
We provide some additional codes to make it easy for you to get started
1. Run `testrun.sh` 
2. Move to `_testrun`
3. Run `run.sh`
   1. The training is executed. This may take a few minutes.
   2. You can see the training progress using `tensorboard`.
4. For visualizing the results, run `python3 polt_all.py`

Alternatively, you can do the same without the shell script.
1. Make a directory named `_testrun` and copy the contents in `run` into `_testrun`
2. Move to `_testrun`
3. Run the following:
   ```
   python3 ../pinnlfrg/pinnlfrg.py --Nphi 1 --m2 1e-2 --g4 1e-4 --Nlayer 4 --Nnode 256 \
   --Ncol 500 --Ncol_pre 500 --rseed 57 --Niter 1000 --Niter_pre 10000 \
   --Nprint 100 --Nplot 100 --Nsave 100 --lr 1e-3 --lr_pre 1e-3 --out_tensorboard 1
   ```
4. For visualizing the results, run `python3 polt_all.py`

## GPU environment
Multi-GPU environments are not supported.

## How to cite
Please cite our paper ([arXiv:2312.16038](https://doi.org/10.48550/arXiv.2312.16038)) if you use this code.
