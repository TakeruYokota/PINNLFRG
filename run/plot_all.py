import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy
from scipy import integrate
from scipy.optimize import fsolve

#############################
# input parameters
#############################
path_to_model_def = "../pinnlfrg"  # Directory containing nn.py
cptpath = './out.cpt'
Nphi_ = 1
m2_ = 1e-2
g4_ = 1e-4
Nlayer_ = 4
Nnode_ = 256
#############################

#############################
# Importing model definition
#############################
if os.path.exists(path_to_model_def):
    sys.path.append(path_to_model_def)
    from nn import NN
    NN_def = NN
else:
    print(r"{} does not exist.".format(path_to_model_def))
    quit()
#############################

#############################
# Parameters for outputs
#############################
figname = 'fig_all.pdf'
Method_name = 'PINN-LFRG'
label_size = 13
legend_font_size = 10
#############################


def main():
    device = "cpu"
    print(f"Using {device} device")

    model = NN_def(Nphi_, m2_, Nlayer=Nlayer_, Nnode=Nnode_)
    if os.path.isfile(cptpath):
        cpt = torch.load(cptpath, map_location=torch.device(device))
        model.load_state_dict(cpt['model_state_dict'])
        print("Model loaded (filepath: {})".format(cptpath))
    else:
        print("Model unloaded (file {} does not exist)".format(cptpath))

    model.to(device)
    print(model)

    plot_all(model, Nphi_, g4_, m2_, device=device)
    rel_errors_gam(model, Nphi_, g4_, m2_, device=device)
    print()
    rel_errors_sigma(model, Nphi_, g4_, m2_, device=device)


def rel_errors_gam(model, Nphi, g4, m2, device='cpu'):
    lstart = 5.
    lend = 5.
    p0 = 0
    Nlmesh = 1

    l_s = torch.linspace(lstart, lend, Nlmesh, device=device).view(-1, 1)

    gam_exact = gamint_exact(
        Nphi, g4, m2+np.exp(-2.*lend), p0**2) - gamint_exact(Nphi, g4, m2+1., p0**2)
    gam_perturb = gamint_perturb_O1(
        Nphi, g4, m2+np.exp(-2.*lend), p0**2) - gamint_perturb_O1(Nphi, g4, m2+1., p0**2)
    gam_largeN = gamint_largeN(
        Nphi, g4, m2+np.exp(-2.*lend), p0**2)-gamint_largeN(Nphi, g4, m2+1., p0**2)

    p0_ext = p0*torch.ones(Nlmesh, Nphi, device=device)
    input = torch.cat((l_s, p0_ext), dim=1)
    gam_NN = model(input).cpu().detach().numpy().copy().squeeze()

    gam_perturb_rel = (gam_perturb-gam_exact)/gam_exact
    gam_largeN_rel = (gam_largeN-gam_exact)/gam_exact
    gam_NN_rel = (gam_NN-gam_exact)/gam_exact

    print(r'gam_perturb: {}'.format(gam_perturb_rel))
    print(r'gam_largeN : {}'.format(gam_largeN_rel))
    print(r'gam_NN     : {}'.format(gam_NN_rel))


def rel_errors_sigma(model, Nphi, g4, m2, device='cpu'):
    lstart = 5.
    lend = 5.
    p0 = 0
    Nlmesh = 1

    l_s = torch.linspace(lstart, lend, Nlmesh, device=device).view(-1, 1)

    res_dsigma_exact = dsigma_exact(
        Nphi, g4, m2+np.exp(-2.*lend), m2, p0**2)-dsigma_exact(Nphi, g4, m2+1., m2, p0**2)
    res_dsigma_perturb = dsigma_perturb_O1(
        Nphi, g4, m2+np.exp(-2.*lend), m2, p0**2)-dsigma_perturb_O1(Nphi, g4, m2+1., m2, p0**2)
    res_dsigma_largeN = dsigma_largeN(
        Nphi, g4, m2+np.exp(-2.*lend), m2)-dsigma_largeN(Nphi, g4, m2+1., m2)

    p_s = p0*torch.ones(Nlmesh, Nphi, device=device)

    def model_lp(_l, _p):
        input = torch.cat((_l, _p), dim=0).view(1, 1+Nphi)
        return model(input)[0][0]
    compute_batch_hessian = torch.vmap(
        torch.func.hessian(model_lp, argnums=1), in_dims=(0, 0))
    dsigma_pp = compute_batch_hessian(l_s, p_s)
    dsigma_pp_nn = torch.diagonal(dsigma_pp, dim1=-2, dim2=-1).cpu(
    ).detach().numpy().copy().squeeze()/m2

    dsigma_NN_ave = np.average(dsigma_pp_nn)
    dsigma_NN_std = np.std(dsigma_pp_nn)

    dsigma_perturb_rel = (res_dsigma_perturb-res_dsigma_exact)/res_dsigma_exact
    dsigma_largeN_rel = (res_dsigma_largeN-res_dsigma_exact)/res_dsigma_exact
    dsigma_NN_rel = (dsigma_NN_ave-res_dsigma_exact)/res_dsigma_exact
    dsigma_NN_std_rel = dsigma_NN_std/res_dsigma_exact

    print(r'dsigma_perturb: {}'.format(dsigma_perturb_rel))
    print(r'dsigma_largeN : {}'.format(dsigma_largeN_rel))
    print(r'dsigma_NN_ave : {}'.format(dsigma_NN_rel))
    print(r'dsigma_NN_std : {}'.format(dsigma_NN_std_rel))


def plot_all(model, Nphi, g4, m2, device='cpu'):

    fig = plt.figure(figsize=(5, 8))
    gs_out = fig.add_gridspec(2, 1, hspace=0.25)
    for i in range(2):
        gs_in = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                 subplot_spec=gs_out[i], hspace=0)
        axs = gs_in.subplots(sharex="col")
        axs[0].axis("off")
        axs[1].axis("off")
        ax0 = fig.add_subplot(gs_in[0])
        ax1 = fig.add_subplot(gs_in[1])
        if i == 0:
            axs[0].text(-0.15, 1, '(a)', fontsize=12)
            plot_l(ax0, model, Nphi, g4, m2, device=device)
            plot_sigma_l(ax1, model, Nphi, g4, m2, device=device)
        else:
            axs[0].text(-0.15, 1, '(b)', fontsize=12)
            plot_p(ax0, model, Nphi, g4, m2, device=device)
            plot_sigma_p(ax1, model, Nphi, g4, m2, device=device)

    plt.tick_params(labelsize=10)
    plt.tight_layout()
    fig.align_ylabels()
    plt.savefig(figname, format='pdf', transparent=True, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_l(ax, model, Nphi, g4, m2, device='cpu'):
    lstart = 0
    lend = 5.
    p0 = 0
    Nlmesh = 100

    l_s = torch.linspace(lstart, lend, Nlmesh, device=device).view(-1, 1)
    l_s_cpu = l_s.cpu().detach().numpy().copy().squeeze()

    gam_exact = np.array([gamint_exact(
        Nphi, g4, m2+np.exp(-2.*l), p0**2) - g4*(p0**4)/24. for l in l_s_cpu])
    gam_exact -= gamint_exact(Nphi, g4, m2+1., p0**2) - g4*(p0**4)/24.
    gam_perturb = np.array([gamint_perturb_O1(
        Nphi, g4, m2+np.exp(-2.*l), p0**2) - g4*(p0**4)/24. for l in l_s_cpu])
    gam_perturb -= gamint_perturb_O1(Nphi, g4, m2+1., p0**2) - g4*(p0**4)/24.
    gam_largeN = np.array(
        [gamint_largeN(Nphi, g4, m2+np.exp(-2.*l), p0**2) for l in l_s_cpu])
    gam_largeN -= gamint_largeN(Nphi, g4, m2+1., p0**2)

    p0_ext = p0*torch.ones(Nlmesh, Nphi, device=device)
    input = torch.cat((l_s, p0_ext), dim=1)
    gam_NN = model(input).cpu().detach().numpy().copy().squeeze()

    ax.plot(l_s_cpu, gam_NN, linestyle='-', label=Method_name, color='red')
    ax.plot(l_s_cpu, gam_exact, linestyle='--', label=r'Exact', color='black')
    ax.plot(l_s_cpu, gam_perturb, linestyle='-.',
            label=r'Perturbation', color='green')
    ax.plot(l_s_cpu, gam_largeN, linestyle=':', label=r'Large-N', color='blue')
    ax.axes.xaxis.set_visible(False)
    ax.set_ylabel(r'$\gamma(l,0)$', fontsize=label_size)
    ax.set_xlim([lstart, lend])
    ax.legend(frameon=False, fontsize=legend_font_size,
              loc="best", numpoints=1)


def plot_p(ax, model, Nphi, g4, m2, device='cpu'):
    l0 = 5.
    p0 = 0
    pstart = -10.
    pend = 10.
    Npmesh = 100
    m = m2**0.5

    l0_s = l0*torch.ones(Npmesh, device=device).view(-1, 1)
    p_s = torch.linspace(pstart, pend, Npmesh, device=device)
    p_s_cpu = p_s.cpu().detach().numpy().copy().squeeze()

    ml2 = m2+np.exp(-2.*l0)
    gam_exact = np.array([gamint_exact(Nphi, g4, ml2, p*p) -
                          g4*(p**4)/24. for p in p_s_cpu])
    gam_exact -= np.array([gamint_exact(Nphi, g4, m2+1., p*p) -
                          g4*(p**4)/24. for p in p_s_cpu])
    gam_perturb = np.array([gamint_perturb_O1(Nphi, g4, ml2, p*p) -
                            g4*(p**4)/24. for p in p_s_cpu])
    gam_perturb -= np.array([gamint_perturb_O1(Nphi, g4, m2+1., p*p) -
                            g4*(p**4)/24. for p in p_s_cpu])
    gam_largeN = np.array([gamint_largeN(Nphi, g4, ml2, p*p) for p in p_s_cpu])
    gam_largeN -= np.array([gamint_largeN(Nphi, g4, m2+1., p*p)
                           for p in p_s_cpu])

    gam_NN = []

    for ntarget in range(Nphi):
        p0_u = p0*torch.ones(Npmesh, ntarget, device=device)
        p0_d = p0*torch.ones(Npmesh, Nphi - ntarget - 1, device=device)

        input = torch.cat((l0_s, p0_u), dim=1)
        input = torch.cat((input, p_s.view(-1, 1)), dim=1)
        input = torch.cat((input, p0_d), dim=1)

        gam_NN.append(model(input).cpu().detach().numpy().copy().squeeze())

    gam_NN = np.array(gam_NN)
    gam_NN = np.transpose(gam_NN)

    ax.plot([], [], linestyle='-', label=Method_name, color='red', lw=1)
    ax.plot(m*p_s_cpu, gam_NN, linestyle='-', label=None, color='red', lw=1)
    ax.plot(m*p_s_cpu, gam_exact, linestyle='--', label='Exact', color='black')
    ax.plot(m*p_s_cpu, gam_largeN, linestyle=':',
            label='Large N (leading)', color='blue')
    ax.plot(m*p_s_cpu, gam_perturb, linestyle='-.',
            label='Perturbation (leading)', color='green')
    ax.axes.xaxis.set_visible(False)
    ax.set_ylabel(r'$\gamma(l_{\rm end}, \varphi)$', fontsize=label_size)
    ax.set_xlim([m*pstart, m*pend])


def plot_sigma_l(ax, model, Nphi, g4, m2, device='cpu'):
    lstart = 0
    lend = 5.
    p0 = 0
    Nlmesh = 100

    l_s = torch.linspace(lstart, lend, Nlmesh, device=device).view(-1, 1)
    l_s_cpu = l_s.cpu().detach().numpy().copy().squeeze()

    res_dsigma_exact = np.array(
        [dsigma_exact(Nphi, g4, m2+np.exp(-2.*l), m2, p0**2) for l in l_s_cpu])
    res_dsigma_exact -= dsigma_exact(Nphi, g4, m2+1., m2, p0**2)
    res_dsigma_perturb = np.array(
        [dsigma_perturb_O1(Nphi, g4, m2+np.exp(-2.*l), m2, p0**2) for l in l_s_cpu])
    res_dsigma_perturb -= dsigma_perturb_O1(Nphi, g4, m2+1., m2, p0**2)
    res_dsigma_largeN = np.array(
        [dsigma_largeN(Nphi, g4, m2+np.exp(-2.*l), m2) for l in l_s_cpu])
    res_dsigma_largeN -= dsigma_largeN(Nphi, g4, m2+1., m2)

    p_s = p0*torch.ones(Nlmesh, Nphi, device=device)

    def model_lp(_l, _p):
        input = torch.cat((_l, _p), dim=0).view(1, 1+Nphi)
        return model(input)[0][0]
    compute_batch_hessian = torch.vmap(
        torch.func.hessian(model_lp, argnums=1), in_dims=(0, 0))
    dsigma_pp = compute_batch_hessian(l_s, p_s)
    dsigma_pp_nn = torch.diagonal(dsigma_pp, dim1=-2, dim2=-1).cpu(
    ).detach().numpy().copy().squeeze()/m2

    ax.plot([], [], linestyle='-', label=Method_name, color='red', lw=1)
    ax.plot(l_s_cpu, dsigma_pp_nn, linestyle='-',
            label=None, color='red', lw=1)
    ax.plot(l_s_cpu, res_dsigma_exact, linestyle='--',
            label='Exact', color='black')
    ax.plot(l_s_cpu, res_dsigma_largeN, linestyle=':',
            label='Large N (leading)', color='blue')
    ax.plot(l_s_cpu, res_dsigma_perturb, linestyle='-.',
            label='Perturbation (leading)', color='green')
    ax.set_xlabel(r'$l$', fontsize=label_size)
    ax.set_ylabel(r'$\sigma(l, 0)/m^2$', fontsize=label_size)
    ax.set_xlim([lstart, lend])


def plot_sigma_p(ax, model, Nphi, g4, m2, device='cpu'):
    l0 = 5.
    p0 = 0
    pstart = -10.
    pend = 10.
    Npmesh = 100
    m = m2**0.5

    l0_s = l0*torch.ones(Npmesh, device=device).view(-1, 1)
    p_s = torch.linspace(pstart, pend, Npmesh, device=device)
    p_s_cpu = p_s.cpu().detach().numpy().copy().squeeze()

    ml2 = m2+np.exp(-2.*l0)
    res_dsigma_exact = np.array(
        [dsigma_exact(Nphi, g4, ml2, m2, p*p) for p in p_s_cpu])
    res_dsigma_exact -= np.array([dsigma_exact(Nphi,
                                 g4, m2+1., m2, p*p) for p in p_s_cpu])
    res_dsigma_perturb = np.array(
        [dsigma_perturb_O1(Nphi, g4, ml2, m2, p*p) for p in p_s_cpu])
    res_dsigma_perturb -= np.array([dsigma_perturb_O1(Nphi,
                                   g4, m2+1., m2, p*p) for p in p_s_cpu])
    res_dsigma_largeN = np.array(
        [dsigma_largeN(Nphi, g4, ml2, m2) for p in p_s_cpu])
    res_dsigma_largeN -= np.array([dsigma_largeN(Nphi, g4, m2+1., m2)
                                  for p in p_s_cpu])

    dsigma_NN = []

    for ntarget in range(Nphi):
        p0_u = p0*torch.ones(Npmesh, ntarget, device=device)
        p0_d = p0*torch.ones(Npmesh, Nphi - ntarget - 1, device=device)
        p_bc = p0_u
        p_bc = torch.cat((p_bc, p_s.view(-1, 1)), dim=1)
        p_bc = torch.cat((p_bc, p0_d), dim=1)

        def model_lp(_l, _p):
            input = torch.cat((_l, _p), dim=0).view(1, 1+Nphi)
            return model(input)[0][0]
        compute_batch_hessian = torch.vmap(
            torch.func.hessian(model_lp, argnums=1), in_dims=(0, 0))
        dsigma_pp = compute_batch_hessian(l0_s, p_bc)
        dsigma_pp_nn = torch.diagonal(
            dsigma_pp, dim1=-2, dim2=-1).cpu().detach().numpy().copy().squeeze()
        if Nphi > 1:
            dsigma_pp_nn = dsigma_pp_nn[:, ntarget]
        dsigma_NN.append(dsigma_pp_nn/m2)

    dsigma_NN = np.array(dsigma_NN)
    dsigma_NN = np.transpose(dsigma_NN)

    ax.plot([], [], linestyle='-', label=Method_name, color='red', lw=1)
    ax.plot(m*p_s_cpu, dsigma_NN, linestyle='-', label=None, color='red', lw=1)
    ax.plot(m*p_s_cpu, res_dsigma_exact, linestyle='--',
            label='Exact', color='black')
    ax.plot(m*p_s_cpu, res_dsigma_largeN, linestyle=':',
            label='Large N (leading)', color='blue')
    ax.plot(m*p_s_cpu, res_dsigma_perturb, linestyle='-.',
            label='Perturbation (leading)', color='green')
    ax.set_xlabel(r'$m\varphi$', fontsize=label_size)
    ax.set_ylabel(r'$\sigma(l_{\rm end}, \varphi)/m^2$', fontsize=label_size)
    ax.set_xlim([m*pstart, m*pend])


def LN(N, h, r):
    if h == 0.:
        return 1./pow(1.+r, 0.5*N)

    x = 1.5*(1.+r)**2/h
    return pow(x, 0.25*N)*scipy.special.hyperu(0.25*N, 0.5, x)/pow(1.+r, 0.5*N)


def calc_Djsup(N, gm, mphi):
    def eff_weight(x, Dj):
        Dx = x - mphi
        return np.exp(-0.5*Dx**2-gm*x**4/24.+Dj*Dx)*LN(N-1, gm, gm*(x**2)/6.)

    def Dj_equation(j):
        def integrand_num(x):
            return x * eff_weight(x, j)

        def integrand_deno(x):
            return eff_weight(x, j)

        return mphi - integrate.quad(integrand_num, -np.inf, np.inf)[0] / integrate.quad(integrand_deno, -np.inf, np.inf)[0]

    return fsolve(Dj_equation, np.array([0.]))[0]


def gamint_exact(N, g, ml2, phi2):
    gm = g/(ml2*ml2)
    mphi2 = ml2*phi2
    mphi = np.sqrt(mphi2)
    Djsup = calc_Djsup(N, gm, mphi)

    def integrand(x):
        Dx = x-mphi
        return np.exp(-0.5*Dx**2-gm*x**4/24.+Djsup*Dx)*LN(N-1, gm, gm*(x**2)/6.)

    Z = integrate.quad(integrand, -np.inf, np.inf)[0]
    return -np.log(Z/np.sqrt(2.*np.pi))


def dsigma_exact(N, g, ml2, m2, phi2):
    gm = g/(ml2*ml2)
    mphi2 = ml2*phi2
    mphi = np.sqrt(mphi2)
    Djsup = calc_Djsup(N, gm, mphi)

    def integrand_num(x):
        Dx = x-mphi
        return np.exp(-0.5*Dx**2-gm*x**4/24.+Djsup*Dx)*LN(N-1, gm, gm*(x**2)/6.)

    def integrand_deno(x):
        Dx = x-mphi
        return Dx**2*np.exp(-0.5*Dx**2-gm*x**4/24.+Djsup*Dx)*LN(N-1, gm, gm*(x**2)/6.)

    Gi = integrate.quad(integrand_num, -np.inf,
                        np.inf)[0]/integrate.quad(integrand_deno, -np.inf, np.inf)[0]
    return (ml2/m2)*(Gi-1.)-g*phi2/(2.*m2)


def gamint_exact0(N, g, ml2):
    return -np.log(LN(g/ml2**2, 0)/LN(N, 0., 0.))


def dsigma_exact0(N, g, ml2, m2):
    return (LN(N, g/ml2**2, 0.)/LN(N+2, g/ml2**2, 0.)-1.)*ml2/m2


def gamint_largeN(N, g, ml2, phi2):
    gm = N*g/(1.5*ml2**2)
    denom = 1./(1.+np.sqrt(1.+gm))
    zl = 2.*denom

    t1 = 0.25*ml2*gm*denom
    t2 = N*(-0.25*gm*denom**2-0.5*np.log(zl))

    return t1*phi2+t2


def dsigma_largeN(N, g, ml2, m2):
    gm = N*g/(1.5*ml2**2)
    denom = 1./(1.+np.sqrt(1.+gm))
    return 0.5*gm*denom*ml2/m2


def gamint_perturb_O1(N, g, ml2, phi2):
    gm = N*g/(ml2**2)
    Ni = 1./N
    t1 = N*gm*(1.+2.*Ni)/24.
    t2 = 0.5*ml2*gm*(1.+2.*Ni)/6.
    t3 = g/24.
    return t1+t2*phi2+t3*phi2**2


def gamint_perturb_O2(N, g, ml2, phi2):
    gm = N*g/(ml2**2)
    Ni = 1./N
    t1 = N*(gm*(1.+2.*Ni)/24.-gm**2*(1.+5.*Ni+6.*Ni**2)/144.)
    t2 = 0.5*ml2*(gm*(1.+2.*Ni)/6.-gm**2*(1.+6.*Ni+8.*Ni**2)/36.)
    t3 = (g/24.)*(1.-gm*(1.+8.*Ni)/6.)
    return t1+t2*phi2+t3*phi2**2


def dsigma_perturb_O1(N, g, ml2, m2, phi2):
    gm = N*g/(ml2**2)
    Ni = 1./N
    t1 = gm*(1.+2.*Ni)/6.
    return (ml2/m2)*t1


def dsigma_perturb_O2(N, g, ml2, m2, phi2):
    gm = N*g/(ml2**4)
    Ni = 1./N
    t1 = gm*(1.+2.*Ni)/6.-gm**2*(1.+6.*Ni+8.*Ni**2)/36.
    t2 = 3.*phi2*(-gm**2*(1.+8.*Ni)/6.)*(ml2*Ni)/12.
    return (ml2/m2)*(t1+t2)


if __name__ == '__main__':
    main()
