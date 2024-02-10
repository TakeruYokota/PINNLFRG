import torch
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from scipy import integrate
from scipy.optimize import fsolve
from scipy.special import hyperu
import numpy as np


class output_tensorboard():
    """
    This class offeres functions regarding outputs to tensorboard

    p: phi. the range of p is set to [p0, p1]
    l: Wilson RG parameter
    """

    def __init__(self, loss_fn, lend, p0, p1, Nphi, m2, g4, device='cpu'):
        self.Nlmesh = 50
        self.Npmesh = 50

        self.device = device

        self.Nphi = Nphi
        self.m2 = m2
        self.g4 = g4
        self.g4_3 = g4 / 3.
        self.g4_24 = g4 / 24.
        self.lend = lend
        self.p0 = p0

        self.writer = SummaryWriter()

        self.loss_fn = loss_fn

        pNm1 = p1/(self.m2**0.5)

        self.input_lend_origin = torch.cat(
            (lend*torch.ones((1, 1), device=device), torch.zeros((1, Nphi), device=device)), dim=1)

        self.p_bc_n1 = torch.linspace(-pNm1, pNm1,
                                      self.Npmesh, device=device).view(-1, 1)
        self.p_bc_n1_cpu = self.p_bc_n1.cpu().detach().numpy().copy().squeeze()

        self.p_bc = torch.cat((self.p_bc_n1, torch.zeros(
            self.Npmesh, Nphi-1, device=device)), dim=1)
        self.p_bc.requires_grad_()
        self.p0_bc = p0 * torch.ones(self.Npmesh, Nphi, device=device)
        self.p0_bc.requires_grad_()

        self.l_bc = torch.linspace(
            0., lend, self.Nlmesh, device=device).view(-1, 1)
        self.l_bc.requires_grad_()
        self.l1_bc = lend * torch.ones(self.Nlmesh, device=device).view(-1, 1)
        self.l1_bc.requires_grad_()

        self.ones_l = torch.ones(self.l_bc.size(), device=device)
        self.ones_p = torch.ones(self.p_bc.size(), device=device)
        self.ones_l1 = torch.ones(self.l1_bc.size(), device=device)
        self.ones_p0 = torch.ones(self.p0_bc.size(), device=device)

        self.l_bc_cpu = self.l_bc.cpu().detach().numpy().copy().squeeze()

        self.input_l1_p = torch.cat((self.l1_bc, self.p_bc), dim=1)
        self.input_l_p0 = torch.cat((self.l_bc, self.p0_bc), dim=1)

        self.p0_p0lend = p0 * torch.ones(1, Nphi, device=device)
        self.p0_p0lend.requires_grad_()
        self.l_p0lend = lend*torch.ones(1, device=device).view(-1, 1)
        self.l_p0lend.requires_grad_()
        self.input_lend_p0 = torch.cat((self.l_p0lend, self.p0_p0lend), dim=1)

        input_p_shift_bc_list = []
        for p in self.p_bc_n1_cpu:
            for ntarget in range(Nphi):
                new_batch = [0.]*(Nphi+1)
                new_batch[0] = lend
                new_batch[ntarget+1] = p
                input_p_shift_bc_list.append(new_batch)
        self.input_p_shift_bc = torch.tensor(
            input_p_shift_bc_list, device=device)

        self.Gams_exact_l1_p = None
        self.Gams_exact_l_p0 = None
        self.dsigma_exact_l1_p = None
        self.dsigma_exact_l_p0 = None
        Gams_exact_origin = self.__calc_exact_gamma(
            0, lend) - self.__calc_exact_gamma(0, 0)
        self.Gams_exact_l1_p = np.array(
            [self.__calc_exact_gamma(p, lend) - self.__calc_exact_gamma(p, 0) for p in self.p_bc_n1_cpu]) - Gams_exact_origin
        self.Gams_exact_l_p0 = np.array(
            [self.__calc_exact_gamma(p0, l) for l in self.l_bc_cpu]) - self.__calc_exact_gamma(p0, 0)
        self.dsigma_exact_l1_p = np.array(
            [self.__calc_exact_dsigma(p, lend) - self.__calc_exact_dsigma(p, 0) for p in self.p_bc_n1_cpu])
        self.dsigma_exact_l_p0 = np.array(
            [self.__calc_exact_dsigma(p0, l) for l in self.l_bc_cpu]) - self.__calc_exact_dsigma(p0, 0)

        Gams_perturb_origin = self.__calc_perturb_gamma(
            0, lend) - self.__calc_perturb_gamma(0, 0)
        self.Gams_perturb_l1_p = np.array(
            [self.__calc_perturb_gamma(p, lend) - self.__calc_perturb_gamma(p, 0) for p in self.p_bc_n1_cpu]) - Gams_perturb_origin
        self.Gams_perturb_l_p0 = np.array(
            [self.__calc_perturb_gamma(p0, l) for l in self.l_bc_cpu]) - self.__calc_perturb_gamma(p0, 0)
        self.dsigma_perturb_l1_p = np.array(
            [self.__calc_perturb_dsigma(p, lend) - self.__calc_perturb_dsigma(p, 0) for p in self.p_bc_n1_cpu])
        self.dsigma_perturb_l_p0 = np.array(
            [self.__calc_perturb_dsigma(p0, l) for l in self.l_bc_cpu]) - self.__calc_perturb_dsigma(p0, 0)

        Gams_largeN_l1_origin = self.__calc_largeN_gamma(
            0, lend) - self.__calc_largeN_gamma(0, 0)
        self.Gams_largeN_l1_p = np.array(
            [self.__calc_largeN_gamma(p, lend) - self.__calc_largeN_gamma(p, 0) for p in self.p_bc_n1_cpu]) - Gams_largeN_l1_origin
        self.Gams_largeN_l_p0 = np.array(
            [self.__calc_largeN_gamma(p0, l) for l in self.l_bc_cpu]) - self.__calc_largeN_gamma(p0, 0)
        self.dsigma_largeN_l1_p = np.array(
            [self.__calc_largeN_dsigma(p, lend) - self.__calc_largeN_dsigma(p, 0) for p in self.p_bc_n1_cpu])
        self.dsigma_largeN_l_p0 = np.array(
            [self.__calc_largeN_dsigma(p0, l) for l in self.l_bc_cpu]) - self.__calc_largeN_dsigma(p0, 0)

        self.massvec = torch.tensor([m2]*Nphi, device=device)
        self.massmatrix = torch.diag(self.massvec)
        self.eyeNphi = torch.eye(Nphi, device=device)
        self.g4eyeNphi = (g4 / 6.) * self.eyeNphi

    def __del__(self):
        self.writer.close()

    def __calc_exact_gamma(self, p, l):
        """
        Exact calculation of gamma
        """

        def LN(N, h, r):
            if h == 0.:
                return 1./pow(1.+r, 0.5*N)

            x = 1.5*(1.+r)**2/h
            return pow(x, 0.25*N)*hyperu(0.25*N, 0.5, x)/pow(1.+r, 0.5*N)

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

        ml2 = self.m2 + np.exp(-2.*l)

        gm = self.g4/(ml2*ml2)
        mphi2 = ml2*p**2
        mphi = np.sqrt(mphi2)
        Djsup = calc_Djsup(self.Nphi, gm, mphi)

        def integrand(x):
            Dx = x-mphi
            return np.exp(-0.5*Dx**2-gm*x**4/24.+Djsup*Dx)*LN(self.Nphi-1, gm, gm*(x**2)/6.)

        Z = integrate.quad(integrand, -np.inf, np.inf)[0]
        return -np.log(Z/np.sqrt(2.*np.pi)) - self.g4_24*p**4

    def __calc_exact_dsigma(self, p, l):
        """
        Exact calculation of sigma (reduced self-energy)
        """

        def LN(N, h, r):
            if h == 0.:
                return 1./pow(1.+r, 0.5*N)

            x = 1.5*(1.+r)**2/h
            return pow(x, 0.25*N)*hyperu(0.25*N, 0.5, x)/pow(1.+r, 0.5*N)

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

        ml2 = self.m2 + np.exp(-2.*l)

        gm = self.g4/(ml2*ml2)
        mphi2 = ml2*p**2
        mphi = np.sqrt(mphi2)
        Djsup = calc_Djsup(self.Nphi, gm, mphi)

        def integrand_n(x):
            Dx = x-mphi
            return np.exp(-0.5*Dx**2-gm*x**4/24.+Djsup*Dx)*LN(self.Nphi-1, gm, gm*(x**2)/6.)

        def integrand_d(x):
            Dx = x-mphi
            return (Dx**2)*np.exp(-0.5*Dx**2-gm*x**4/24.+Djsup*Dx)*LN(self.Nphi-1, gm, gm*(x**2)/6.)

        Gam2 = integrate.quad(
            integrand_n, -np.inf, np.inf)[0]/integrate.quad(integrand_d, -np.inf, np.inf)[0]
        return (ml2/self.m2)*(Gam2-1.)-self.g4*(p**2)/(2.*self.m2)

    def __calc_perturb_gamma(self, p, l):
        """
        Perturbative result of gamma
        """
        ml2 = self.m2+np.exp(-2.*l)
        gtm = self.Nphi*self.g4/(ml2**2)
        Ni = 1./self.Nphi
        t1 = self.Nphi*gtm*(1.+2.*Ni)/24.
        t2 = 0.5*ml2*gtm*(1.+2.*Ni)/6.
        return t1+t2*p**2

    def __calc_perturb_dsigma(self, p, l):
        """
        Perturbative result of sigma (reduced self-energy)
        """
        ml2 = self.m2+np.exp(-2.*l)
        gtm = self.Nphi*self.g4/(ml2**2)
        Ni = 1./self.Nphi
        t1 = gtm*(1.+2.*Ni)/6.
        return (ml2/self.m2)*t1

    def __calc_largeN_gamma(self, p, l):
        """
        Large-N expansion result of gamma
        """
        ml2 = self.m2+np.exp(-2.*l)
        gm = self.Nphi*self.g4/(1.5*ml2**2)
        denom = 1./(1.+np.sqrt(1.+gm))
        zl = 2.*denom

        t1 = 0.25*ml2*gm*denom
        t2 = self.Nphi*(-0.25*gm*denom**2-0.5*np.log(zl))

        return t1*p**2+t2

    def __calc_largeN_dsigma(self, p, l):
        """
        Large-N expansion result of sigma (reduced self-energy)
        """
        ml2 = self.m2+np.exp(-2.*l)
        gm = self.Nphi*self.g4/(1.5*ml2**2)

        return (ml2/(2.*self.m2))*gm/(1.+np.sqrt(1.+gm))

    def __create_figure(self, x, y, y_exact=None, y_perturb=None, y_largeN=None, label='plot', title='result', iter=0):
        """
        Adding the plot of (x, y) to tensorboard

        y_exact:   exact results
        y_perturb: perturbative results
        y_largeN:  large-N expansion results
        """
        fig = plt.figure()
        plt.plot(x, y, linestyle='-', label='NN', color='red')
        if y_exact is not None:
            plt.plot(x, y_exact, linestyle='--', label='exact', color='black')
        if y_perturb is not None:
            plt.plot(x, y_perturb, linestyle='-.',
                     label='perturb', color='green')
        if y_largeN is not None:
            plt.plot(x, y_largeN, linestyle=':',
                     label='largeN', color='blue')
        plt.title(title)
        plt.legend(frameon=False, fontsize=9, loc="best", numpoints=1)
        fig.canvas.draw()
        plot_image = fig.canvas.renderer._renderer
        plot_image_array = np.array(plot_image).transpose(2, 0, 1)
        self.writer.add_image(label, plot_image_array, iter)
        plt.close(fig)

    def __plot_pdep_l1(self, model, loss_eval, iter, ispre=0):
        """
        Adding the phi-dependence plots of gamma and sigma at l=lend:
        """

        # Setting the label (for pretrain of PDE)
        label = "Pre" if ispre == 1 else "PDE"

        # Calculation of gamma from NN in all the direction of \vec{\phi}
        gam_l1_bc = model(self.input_p_shift_bc)
        gam_l1_origin = model(self.input_lend_origin)
        Gams_l1_origin = gam_l1_origin.cpu().detach().numpy().copy().squeeze()
        Gams_l1 = gam_l1_bc.cpu().detach().numpy().copy().squeeze()-Gams_l1_origin
        if self.Nphi > 1:
            Gams_l1 = Gams_l1.reshape(-1, self.Nphi)

        # Adding figure for gamma
        self.__create_figure(self.p_bc_n1_cpu, Gams_l1, y_exact=self.Gams_exact_l1_p, y_perturb=self.Gams_perturb_l1_p, y_largeN=self.Gams_largeN_l1_p,
                             label=f'Gamma_{label}/p-dep-l1', title='p-dep. (l=l1, loss={})'.format(loss_eval), iter=iter)

        # Calculation of sigma from NN in all the direction of \vec{\phi}
        lend_s = self.lend * \
            torch.ones(self.Npmesh, device=self.device).view(-1, 1)
        dsigma_NN = []
        for ntarget in range(self.Nphi):
            p0_u = self.p0*torch.ones(self.Npmesh, ntarget, device=self.device)
            p0_d = self.p0 * \
                torch.ones(self.Npmesh, self.Nphi -
                           ntarget - 1, device=self.device)
            p_s = p0_u
            p_s = torch.cat((p_s, self.p_bc_n1.view(-1, 1)), dim=1)
            p_s = torch.cat((p_s, p0_d), dim=1)

            def model_lp(_l, _p):
                input = torch.cat((_l, _p), dim=0).view(1, 1+self.Nphi)
                return model(input)[0][0]
            compute_batch_hessian = torch.vmap(
                torch.func.hessian(model_lp, argnums=1), in_dims=(0, 0))
            dsigma_pp = compute_batch_hessian(lend_s, p_s)
            if self.Nphi == 1:
                dsigma_pp_nn = torch.diagonal(dsigma_pp, dim1=-2, dim2=-1).cpu(
                ).detach().numpy().copy().squeeze()
                dsigma_NN.append(dsigma_pp_nn/self.m2)
            else:
                dsigma_pp_nn = torch.diagonal(dsigma_pp, dim1=-2, dim2=-1).cpu(
                ).detach().numpy().copy().squeeze()[:, ntarget]
                dsigma_NN.append(dsigma_pp_nn/self.m2)

        dsigma_NN = np.array(dsigma_NN)
        dsigma_NN = np.transpose(dsigma_NN)

        # Adding figure for sigma
        self.__create_figure(self.p_bc_n1_cpu, dsigma_NN,
                             y_exact=self.dsigma_exact_l1_p, y_perturb=self.dsigma_perturb_l1_p, y_largeN=self.dsigma_largeN_l1_p,
                             label=f'Dsigma_{label}/p-dep-l1', title='p-dep. (l=l1, loss={})'.format(loss_eval), iter=iter)

    def __plot_ldep_p0(self, model, loss_eval, iter, ispre=0):
        """
        Adding the l-dependence plots of gamma and sigma at phi=p0:
        """

        # Setting the label (for pretrain of PDE)
        label = "Pre" if ispre == 1 else "PDE"

        # Calculation of gamma from NN
        gam_p0_bc = model(self.input_l_p0)
        Gams_p0 = gam_p0_bc.cpu().detach().numpy().copy().squeeze()

        # Adding figure for gamma
        self.__create_figure(self.l_bc_cpu, Gams_p0, y_exact=self.Gams_exact_l_p0, y_perturb=self.Gams_perturb_l_p0, y_largeN=self.Gams_largeN_l_p0,
                             label=f'Gamma_{label}/l-dep-p0', title='l-dep. (p=p0, loss={})'.format(loss_eval), iter=iter)

        # Calculation of sigma from NN
        def model_lp(_l, _p):
            input = torch.cat((_l, _p), dim=0).view(1, 1+self.Nphi)
            return model(input)[0][0]
        compute_batch_hessian = torch.vmap(
            torch.func.hessian(model_lp, argnums=1), in_dims=(0, 0))
        dsigma_pp = compute_batch_hessian(self.l_bc, self.p0_bc)
        dsigma_pp_11 = torch.diagonal(dsigma_pp, dim1=-2, dim2=-1).cpu(
        ).detach().numpy().copy().squeeze()

        # Adding figure for sigma
        self.__create_figure(self.l_bc_cpu, dsigma_pp_11/self.m2,
                             y_exact=self.dsigma_exact_l_p0, y_perturb=self.dsigma_perturb_l_p0, y_largeN=self.dsigma_largeN_l_p0,
                             label=f'Dsigma_{label}/l-dep-p0', title='l-dep. (p=p0, loss={})'.format(loss_eval), iter=iter)

    def save_gamma0(self, model, iter, ispre=0):
        """
        Saving the value of gamma at phi=0 and l=lend 
        """

        # Setting the label (for pretrain of PDE)
        label = "Pre" if ispre == 1 else "PDE"

        # Calculation of gamma
        gam_NN = model(self.input_lend_p0).cpu(
        ).detach().numpy().copy().squeeze()

        self.writer.add_scalar(
            f"Phys_{label}/Gamma", gam_NN, iter)
        self.writer.flush()

    def save_sigma0(self, model, iter, ispre=0):
        """
        Saving the data of sigma at phi=0 and l=lend 
        Ave and std among the results in all the direction of \vec{\phi} are provided.
        """

        # Setting the label (for pretrain of PDE)
        label = "Pre" if ispre == 1 else "PDE"

        # Calculation of sigma
        def model_lp(_l, _p):
            input = torch.cat((_l, _p), dim=0).view(1, 1+self.Nphi)
            return model(input)[0][0]
        compute_batch_hessian = torch.vmap(
            torch.func.hessian(model_lp, argnums=1), in_dims=(0, 0))
        dsigma_pp = compute_batch_hessian(self.l_p0lend, self.p0_p0lend)
        dsigma_pp_nn = torch.diagonal(dsigma_pp, dim1=-2, dim2=-1).cpu(
        ).detach().numpy().copy().squeeze()/self.m2

        dsigma_NN_ave = np.average(dsigma_pp_nn)
        dsigma_NN_std = np.std(dsigma_pp_nn)

        self.writer.add_scalar(
            f"Phys_{label}/Sigma_ave", dsigma_NN_ave, iter)

        self.writer.add_scalar(
            f"Phys_{label}/Sigma_std", dsigma_NN_std, iter)
        self.writer.flush()

    def train(self, loss, iter, ispre=0):
        """
        Saving the loss (training phase)
        """

        label = "Pre" if ispre == 1 else "PDE"
        self.writer.add_scalar(
            f"Loss_{label}/train_logscale", torch.log10(loss), iter)
        self.writer.flush()

    def eval(self, loss, iter, ispre=0):
        """
        Saving the loss (evaluation phase)
        """

        label = "Pre" if ispre == 1 else "PDE"
        self.writer.add_scalar(
            f"Loss_{label}/eval_logscale", torch.log10(loss), iter)
        self.writer.flush()

    def plots(self, model, loss_eval, iter, ispre=0):
        self.__plot_pdep_l1(model, loss_eval, iter, ispre)
        self.__plot_ldep_p0(model, loss_eval, iter, ispre)
        self.writer.flush()
