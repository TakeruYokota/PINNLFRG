import torch
from torch import nn


class DELoss(nn.Module):
    """
    This class offers the loss functions 
    used in the pretraining and training of Wetterich eq
    """

    def __init__(self, Nphi, lend, m2, g4, Ncol, Ncol_pre=0,
                 std_col_mul=1., std_col_mul_pre=1., device='cpu'):
        super(DELoss, self).__init__()
        self.lend = lend
        self.Nphi = Nphi
        self.sqNphi = (float(Nphi))**0.5
        self.Ncol = Ncol
        self.Ncol_pre = Ncol_pre
        self.m2 = m2
        self.m = m2**0.5
        self.g4 = g4
        self.g4_3 = g4 / 3.
        self.std_col = std_col_mul * self.sqNphi / self.m
        self.std_col_pre = std_col_mul_pre*self.sqNphi / self.m
        self.device = device

        self.massvec = torch.tensor([m2]*Nphi, device=device)
        self.massmatrix = torch.diag(self.massvec)
        self.eyeNphi = torch.eye(self.Nphi, device=device)
        self.g4eyeNphi = (g4 / 6.) * self.eyeNphi

    def gam2_int_0(self, _p):
        """
        2nd derivative of gamma from bare interaction term
        """
        phi2 = torch.mul(_p, _p)
        phi2sum = phi2.sum(dim=-1).view(-1, 1)
        phi4_diag = torch.einsum('ij,jk->ijk', phi2sum, self.g4eyeNphi)
        phi4_nondiag = self.g4_3 * torch.einsum('ij,ik->ijk', _p, _p)
        return phi4_diag + phi4_nondiag

    def Gfree(self, R):
        """
        Free propagator
        """
        Lam_bc_batch = R.repeat(1, self.Nphi)
        invprop = torch.unsqueeze(self.massvec, dim=0).repeat(
            R.size()[0], 1) + Lam_bc_batch
        return torch.einsum('bj,jk->bjk', 1./invprop, self.eyeNphi)

    def DE_LHS(self, model, l_bc, p_bc):
        """
        LHS of Wetterich eq
        """
        gam_bc = model(torch.cat((l_bc, p_bc), dim=1))
        gam_l_bc = torch.autograd.grad(gam_bc, l_bc, grad_outputs=torch.ones(l_bc.size(), device=self.device),
                                       create_graph=True)
        return (gam_l_bc[0])[:, 0]

    def DE_RHS(self, model, l_bc, p_bc):
        """
        RHS of Wetterich eq
        """
        def model_lp(_l, _p):
            input = torch.cat((_l, _p), dim=0).view(1, 1+self.Nphi)
            return model(input)[0][0]

        # 2nd derivative of gamma
        compute_batch_hessian = torch.vmap(
            torch.func.hessian(model_lp, argnums=1), in_dims=(0, 0))
        gam_pp = compute_batch_hessian(l_bc, p_bc)

        # Batch of mass matrix
        mmat = torch.einsum(
            'i,jk->ijk', torch.ones(l_bc.size()[0], device=self.device), self.massmatrix)

        # Batch of regulator
        R = torch.exp(-2.*l_bc)
        Rmat = torch.einsum('bi,jk->bjk', R, self.eyeNphi)

        # Regulated propagator G
        gamref_int2 = self.gam2_int_0(p_bc)
        G = torch.linalg.inv(mmat + gamref_int2 + gam_pp + Rmat)

        # Free propagator G0
        G0 = self.Gfree(R)

        # RHS of Wetterich eq
        # Prefactor is not 1/2 but -1. See the choice of R
        return -torch.einsum('bii,bii->b', Rmat, G - G0)

    def DE(self, model, l_bc, p_bc):
        """
        LHS and RHS of Wetterich eq
        """
        return (self.DE_LHS(model, l_bc, p_bc), self.DE_RHS(model, l_bc, p_bc))

    def Loss_PDE(self, model):
        """
        Loss function for PDE training
        """

        if self.Ncol == 0:
            return 0.

        # Generating collocation points
        l_bc_origin = self.lend * torch.rand(self.Ncol, device=self.device)
        l_bc = l_bc_origin.view(-1, 1)
        l_bc.requires_grad_()

        p_n_bc = torch.nn.functional.normalize(
            2.*torch.rand(self.Ncol, self.Nphi, device=self.device) - 1.)
        p_r_bc = torch.abs(torch.normal(
            mean=0., std=self.std_col, size=(self.Ncol,), device=self.device))
        p_bc = torch.einsum('b,bi->bi', p_r_bc, p_n_bc)
        p_bc.requires_grad_()

        # LHS & RHS of Wetterich equation
        eleft, eright = self.DE(model, l_bc, p_bc)

        return torch.mean(torch.square(eleft-eright))

    def Loss_pre(self, model):
        """
        Loss function for pretraining
        """
        if self.Ncol_pre == 0:
            return 0.

        # Generating collocation points
        l_bc_origin = self.lend * torch.rand(self.Ncol_pre, device=self.device)
        l_bc = l_bc_origin.view(-1, 1)

        p_n_bc = torch.nn.functional.normalize(
            2.*torch.rand(self.Ncol_pre, self.Nphi, device=self.device) - 1.)
        p_r_bc = torch.abs(torch.normal(
            mean=0., std=self.std_col_pre, size=(self.Ncol_pre,), device=self.device))
        p_bc = torch.einsum('b,bi->bi', p_r_bc, p_n_bc)

        # gamma given by first order perturbation
        ml2 = self.m2+torch.exp(-2.*l_bc_origin)
        gtm = self.Nphi*self.g4/(ml2**2)
        Ni = 1./self.Nphi
        t1 = (self.Nphi*(1.+2.*Ni)/24.)*gtm
        t2 = ((1.+2.*Ni)/12.)*ml2*gtm
        p2sum_bc = torch.einsum('bi,bi->b', p_bc, p_bc)
        gam_perturb_bc = torch.unsqueeze(t2*p2sum_bc + t1, dim=-1)

        # gamma given by NN
        gam_bc = model(torch.cat((l_bc, p_bc), dim=1))

        return torch.mean(torch.square(gam_bc-gam_perturb_bc))

    def forward(self, model, ispre=0):
        # if ispre == 1:
        #     return self.Loss_pre(model)
        # else:
        #     return self.Loss_PDE(model)
        return self.Loss_PDE(model) if ispre != 1 else self.Loss_pre(model)
        # return self.Loss_pre(model) if ispre == 1 else self.Loss_PDE(model)
