import torch
import time
from datetime import timedelta


class solver():
    """
    In this class, the NN model is iteratively trained 
    to minimize loss functions.
    """

    def __init__(self, model, loss_fn, optimizer, ot=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.ot = ot

    def train(self, ispre=0):
        self.model.train()
        loss = self.loss_fn(self.model, ispre)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def evaluate(self, ispre=0):
        self.model.eval()
        loss = self.loss_fn(self.model, ispre)
        return loss

    def solve_eq(self, Niter, Nprint, Nplot, Nsave, cptpath_save, istart, lr, lr_decay=1., ispre=0):
        """
        Iteration for training
        """

        tstart = time.perf_counter()

        if istart == 1:
            eval_loss = self.evaluate(ispre=ispre)
            tnow = time.perf_counter()
            td = timedelta(seconds=int(tnow-tstart))
            if ispre == 1:
                print(f"\n[{td},pretraining] Iteration 0 (lr:{lr})")
            else:
                print(f"\n[{td}] Iteration 0 (lr:{lr})")
            print(f"-------------------------------")
            print(f"Test Error: \n Avg loss: {eval_loss.item()}")
            if self.ot is not None:
                self.ot.eval(eval_loss, 0, ispre=ispre)
                self.ot.save_gamma0(self.model, 0, ispre=ispre)
                self.ot.save_sigma0(self.model, 0, ispre=ispre)
                self.ot.plots(self.model, eval_loss, 0, ispre=ispre)

        for i in range(Niter):
            # Setting decayed learining rate
            lr_new = lr*(lr_decay**(i+istart-1))
            for g in self.optimizer.param_groups:
                g['lr'] = lr_new

            # Training
            train_loss = self.train(ispre=ispre)

            # Output of results
            if i % Nprint == 0 or i == Niter-1:
                eval_loss = self.evaluate(ispre=ispre)

                tnow = time.perf_counter()
                td = timedelta(seconds=int(tnow-tstart))

                if ispre == 1:
                    print(
                        f"\n[{td},pretraining] Iteration {i + istart} (lr:{lr_new})")
                else:
                    print(f"\n[{td}] Iteration {i + istart} (lr:{lr_new})")
                print(f"-------------------------------")
                print(f"loss: {train_loss.item()}")
                print(f"Test Error: \n Avg loss: {eval_loss.item()}")

                if self.ot is not None:
                    self.ot.train(train_loss, i + istart, ispre=ispre)
                    self.ot.eval(eval_loss, i + istart, ispre=ispre)
                    self.ot.save_gamma0(self.model, i + istart, ispre=ispre)
                    self.ot.save_sigma0(self.model, i + istart, ispre=ispre)

            if (i % Nplot == 0 or i == Niter-1) and self.ot is not None:
                eval_loss = self.evaluate(ispre=ispre)
                self.ot.plots(self.model, eval_loss, i + istart, ispre=ispre)

            # Saving check point
            if i > 0 and (i % Nsave == 0 or i == Niter-1):
                it = 1 if ispre else i + 1 + istart
                torch.save({'iter': it,
                            'model_state_dict': self.model.state_dict(),
                            'opt_state_dict': self.optimizer.state_dict()
                            }, cptpath_save)

        print(f"\nPretrain Done!" if ispre == 1 else f"\nDone!")
