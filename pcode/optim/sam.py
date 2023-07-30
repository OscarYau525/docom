import torch
from copy import deepcopy

class SAM_wrapper():
    def __init__(self, model):
        self.old_model = deepcopy(
            model.module if "DataParallel" == model.__class__.__name__ else model
        )

    @torch.no_grad()
    def SAM_first_step(self, model, rho=0.05, adaptive=False):
        grad_norm = torch.norm(
            torch.stack(
                [((torch.abs(p) if adaptive else 1.0) * p.grad).norm(p=2) for p in model.parameters()]
            ), p=2
        )
        for prm, mem_prm in zip(model.parameters(), self.old_model.parameters()):
            mem_prm.data = prm.data.clone() # record the usual state
        scale = rho / (grad_norm + 1e-12)

        for prm in model.parameters():
            e_w = (torch.pow(prm, 2) if adaptive else 1.0) * prm.grad * scale
            prm.data.add_(e_w) # accent to maxima
            prm.grad.data.zero_()
    
    @torch.no_grad()
    def SAM_second_step(self, model):
        for prm, mem_prm in zip(model.parameters(), self.old_model.parameters()):
            prm.data = mem_prm.data.clone()
        

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, adaptive=False, **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.adaptive = adaptive
        self.param_groups = base_optimizer.param_groups
        self.defaults.update(base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, rho=0.05, zero_grad=False):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = rho / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        # leave the gradient estimate on model.grad
        # self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
