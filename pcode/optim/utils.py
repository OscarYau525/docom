# -*- coding: utf-8 -*-
import threading

import torch

from pcode.utils.tensor_buffer import TensorBuffer
import pcode.utils.communication as comm


"""common utilities"""


def apply_gradient(param_groups, state, apply_grad_to_model=True, set_model_prm_as_grad=False, **kwargs):
    for group in param_groups:
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        dampening = group["dampening"]
        nesterov = group["nesterov"]

        for p in group["params"]:
            if p.grad is None:
                continue
            d_p = p.grad.data

            # get param_state
            param_state = state[p]

            # add weight decay.
            if weight_decay != 0:
                d_p.add_(p.data, alpha=weight_decay)

            # apply the momentum.
            if momentum != 0:
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                    buf.mul_(momentum).add_(d_p)
                else:
                    buf = param_state["momentum_buffer"]
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf
            if apply_grad_to_model:
                p.data.add_(d_p, alpha= -kwargs.get("lr", group["lr"]) )
            elif set_model_prm_as_grad:
                p.data = d_p * (-kwargs.get("lr", group["lr"]) )
            else:
                p.grad.data = d_p


def apply_gradient_from_gradient_tracker(param_groups, state, apply_grad_to_model=True):
    for group in param_groups:
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        dampening = group["dampening"]
        nesterov = group["nesterov"]

        for p, gt in zip(group["params"], group["grad_tracker"]):
            # get param_state
            param_state = state[p]

            # apply the momentum.
            if momentum != 0:
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                    buf.mul_(momentum).add_(gt)
                else:
                    buf = param_state["momentum_buffer"]
                    buf.mul_(momentum).add_(gt, alpha=1 - dampening)
                if nesterov:
                    gt = gt.add(momentum, buf)
                else:
                    gt = buf
            if apply_grad_to_model:
                # p.data.add_(gt + weight_decay * p.data, alpha=-group["lr"])
                # Gradient tracker algorithm applies weight decay to the gradient tracker
                p.data.add_(gt, alpha=-group["lr"])
            else:
                raise ValueError("Unconsidered route")
                p.grad.data = gt


def recover_params(
    param_groups, param_names, rank=None, neighbor_hat_params=None, get_hat_params=True, is_recover_grad_tracker=False
):
    # get flattened params.
    params, _ = comm.get_data(param_groups, param_names, is_get_grad=False, is_get_grad_tracker=is_recover_grad_tracker)
    flatten_params = TensorBuffer(params)

    if get_hat_params:
        assert neighbor_hat_params is not None and rank is not None
        # recover the hat_params.
        flatten_hat_params = TensorBuffer(params)
        flatten_hat_params.buffer.data[:] = neighbor_hat_params[rank].buffer
        return params, flatten_params, flatten_hat_params
    else:
        return params, flatten_params


def update_params_from_neighbor(
    neighbor_hat_params, flatten_params, consensus_stepsize, self_rank
): 
    # the same apply for gradient tracker
    flatten_params.buffer += consensus_stepsize * (
        neighbor_hat_params["memory"].buffer - neighbor_hat_params[self_rank].buffer
    )


"""utilities for parallel choco."""


class HelperThread(threading.Thread):
    def __init__(self, name, func, *args, **kargs):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func

        # task-related.
        self.args = args
        self.kargs = kargs

    def run(self):
        self.func(**self.kargs)


def join_thread(thread):
    if thread is None:
        return False
    thread.join()
    return True
