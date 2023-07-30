# -*- coding: utf-8 -*-
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
import pcode.utils.communication as comm
from pcode.utils.sparsification import get_n_bits
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.create_dataset import load_data_batch

import torch
import copy

class GNSD(Optimizer):
    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        conf=None,
        model=None,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(GNSD, self).__init__(params, defaults)

        # alloc memory for gradient tracker and momentum_v
        for groups in self.param_groups:
            groups["grad_tracker"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["grad_p"] = [torch.zeros_like(prm) for prm in groups["params"]]
        
        self.model = model
        self.first_step = True
        # store the whole training arguments.
        self.conf = conf
        self.rank = conf.graph.rank
        self.neighbors_info = conf.graph.get_neighborhood()

        # define the aggregator.
        self.decentralized_aggregator = comm.get_aggregators(
            cur_rank=self.rank,
            world=conf.graph.ranks,
            neighbors_info=self.neighbors_info,
            aggregator_type="decentralized",
        )
        self.world_aggregator = comm.get_aggregators(
            cur_rank=self.rank,
            world=conf.graph.ranks,
            neighbors_info=dict(
                (rank, 1.0 / conf.graph.n_nodes) for rank in conf.graph.ranks
            ),
            aggregator_type="centralized",
        )

        # define reducer.
        self.backend = conf.backend

        # define sorted param names.
        self.param_names = list(
            enumerate([group["name"] for group in self.param_groups])
        )

    def __setstate__(self, state):
        super(GNSD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
    

    def L2_regularize_grad(self):
        weight_decay = self.conf.weight_decay
        for key, prm in self.model.named_parameters():
            if not "bn" in key and weight_decay != 0:
                prm.grad.data = prm.grad.data.detach().clone() + weight_decay * prm.data.detach().clone()


    def get_gt(self):
        gts = []
        for groups in self.param_groups:
            for gt in groups["grad_tracker"]:
                gts += gt.data.detach().clone().flatten().tolist()
        return gts


    def update_prm_only(self, timer):
        # update gradient tracker
        with timer("sync.update_gt", epoch=self.conf.epoch_):
            for group in self.param_groups:
                for gt, prm, g_p in zip(group["grad_tracker"], group["params"], group["grad_p"]):
                    gt.data =  prm.grad.data.detach().clone()
                    # put current gradient into grad_p
                    g_p.data = prm.grad.data.detach().clone()
        

        # comm prms
        with timer("sync.update_theta", epoch=self.conf.epoch_):
            # first get and flatten all params.
            theta_tensors = []
            # utils.apply_gradient(self.param_groups, self.state)
            for groups in self.param_groups:
                for prm in groups["params"]:
                    theta_tensors.append(prm)
            flatten_send_theta = TensorBuffer(theta_tensors)
        
        with timer("sync.sync", epoch=self.conf.epoch_):
            # prepare the sync.
            if self.conf.comm_device == "cpu":
                flatten_send_theta.buffer.cpu().detach_()

            # then sync.
            flatten_send_theta.buffer = self.decentralized_aggregator._agg(
                flatten_send_theta.buffer, op="weighted"
            )
        
        with timer("sync.update_theta", epoch=self.conf.epoch_):
            # finally unflatten.
            flatten_send_theta.unpack(theta_tensors)


        # update prms
        with timer("sync.update_theta", epoch=self.conf.epoch_):
            utils.apply_gradient_from_gradient_tracker(self.param_groups, self.state)

        n_bits = get_n_bits(flatten_send_theta.buffer)
        return n_bits


    def step(self, closure=None, **kargs):
        self.L2_regularize_grad()

        if self.first_step:
            self.first_step = False
            return self.update_prm_only(kargs["timer"])
        
        # comm gradient tracker
        with kargs["timer"]("sync.update_gt", epoch=self.conf.epoch_):
            # first get and flatten all params.
            gt_tensors = []
            for group in self.param_groups:
                for gt in group["grad_tracker"]:
                    gt_tensors.append(gt)
            flatten_send_g = TensorBuffer(gt_tensors)


        with kargs["timer"]("sync.sync", epoch=self.conf.epoch_):
            # prepare the sync.
            if self.conf.comm_device == "cpu":
                flatten_send_g.buffer.cpu().detach_()

            # then sync.
            flatten_send_g.buffer = self.decentralized_aggregator._agg(
                flatten_send_g.buffer, op="weighted"
            )
        

        with kargs["timer"]("sync.update_gt", epoch=self.conf.epoch_):
            # finally unflatten.
            for recv_gt, gt in zip(flatten_send_g, gt_tensors):
                gt.data = recv_gt.data.detach().clone()
            flatten_send_g.unpack(gt_tensors)

        # update gradient tracker
        with kargs["timer"]("sync.update_gt", epoch=self.conf.epoch_):
            for group in self.param_groups:
                for gt, prm, g_p in zip(group["grad_tracker"], group["params"], group["grad_p"]):
                    gt.data = gt.data.detach().clone() + prm.grad.data.detach().clone() - g_p.data.detach().clone()
                    # put current gradient into grad_p
                    g_p.data = prm.grad.data.detach().clone()

        
        # comm prms
        with kargs["timer"]("sync.update_theta", epoch=self.conf.epoch_):
            # first get and flatten all params.
            theta_tensors = []
            # utils.apply_gradient(self.param_groups, self.state)
            for groups in self.param_groups:
                for prm in groups["params"]:
                    theta_tensors.append(prm)
            flatten_send_theta = TensorBuffer(theta_tensors)
        
        with kargs["timer"]("sync.sync", epoch=self.conf.epoch_):
            # prepare the sync.
            if self.conf.comm_device == "cpu":
                flatten_send_theta.buffer.cpu().detach_()

            # then sync.
            flatten_send_theta.buffer = self.decentralized_aggregator._agg(
                flatten_send_theta.buffer, op="weighted"
            )
        
        with kargs["timer"]("sync.update_theta", epoch=self.conf.epoch_):
            # finally unflatten.
            flatten_send_theta.unpack(theta_tensors)


        # update prms
        with kargs["timer"]("sync.update_theta", epoch=self.conf.epoch_):
            utils.apply_gradient_from_gradient_tracker(self.param_groups, self.state)

        # Get n_bits to transmit.
        n_bits = (get_n_bits(flatten_send_g.buffer) + get_n_bits(flatten_send_theta.buffer)) * (len(self.neighbors_info) - 1)
        return n_bits
