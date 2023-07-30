# -*- coding: utf-8 -*-
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
import pcode.utils.communication as comm
from pcode.utils.sparsification import get_n_bits
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.create_dataset import load_data_batch
import pcode.utils.error_handler as error_handler

import torch
import copy

class GtHsgd(Optimizer):
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
        super(GtHsgd, self).__init__(params, defaults)

        # alloc memory for gradient tracker and momentum_v
        for groups in self.param_groups:
            groups["grad_tracker"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["momentum_v"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["momentum_vp"] = [torch.zeros_like(prm) for prm in groups["params"]]
        
        self.model = model
        self.model_prev = copy.deepcopy(
            model.module if "DataParallel" == model.__class__.__name__ else model
        )
        self.model_prev.zero_grad()
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
        self.first_step = True
        self.beta = conf.momentum_beta
        self.beta_change_epochs = [int(t) for t in conf.beta_change_epochs.split(",") ] if not conf.beta_change_epochs is None else None
  

    def __setstate__(self, state):
        super(GtHsgd, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
    
    def inference(self, model, criterion, _input, _target):
        output = model(_input)
        loss = criterion(output, _target)
        return loss
    

    def L2_regularize_grad(self, model):
        weight_decay = self.conf.weight_decay
        for key, prm in model.named_parameters():
            if not "bn" in key and weight_decay != 0:
                prm.grad.data = prm.grad.data.detach().clone() + weight_decay * prm.data.detach().clone()

    
    def init_momentum_tracker(self, conf, criterion, dataloader, **kargs):
        assert not conf.initial_batch_num is None, "conf.initial_batch_num is required for GT-HSGD"
        assert not conf.momentum_beta is None, "conf.momentum_beta is required for GT-HSGD"
        #  Full batch gradient if initial_batch_num > # batches available
        num_of_batches = conf.initial_batch_num if conf.initial_batch_num < len(dataloader) else len(dataloader)
        self.model.train()
        for i, (_input, _target) in enumerate(dataloader):
            _input, _target = load_data_batch(conf, _input, _target)
            loss = self.inference(self.model, criterion, _input, _target) / num_of_batches
            loss.backward()
            if i == num_of_batches - 1:
                break
        self.L2_regularize_grad(self.model)
        
        # copy gradient from model to momentum_v
        for groups in self.param_groups:
            for v, prm, gt in zip(groups["momentum_v"], groups["params"], groups["grad_tracker"]):
                v.data = prm.grad.data.detach().clone()
    

    def update_momentum_v(self, _input, _target, criterion):
        self.model_prev.zero_grad()
        self.model_prev.train()
        loss = self.inference(self.model_prev, criterion, _input, _target)
        loss.backward()
        self.L2_regularize_grad(self.model)
        self.L2_regularize_grad(self.model_prev)
        for prm, prm_prev in zip(self.model.parameters(), self.model_prev.parameters()):
            prm.grad.data = prm.grad.data.detach().clone() - (1 - self.conf.momentum_beta) * prm_prev.grad.data.detach().clone()
        
        for groups in self.param_groups:
            for v, vp, prm in zip(groups["momentum_v"], groups["momentum_vp"], groups["params"]):
                vp.data = v.data.detach().clone()
                v.data = prm.grad.data.detach().clone() + (1 - self.conf.momentum_beta) * v.data.detach().clone()
            
    def get_gt(self):
        gts = []
        for groups in self.param_groups:
            for gt in groups["grad_tracker"]:
                gts += gt.data.detach().clone().flatten().tolist()
        return gts

    def update_beta(self):
        if not self.beta_change_epochs is None and len(self.beta_change_epochs) > 0 and self.conf.epoch_ >= self.beta_change_epochs[0]:
            self.beta /= self.conf.beta_decay
            self.beta_change_epochs.pop(0)
    
    def step(self, closure=None, **kargs):
        self.update_beta()
        # momentum step
        if not self.first_step:
            # stochastic gradient in self.model is used for non-first step.
            self.update_momentum_v(kargs["input"], kargs["target"], kargs["criterion"])
        else:
            # stochastic gradient in self.model is not used for the first step.
            self.first_step = False

        # copy model to model_prev
        for prm, prm_prev in zip(self.model.parameters(), self.model_prev.parameters()):
            prm_prev.data = prm.data.detach().clone()

        # with momentum v updated
        with kargs["timer"]("sync.update_gt", epoch=self.conf.epoch_):
            # first get and flatten all params.
            gt_tensors = []
            for group in self.param_groups:
                for gt, v, vp, prm in zip(group["grad_tracker"], group["momentum_v"], group["momentum_vp"], group["params"]):
                    gt.data = gt.data.detach().clone() + v.data.detach().clone() - vp.data.detach().clone()
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

        
        with kargs["timer"]("sync.update_theta", epoch=self.conf.epoch_):
            # first get and flatten all params.
            theta_tensors = []
            utils.apply_gradient_from_gradient_tracker(self.param_groups, self.state)
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

        # Get n_bits to transmit.
        n_bits = (get_n_bits(flatten_send_g.buffer) + get_n_bits(flatten_send_theta.buffer)) * (len(self.neighbors_info) - 1)
        return n_bits
