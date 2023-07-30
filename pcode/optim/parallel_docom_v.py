# -*- coding: utf-8 -*-
from copy import deepcopy

import torch
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
import pcode.utils.communication as comm
from pcode.utils.sparsification import (
    get_n_bits,
    SignCompressor,
    SparsificationCompressor,
    QuantizationCompressor,
)
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.create_dataset import load_data_batch
from .parallel_choco_v import CHOCOCompressor
from .sam import SAM_wrapper

class ParallelDoCoM_V(Optimizer):
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
        super(ParallelDoCoM_V, self).__init__(params, defaults)

        # initialize gradient tracker
        for groups in self.param_groups:
            groups["grad_tracker"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["momentum_v"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["momentum_vp"] = [torch.zeros_like(prm) for prm in groups["params"]]

        self.model = model
        self.model_prev = deepcopy(
            model.module if "DataParallel" == model.__class__.__name__ else model
        )
        self.model_prev.zero_grad()

        # store the whole training arguments.
        self.conf = conf
        if not self.conf.lr_change_epochs is None:
            self.gamma_scheduling = [int(x) for x in conf.lr_change_epochs.split(",")] + [int(conf.num_iterations / conf.eval_freq)+1]
            self.gamma_decay = conf.gamma_decay
            self.gamma_sche_ind = 0
        
        # define the aggregator.
        self.rank = conf.graph.rank
        self.neighbors_info = conf.graph.get_neighborhood()
        self.aggregator = comm.get_aggregators(
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

        # define param names and init model_hat.
        self.param_names = list(
            enumerate([group["name"] for group in self.param_groups])
        )
        self.init_neighbor_hat_params()
        self.init_neighbor_hat_gt()
        self.consensus_stepsize = conf.consensus_stepsize

        # related to sparsification/quantization.
        self.compressor = CHOCOCompressor(
            aggregator=self.aggregator,
            comm_op=conf.comm_op,
            comm_device=self.conf.comm_device,
            compress_ratio=conf.compress_ratio,
            quantize_level=conf.quantize_level,
            is_biased=conf.is_biased,
            backend=conf.backend,
            use_ipc=conf.use_ipc,
        )

        # define auxilary functions.
        self.helper_thread = None
        self.sync_buffer = {}
        self.sync_buffer_gt = {}
        self.n_bits = 0
        self.first_step = True
        self.lr = lr
        self.beta = conf.momentum_beta
        self.beta_change_epochs = [int(t) for t in conf.beta_change_epochs.split(",") ] if not conf.beta_change_epochs is None else None
        
        if self.conf.sam:
            self.model_prev_sam_opt = SAM_wrapper(self.model_prev)

    
    def L2_regularize_grad(self, model):
        weight_decay = self.conf.weight_decay
        if weight_decay > 0:
            for key, prm in model.named_parameters():
                if not "bn" in key and weight_decay != 0:
                    prm.grad.data = prm.grad.data.detach().clone() + weight_decay * prm.data.detach().clone()

    def init_neighbor_hat_params(self):
        params, self.shapes = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        flatten_params = TensorBuffer(params)
        # flatten_params.buffer = torch.zeros_like(flatten_params.buffer)

        # init the neighbor_params.
        self.neighbor_hat_params = {
            self.rank: deepcopy(flatten_params),
            "memory": deepcopy(flatten_params),
        }

    def init_neighbor_hat_gt(self):
        params, self.shapes = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        flatten_params = TensorBuffer(params)
        flatten_params.buffer = torch.zeros_like(flatten_params.buffer)

        # init the neighbor_params.
        self.neighbor_hat_gt = {
            self.rank: deepcopy(flatten_params),
            "memory": deepcopy(flatten_params),
        }
    
    def __setstate__(self, state):
        super(ParallelDoCoM_V, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def inference(self, model, criterion, _input, _target):
        output = model(_input)
        loss = criterion(output, _target)
        return loss

                
    def init_momentum_tracker(self, conf, criterion, dataloader, **kargs):
        assert not conf.initial_batch_num is None, "conf.initial_batch_num is required for GT-HSGD"
        assert not conf.momentum_beta is None, "conf.momentum_beta is required for GT-HSGD"
        #  Full batch gradient if initial_batch_num > # batches available
        num_of_batches = conf.initial_batch_num if conf.initial_batch_num < len(dataloader) else len(dataloader)
        self.model.train()
        data_batches = []
        for i, (_input, _target) in enumerate(dataloader):
            _input, _target = load_data_batch(conf, _input, _target)
            data_batches.append((_input, _target))
            loss = self.inference(self.model, criterion, _input, _target) / num_of_batches
            loss.backward()
            if i == num_of_batches - 1:
                break
        
        if conf.sam:
            # Get large batch gradient of the SAM objective
            sam_optimizer = kargs["sam_opt"]
            sam_optimizer.first_step(zero_grad=True) # climb to local maxima
            for i, (_input, _target) in enumerate(data_batches):
                _input, _target = load_data_batch(conf, _input, _target)
                loss = self.inference(self.model, criterion, _input, _target) / num_of_batches
                loss.backward()
                if i == num_of_batches - 1:
                    break
            sam_optimizer.second_step(zero_grad=False) # revert model prm to the usual state, keep the SAM gradient for decentralized optimizer
        
        self.L2_regularize_grad(self.model)

        # copy gradient from model to momentum_v
        for groups in self.param_groups:
            for v, prm, gt in zip(groups["momentum_v"], groups["params"], groups["grad_tracker"]):
                gt.data = prm.grad.data.detach().clone()
                v.data = prm.grad.data.detach().clone()
    

    def update_momentum_v(self, _input, _target, criterion):
        # calculate stochastic gradient for model_prev, on the same batch of (_input, _target) that is used in distributed_running_cv.py on model
        self.model_prev.zero_grad()
        self.model_prev.train()
        loss = self.inference(self.model_prev, criterion, _input, _target)
        loss.backward()
        if self.conf.sam:
            # get gradient estimation of the SAM objective
            # self.model_prev_sam_opt.SAM_first_step(self.model_prev, rho=0.05 * self.lr / self.conf.lr) # climb to local maxima
            self.model_prev_sam_opt.SAM_first_step(self.model_prev) # climb to local maxima
            self.inference(self.model_prev, criterion, _input, _target).backward()
            self.model_prev_sam_opt.SAM_second_step(self.model_prev) # revert model prm to the usual state, keep the SAM gradient for decentralized optimizer

        self.L2_regularize_grad(self.model_prev)
        self.L2_regularize_grad(self.model)
        for prm, prm_prev in zip(self.model.parameters(), self.model_prev.parameters()):
            prm.grad.data = prm.grad.data.detach().clone() - (1 - self.beta) * prm_prev.grad.data.detach().clone()
        
        for groups in self.param_groups:
            for v, vp, prm in zip(groups["momentum_v"], groups["momentum_vp"], groups["params"]):
                vp.data = v.data.detach().clone()
                v.data = prm.grad.data.detach().clone() + (1 - self.beta) * v.data.detach().clone()
                

    def update_gradient_tracker(self):
        for group in self.param_groups:
            for v, vp, gt in zip(group["momentum_v"], group["momentum_vp"], group["grad_tracker"]):
                gt.data = gt.data.detach().clone() + v.data.detach().clone() - vp.data.detach().clone()


    def get_gt(self):
        gts = []
        for groups in self.param_groups:
            for gt in groups["grad_tracker"]:
                gts += gt.data.detach().clone().flatten().tolist()
        return gts

    # def update_consensus_stepsize(self):
    #     if not self.conf.lr_change_epochs is None:
    #         if self.conf.epoch_ >= self.gamma_scheduling[self.gamma_sche_ind] and not self.gamma_decay is None:
    #             self.gamma_sche_ind += 1
    #             self.consensus_stepsize /= self.gamma_decay

    def update_beta(self):
        if not self.beta_change_epochs is None and len(self.beta_change_epochs) > 0 and self.conf.epoch_ >= self.beta_change_epochs[0]:
            self.beta /= self.conf.beta_decay
            self.beta_change_epochs.pop(0)
        # self.beta = self.conf.momentum_beta * self.lr / self.conf.lr # decreasing beta w.r.t. to lr
        # self.beta = self.conf.momentum_beta * (self.lr / self.conf.lr)**2 # decreasing beta w.r.t. to lr, sqaured rate


    def step(self, closure=None, **kargs):
        # self.update_consensus_stepsize()
        self.update_beta()
        lr = kargs["scheduler"].get_lr()
        self.lr = lr if not lr is None else self.lr
        if self.first_step:
            self.first_step = False
            # stochastic gradient in self.model is not used for the first step.
            return self.prm_optimize_only_step(kargs["scheduler"])
        else:
            self.n_bits = 0
            self.update_momentum_v(kargs["input"], kargs["target"], kargs["criterion"])
            # copy model to model_prev
            for prm, prm_prev in zip(self.model.parameters(), self.model_prev.parameters()):
                prm_prev.data = prm.data.detach().clone()
            
            with kargs["timer"]("sync.update_gt", epoch=self.conf.epoch_):
                self.update_gradient_tracker()
            
            with kargs["timer"]("sync.finish_sync", epoch=self.conf.epoch_):
                utils.join_thread(self.helper_thread)
                self.n_bits = self.sync_buffer_gt.get("n_bits", 0) * (len(self.neighbors_info) - 1)
            
            # recover current params and hat_params
            with kargs["timer"]("sync.recover_hat_params", epoch=self.conf.epoch_):
                gts, flatten_gt, flatten_hat_gt = utils.recover_params(
                    param_groups=self.param_groups,
                    param_names=self.param_names,
                    rank=self.rank,
                    neighbor_hat_params=self.neighbor_hat_gt,
                    get_hat_params=True,
                    is_recover_grad_tracker=True
                )
            with kargs["timer"]("sync.update_flatten_gt", epoch=self.conf.epoch_):
                utils.update_params_from_neighbor(
                    neighbor_hat_params=self.neighbor_hat_gt,
                    flatten_params=flatten_gt,
                    consensus_stepsize=self.conf.consensus_stepsize, # fixed gamma
                    self_rank=self.rank,
                )
             # update the local model.
            with kargs["timer"]("sync.update_local_model", epoch=self.conf.epoch_):
                flatten_gt.unpack(gts)

            # start compress/sync.
            with kargs["timer"]("sync.start_sync", epoch=self.conf.epoch_):
                self.sync_buffer_gt = {
                    "original_shapes": self.shapes,
                    "flatten_params": flatten_gt,
                    "flatten_hat_params": flatten_hat_gt,
                }

                self.helper_thread = utils.HelperThread(
                    name=f"_thread_at_epoch_{self.conf.epoch_}.compress",
                    func=self.compressor.pipeline,
                    # the arguments below will be feeded into the `func`.
                    sync_buffer=self.sync_buffer_gt,
                    neighbor_hat_params=self.neighbor_hat_gt,
                    neighbors_info=self.neighbors_info,
                )
                self.helper_thread.start()

            # =========== theta_round ================

            # Apply the gradients with the weight decay and momentum.
            with kargs["timer"]("sync.apply_grad", epoch=self.conf.epoch_):
                utils.apply_gradient_from_gradient_tracker(
                    self.param_groups, self.state, apply_grad_to_model=True
                )

            with kargs["timer"]("sync.finish_sync", epoch=self.conf.epoch_):
                utils.join_thread(self.helper_thread)
                self.n_bits += self.sync_buffer.get("n_bits", 0) * (len(self.neighbors_info) - 1)

            # recover current params and hat_params
            with kargs["timer"]("sync.recover_hat_params", epoch=self.conf.epoch_):
                params, flatten_params, flatten_hat_params = utils.recover_params(
                    param_groups=self.param_groups,
                    param_names=self.param_names,
                    rank=self.rank,
                    neighbor_hat_params=self.neighbor_hat_params,
                    get_hat_params=True,
                )
            # get updated flatten params.
            with kargs["timer"]("sync.update_flatten_params", epoch=self.conf.epoch_):
                utils.update_params_from_neighbor(
                    neighbor_hat_params=self.neighbor_hat_params,
                    flatten_params=flatten_params,
                    consensus_stepsize=self.consensus_stepsize,
                    self_rank=self.rank,
                )
            # update the local model.
            with kargs["timer"]("sync.update_local_model", epoch=self.conf.epoch_):
                flatten_params.unpack(params)

            # start compress/sync.
            with kargs["timer"]("sync.start_sync", epoch=self.conf.epoch_):
                self.sync_buffer = {
                    "original_shapes": self.shapes,
                    "flatten_params": flatten_params,
                    "flatten_hat_params": flatten_hat_params,
                }

                self.helper_thread = utils.HelperThread(
                    name=f"_thread_at_epoch_{self.conf.epoch_}.compress",
                    func=self.compressor.pipeline,
                    # the arguments below will be feeded into the `func`.
                    sync_buffer=self.sync_buffer,
                    neighbor_hat_params=self.neighbor_hat_params,
                    neighbors_info=self.neighbors_info,
                )
                self.helper_thread.start()
                if kargs["scheduler"].check_eval():
                    utils.join_thread(self.helper_thread)
            return self.n_bits
    

    def prm_optimize_only_step(self, scheduler):
        # Apply the gradients with the weight decay and momentum.
        utils.apply_gradient_from_gradient_tracker(
            self.param_groups, self.state, apply_grad_to_model=True
        )

        utils.join_thread(self.helper_thread)
        self.n_bits = self.sync_buffer.get("n_bits", 0)

        # recover current params and hat_params
        params, flatten_params, flatten_hat_params = utils.recover_params(
            param_groups=self.param_groups,
            param_names=self.param_names,
            rank=self.rank,
            neighbor_hat_params=self.neighbor_hat_params,
            get_hat_params=True,
        )
        # get updated flatten params.
        utils.update_params_from_neighbor(
            neighbor_hat_params=self.neighbor_hat_params,
            flatten_params=flatten_params,
            consensus_stepsize=self.consensus_stepsize,
            self_rank=self.rank,
        )
        # update the local model.
        flatten_params.unpack(params)

        # start compress/sync.
        self.sync_buffer = {
            "original_shapes": self.shapes,
            "flatten_params": flatten_params,
            "flatten_hat_params": flatten_hat_params,
        }

        self.helper_thread = utils.HelperThread(
            name=f"_thread_at_epoch_{self.conf.epoch_}.compress",
            func=self.compressor.pipeline,
            # the arguments below will be feeded into the `func`.
            sync_buffer=self.sync_buffer,
            neighbor_hat_params=self.neighbor_hat_params,
            neighbors_info=self.neighbors_info,
        )
        self.helper_thread.start()
        if scheduler.check_eval():
            utils.join_thread(self.helper_thread)
        return self.n_bits
