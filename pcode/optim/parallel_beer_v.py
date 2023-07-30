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


class ParallelBEER_V(Optimizer):
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
        super(ParallelBEER_V, self).__init__(params, defaults)

        # initialize gradient tracker
        for groups in self.param_groups:
            groups["grad_tracker"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["grad_p"] = [torch.zeros_like(prm) for prm in groups["params"]]

        self.model = model

        # store the whole training arguments.
        self.conf = conf
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
    
    def L2_regularize_grad(self):
        weight_decay = self.conf.weight_decay
        for key, prm in self.model.named_parameters():
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
        super(ParallelBEER_V, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def inference(self, model, criterion, _input, _target):
        output = model(_input)
        loss = criterion(output, _target)
        return loss

    def init_grad_prev(self):
        for group in self.param_groups:
            for prm, g_p in zip(group["params"], group["grad_p"]):
                # put current gradient into grad_p
                g_p.data = prm.grad.data.detach().clone()
            
    
    def init_gradient_tracker(self, conf, criterion, dataloader, **kargs):
        # update gradient tracker with X_0
        self.model.train()
        num_of_batches = len(dataloader)
        for i, (_input, _target) in enumerate(dataloader):
            _input, _target = load_data_batch(conf, _input, _target)
            loss = self.inference(self.model, criterion, _input, _target) / num_of_batches
            loss.backward()
        self.L2_regularize_grad()
        
        for group in self.param_groups:
            for gt, prm, g_p in zip(group["grad_tracker"], group["params"], group["grad_p"]):
                gt.data =  prm.grad.data.detach().clone()
                # put current gradient into grad_p
                g_p.data = prm.grad.data.detach().clone()
        

    def update_gradient_tracker(self):
        self.L2_regularize_grad()
        for group in self.param_groups:
            for gt, prm, g_p in zip(group["grad_tracker"], group["params"], group["grad_p"]):
                    gt.data = gt.data.detach().clone() + prm.grad.data.detach().clone() - g_p.data.detach().clone()
                    # put current gradient into grad_p
                    g_p.data = prm.grad.data.detach().clone()


    def get_gt(self):
        gts = []
        for groups in self.param_groups:
            for gt in groups["grad_tracker"]:
                gts += gt.data.detach().clone().flatten().tolist()
        return gts

    def update_consensus_stepsize(self):
        if self.conf.epoch_ >= self.gamma_scheduling[self.gamma_sche_ind] and not self.gamma_decay is None:
            self.gamma_sche_ind += 1
            self.consensus_stepsize /= self.gamma_decay

    def theta_round_only(self, **kargs):
        n_bits = 0
        # =========== theta_round ================
        # Apply the gradients with the weight decay and momentum.
        with kargs["timer"]("sync.apply_grad", epoch=self.conf.epoch_):
            utils.apply_gradient_from_gradient_tracker(
                self.param_groups, self.state, apply_grad_to_model=True
            )

        with kargs["timer"]("sync.finish_sync", epoch=self.conf.epoch_):
            utils.join_thread(self.helper_thread)
            n_bits += self.sync_buffer.get("n_bits", 0) * (len(self.neighbors_info) - 1)

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

        # eq 3 done

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
        
        # eq 4,5 done
        return n_bits

    def step(self, closure=None, **kargs):
        n_bits = 0
        self.update_consensus_stepsize()
        if self.first_step:
            self.first_step = False
            self.init_grad_prev()
            return self.theta_round_only(**kargs)
        # =========== gt_round ================

        with kargs["timer"]("sync.update_gt", epoch=self.conf.epoch_):
            self.update_gradient_tracker()
        
        with kargs["timer"]("sync.finish_sync", epoch=self.conf.epoch_):
            utils.join_thread(self.helper_thread)
            n_bits += self.sync_buffer.get("n_bits", 0) * (len(self.neighbors_info) - 1)
        
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
        
        # eq 6 done

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
                
        # eq 7,8 done

        # =========== theta_round ================

        # Apply the gradients with the weight decay and momentum.
        with kargs["timer"]("sync.apply_grad", epoch=self.conf.epoch_):
            utils.apply_gradient_from_gradient_tracker(
                self.param_groups, self.state, apply_grad_to_model=True
            )

        with kargs["timer"]("sync.finish_sync", epoch=self.conf.epoch_):
            utils.join_thread(self.helper_thread)
            n_bits += self.sync_buffer.get("n_bits", 0) * (len(self.neighbors_info) - 1)

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

        # eq 3 done

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
        
        # eq 4,5 done

        
        return n_bits
