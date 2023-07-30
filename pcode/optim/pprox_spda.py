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
import numpy as np

class PProx_SPDA(Optimizer):
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
        super(PProx_SPDA, self).__init__(params, defaults)

        # define the aggregator.
        self.rank = conf.graph.rank
        torch.manual_seed(self.rank)
        np.random.seed(self.rank)
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

        self.rho = conf.rho
        self.gamma = conf.gamma
        self.alpha = conf.alpha
        self.edge_fraction = conf.edge_fraction
        # self.adaptive_lr = conf.adaptive_lr
        self.vx, self.vy = 1, 1

        # initialize dual variable lambda
        for groups in self.param_groups:
            groups["lambdas"] = [{nei: torch.zeros_like(prm) for nei in self.neighbors_info} for prm in groups["params"]]

        self.model = model

        # store the whole training arguments.
        self.conf = conf
        if not self.conf.lr_change_epochs is None:
            self.gamma_scheduling = [int(x) for x in conf.lr_change_epochs.split(",")] + [int(conf.num_iterations / conf.eval_freq)+1]
            self.gamma_decay = conf.gamma_decay
            self.gamma_sche_ind = 0
        
        

        # define param names and init model_hat.
        self.param_names = list(
            enumerate([group["name"] for group in self.param_groups])
        )
        # self.init_neighbor_hat_params()
        # self.init_neighbor_hat_gt()
        self.consensus_stepsize = conf.consensus_stepsize

        # related to sparsification/quantization.
        self.compressor = SPDA_VaryingGraph_Sparsifier(
            aggregator=self.aggregator,
            comm_device=self.conf.comm_device,
            compress_ratio=conf.compress_ratio,
            is_biased=conf.is_biased,
            backend=conf.backend,
            use_ipc=conf.use_ipc,
            edge_fraction=self.edge_fraction
        )

        # define auxilary functions.
        self.helper_thread = None
        self.sync_buffer = {}
        self.sync_buffer_gt = {}
        self.n_bits = 0
        self.first_step = True

        _, self.shapes = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
    
    def L2_regularize_grad(self, model):
        weight_decay = self.conf.weight_decay
        if weight_decay > 0:
            for key, prm in model.named_parameters():
                if not "bn" in key and weight_decay != 0:
                    prm.grad.data = prm.grad.data.detach().clone() + weight_decay * prm.data.detach().clone()

    
    def __setstate__(self, state):
        super(PProx_SPDA, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def inference(self, model, criterion, _input, _target):
        output = model(_input)
        loss = criterion(output, _target)
        return loss


    def get_lambda(self, param_groups, param_names, rank):
        data = []
        for idx, _ in param_names:
            _data = param_groups[idx]["lambdas"][0][rank]
            if _data is not None:
                data.append(_data)
        flatten_lambda = TensorBuffer(data)
        return data, flatten_lambda
    
    def get_prm(self, param_groups, param_names):
        data = []
        for idx, _ in param_names:
            _data = param_groups[idx]["params"][0]
            if _data is not None:
                data.append(_data)
        flatten_params = TensorBuffer(data)
        return data, flatten_params
    
    def get_zeros_prm_buffer(self, param_groups, param_names):
        _, flatten_params = self.get_prm(param_groups, param_names)
        flatten_params.buffer = torch.zeros_like(flatten_params.buffer)
        return flatten_params
    
    
    def step(self, **kargs):
        self.n_bits = 0
        params, flatten_params = self.get_prm(self.param_groups, self.param_names)

        #  ==== update dual variable lambda from aggregates ==== 
        if not self.helper_thread is None:
            utils.join_thread(self.helper_thread)
            for nei in self.neighbors_info:
                _lambda, flatten_lambda = self.get_lambda(self.param_groups, self.param_names, nei)
                flatten_lambda.buffer *= 1 - self.rho * self.gamma # mingyi
                # flatten_lambda.buffer *= 1 - self.gamma / self.rho # bianchi
                if nei in self.sync_buffer["edge_result"]:
                    sign = 1 if self.rank < nei else -1
                    sparse_values, indices = self.sync_buffer["edge_result"][nei]
                    flatten_lambda.buffer[ indices ] += self.rho * sign * (flatten_params.buffer[indices] - sparse_values) # mingyi
                    # flatten_lambda.buffer[ indices ] += 1/self.rho * sign * (flatten_params.buffer[indices] - sparse_values) # bianchi
                flatten_lambda.unpack(_lambda)
        
        # draw new \xi
        self.compressor.prepare_round(flatten_params)

        self.n_bits += self.sync_buffer.get("n_bits", 0)
        # start compress/sync.
        self.sync_buffer = {
            "original_shapes": self.shapes,
            "flatten_params": flatten_params,
            "edge_result": {},
            "n_bits": 0
        }

        self.helper_thread = utils.HelperThread(
            name=f"_thread_at_epoch_{self.conf.epoch_}.compress",
            func=self.compressor.pipeline,
            # the arguments below will be feeded into the `func`.
            sync_buffer=self.sync_buffer,
        )
        self.helper_thread.start()
        utils.join_thread(self.helper_thread)

        #  ==== update primal variable theta from aggregates ==== 
        
        # diagonal of B^T B
        agg_buffer = self.get_zeros_prm_buffer(self.param_groups, self.param_names)
        agg_buffer.buffer += self.alpha * flatten_params.buffer
        for nei in self.sync_buffer["edge_result"]:
            _, indices = self.sync_buffer["edge_result"][nei]
            agg_buffer.buffer[ indices ] += flatten_params.buffer[indices]

        # off-diagonal of B^T B
        for nei in self.sync_buffer["edge_result"]:
            sparse_values, indices = self.sync_buffer["edge_result"][nei]
            agg_buffer.buffer[indices] += sparse_values
        
        # apply dual variable, i.e., sparse update under the current round mask
        for nei in self.compressor.active_neighbors:
            _, selected_idx = self.sync_buffer["edge_result"][nei]
            sign = 1 if self.rank < nei else -1
            _, flatten_lambda = self.get_lambda(self.param_groups, self.param_names, nei)
            agg_buffer.buffer[selected_idx] -= sign * (1 - self.rho * self.gamma) / self.rho * flatten_lambda.buffer[selected_idx]

        
        # set model prm as prm.grad + weight_decay
        utils.apply_gradient(
            self.param_groups, self.state, apply_grad_to_model=False, set_model_prm_as_grad=True, lr=1/self.rho
        )

        params, flatten_params = self.get_prm(self.param_groups, self.param_names)
        # if self.adaptive_lr:
        #     self.vx += torch.linalg.vector_norm(flatten_params.buffer)**2
        flatten_params.buffer += agg_buffer.buffer

        # update the local model with agg.
        flatten_params.unpack(params)

        # apply D^-1
        params, flatten_params = self.get_prm(self.param_groups, self.param_names)

        empty_buffer = self.get_zeros_prm_buffer(self.param_groups, self.param_names)
        empty_buffer.buffer += self.alpha

        for nei in self.sync_buffer["edge_result"]:
            _, indices = self.sync_buffer["edge_result"][nei]
            empty_buffer.buffer[indices] += 2 

        flatten_params.buffer /= empty_buffer.buffer
        flatten_params.unpack(params)

        
        #  ==== do aggregate for dual variable ==== 
        params, flatten_params = self.get_prm(self.param_groups, self.param_names)

        self.n_bits += self.sync_buffer.get("n_bits", 0)
        # start compress/sync.
        self.sync_buffer = {
            "original_shapes": self.shapes,
            "flatten_params": flatten_params,
            "edge_result": {},
            "n_bits": 0
        }

        self.helper_thread = utils.HelperThread(
            name=f"_thread_at_epoch_{self.conf.epoch_}.compress",
            func=self.compressor.pipeline,
            # the arguments below will be feeded into the `func`.
            sync_buffer=self.sync_buffer,
        )
        self.helper_thread.start()

        if self.conf.epoch_ % 1 == 0:
            utils.join_thread(self.helper_thread)

        # threads behind are aggregating for dual update while this function returns to continue subsequent gradient calculation
        return self.n_bits

class SPDA_VaryingGraph_Sparsifier(object):
    def __init__(
        self,
        aggregator,
        comm_device,
        compress_ratio,
        is_biased,
        backend,
        use_ipc,
        **kargs,
    ):
        # assign the common hyper-parameters
        self.aggregator_fn = aggregator
        self.comm_device = comm_device
        self.compress_ratio = compress_ratio
        self.is_biased = is_biased
        self.backend = backend
        self.use_ipc = use_ipc
        self.kargs = kargs
        self.compressor_fn = SparsificationCompressor()

        # define gossip_stream
        if torch.cuda.is_available():
            self.gossip_stream = torch.cuda.current_stream()

    def pipeline(self, sync_buffer):
        if torch.cuda.is_available():
            with torch.cuda.stream(self.gossip_stream):
                try:
                    self.compress(sync_buffer)
                    self.sync(sync_buffer)
                    self.uncompress(sync_buffer)
                except RuntimeError as e:
                    print("Error: {}".format(e))
        else:
            self.compress(sync_buffer)
            self.sync(sync_buffer)
            self.uncompress(sync_buffer)
    
    
    def prepare_round(self, flatten_params):
        edge_activation = {nei: torch.rand(1) for nei in self.aggregator_fn.neighbor_ranks}
        edge_activation = self.aggregator_fn.one_way_sendrecv(edge_activation, force_wait=True)

        self.active_neighbors = [nei for nei in edge_activation if edge_activation[nei] <= self.kargs["edge_fraction"]]
        self.edge_masks = {nei: self.prepare_compress(flatten_params) for nei in self.active_neighbors}

        n_layers = len(flatten_params)
        self.comm_edge_masks = [{nei: self.edge_masks[nei][i] for nei in self.active_neighbors}
                                                                for i in range(n_layers)]

        for layer_j in range(n_layers):
            self.comm_edge_masks[layer_j] = self.aggregator_fn.one_way_sendrecv(self.comm_edge_masks[layer_j], 
                                            force_wait=True, active_neighbors=self.active_neighbors)
       
        self.edge_masks = {nei: [self.comm_edge_masks[j][nei] for j in range(n_layers)] for nei in self.active_neighbors}

        

    def prepare_compress(self, flatten_params):
        selected_values, selected_indices = [], []

        for param in flatten_params:
            _selected_values, _selected_indices = self.compressor_fn.get_random_k(
                param,
                self.compress_ratio,
                self.is_biased,
            )
            selected_values.append(_selected_values)
            selected_indices.append(_selected_indices)

        return selected_indices
    
    def compress(self, sync_buffer):
        sync_buffer["send_dict"] = {}
        sync_buffer["selected_shapes"] = {}
        for nei in self.active_neighbors:
            selected_indices = self.edge_masks[nei]
            selected_values = []

            for param, _selected_indices in zip(sync_buffer["flatten_params"], selected_indices):
                _selected_values = param.view(-1)[_selected_indices]
                selected_values.append(_selected_values)
            
            selected_shapes = [len(_value) for _value in selected_values]

            flatten_selected_values = TensorBuffer(selected_values)
            flatten_selected_indices = TensorBuffer(selected_indices)

            sync_buffer["send_dict"][nei] = torch.cat(
                [flatten_selected_values.buffer, 
                flatten_selected_indices.buffer]
            )

            sync_buffer["selected_shapes"][nei] = selected_shapes
            if self.comm_device == "cpu":
                sync_buffer["send_dict"][nei] = sync_buffer["send_dict"][nei].cpu().pin_memory()
            
            # get n_bits to transmit.
            if self.aggregator_fn.rank > nei:
                n_bits = get_n_bits(flatten_selected_values.buffer) + get_n_bits(
                    flatten_selected_indices.buffer
                )
            else:
                n_bits = get_n_bits(flatten_selected_values.buffer)
            sync_buffer["n_bits"] += n_bits

    def sync(self, sync_buffer):
        # sync.
        sync_buffer["recv_dict"] = {}
        for rank in sync_buffer["send_dict"]:
            sync_buffer["recv_dict"][rank] = torch.empty_like(sync_buffer["send_dict"][rank])

        sync_message_reqs, synced_message = self.aggregator_fn.two_way_sendrecv(sync_buffer["send_dict"], sync_buffer["recv_dict"], 
                        force_wait=False, active_neighbors=self.active_neighbors)
       
        # update sync_buffer.
        sync_buffer["sync_reqs"] = sync_message_reqs
        sync_buffer["synced_message"] = synced_message
        sync_buffer["sycned_message_size"] = {nei: len(sync_buffer["send_dict"][nei]) for nei in sync_buffer["send_dict"]}

    def uncompress(self, sync_buffer):
        # wait the sync.
        self.aggregator_fn.complete_wait(sync_buffer["sync_reqs"])

        # uncompress and update.
        for rank in self.active_neighbors:
            # tmp_params_memory = neighbor_tmp_params["memory"]
            message_size = int(sync_buffer["sycned_message_size"][rank] / 2)

            # recover values/indices to the correct device.
            q_values, q_indices = self._uncompress_helper(
                sync_buffer["flatten_params"].buffer.device,
                rank,
                sync_buffer["synced_message"],
                message_size,
                sync_buffer["selected_shapes"],
                sync_buffer["original_shapes"],
            )

            # have (rank)-neighbour sparse param here
            sync_buffer["edge_result"][rank] = (q_values, q_indices) # can be used directly on buffer
            
    def _uncompress_helper(
        self,
        _device,
        _rank,
        synced_message,
        sycned_message_size,
        selected_shapes,
        original_shapes,
    ):
        # recover the message and the corresponding device.
        _message = comm.recover_device(
            synced_message[_rank], device=_device
        )
        values = _message[:sycned_message_size]
        indices = _message[sycned_message_size:]

        # deal with unbalanced values/indieces
        q_values, q_indices = self.compressor_fn.uncompress(
            values, indices, selected_shapes[_rank], original_shapes
        )
        return q_values, q_indices