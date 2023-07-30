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

class Di_CS(Optimizer):
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
        super(Di_CS, self).__init__(params, defaults)

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

        self.alpha = conf.alpha
        self.alpha_init = self.alpha
        self.gamma = conf.gamma
        self.B = conf.B_connected
        self.T = conf.outer_loop_T
        self.svrg = conf.SVRG
        self.edge_fraction = conf.edge_fraction
        self.B_round_active_neighbours = set()
        self.it = 0

        # initialize gradient tracker
        for groups in self.param_groups:
            groups["params_prev"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["mu_true_grad"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["surplus_y"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["surplus_y_mem"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["grad_tracker"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["momentum_v"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["momentum_vp"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["tmp"] = [torch.zeros_like(prm) for prm in groups["params"]]

        self.model = model
        self.model_w = deepcopy(
            model.module if "DataParallel" == model.__class__.__name__ else model
        )
        self.model_w.zero_grad()
        
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
        self.compressor = Di_CS_DirectedGraph_Sparsifier(
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
        super(Di_CS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def inference(self, model, criterion, _input, _target):
        output = model(_input)
        loss = criterion(output, _target)
        return loss

    
    def get_buffer(self, param_groups, param_names, param_tag):
        # param_tag example
        # prm x: "params"
        # surplus y: "surplus_y"
        data = []
        for idx, _ in param_names:
            _data = param_groups[idx][param_tag][0]
            if _data is not None:
                data.append(_data)
        flatten_params = TensorBuffer(data)
        return data, flatten_params
    
    def get_zeros_prm_buffer(self, param_groups, param_names):
        _, flatten_params = self.get_buffer(param_groups, param_names, "params")
        flatten_params.buffer = torch.zeros_like(flatten_params.buffer)
        return flatten_params

    def full_grad_handler(self, **kwargs):
        # set model_w as current model.
        for prm, prm_w in zip(self.model.parameters(), self.model_w.parameters()):
            prm_w.data = prm.data.detach().clone()

        self.model_w.zero_grad()
        self.model_w.train()
        num_of_batches = len(kwargs["dataloader"])
        for i, (_input, _target) in enumerate(kwargs["dataloader"]):
            _input, _target = load_data_batch(kwargs["conf"], _input, _target)
            loss = self.inference(self.model_w, kwargs["criterion"], _input, _target) / num_of_batches
            loss.backward()
            if i == num_of_batches - 1:
                break
        self.L2_regularize_grad(self.model_w)

        # lend self.model to store grad of model_w... there should be a better way to implement this...
        for groups in self.param_groups:
            for tmp, prm in zip(groups["tmp"], groups["params"]):
                tmp.data = prm.grad.data.detach().clone()
        for prm, prm_w in zip(self.model.parameters(), self.model_w.parameters()):
            prm.grad.data = prm_w.grad.data.detach().clone()
        
        if self.svrg:
            # init. grad. tracker and momentum_v
            if self.it == 0:
                for groups in self.param_groups:
                    for v, gt, prm in zip(groups["momentum_v"], groups["grad_tracker"], groups["params"]):
                        gt.data = prm.grad.data.detach().clone()
                        v.data = prm.grad.data.detach().clone()
            
            # update mu
            for groups in self.param_groups:
                for mu, prm in zip(groups["mu_true_grad"], groups["params"]):
                    mu.data = prm.grad.data.detach().clone()
        else:
            # update gt
            for groups in self.param_groups:
                for gt, prm in zip(groups["grad_tracker"], groups["params"]):
                    gt.data = prm.grad.data.detach().clone()
        
        # restore model grad from tmp
        for groups in self.param_groups:
            for tmp, prm in zip(groups["tmp"], groups["params"]):
                prm.grad.data = tmp.data.detach().clone()
        

    def update_momentum_v(self, _input, _target, criterion):
        self.model_w.zero_grad()
        self.model_w.train()
        loss = self.inference(self.model_w, criterion, _input, _target)
        loss.backward()
        self.L2_regularize_grad(self.model_w)
        self.L2_regularize_grad(self.model)
        for prm, prm_prev in zip(self.model.parameters(), self.model_w.parameters()):
            prm.grad.data = prm.grad.data.detach().clone() - prm_prev.grad.data.detach().clone()
        
        for groups in self.param_groups:
            for v, vp, prm, mu_true_grad in zip(groups["momentum_v"], groups["momentum_vp"], groups["params"], groups["mu_true_grad"]):
                vp.data = v.data.detach().clone()
                v.data = prm.grad.data.detach().clone() + mu_true_grad
                

    def di_cs_gossip(self, active_neighbors, edge_result, flatten_local_prms):
        renormalization_sum = self.get_zeros_prm_buffer(self.param_groups, self.param_names)
        for nei in active_neighbors:
            if nei in edge_result:
                _, indices = edge_result[nei]
                renormalization_sum.buffer[indices] += self.neighbors_info[nei]
        renormalization_sum.buffer += self.neighbors_info[self.rank]
        
        weighted_avg = self.get_zeros_prm_buffer(self.param_groups, self.param_names)
        for nei in active_neighbors:
            if nei in edge_result:
                sparse_values, indices = edge_result[nei]
                weighted_avg.buffer[indices] += self.neighbors_info[nei] * sparse_values
        weighted_avg.buffer += self.neighbors_info[self.rank] * flatten_local_prms.buffer

        weighted_avg.buffer /= renormalization_sum.buffer
        return weighted_avg
    
    
    def step(self, **kwargs):
        self.n_bits = 0
        if not self.svrg and self.it % self.B == self.B - 1:
            # use diminishing stepsize
            self.alpha = self.alpha_init / ( self.it // self.B )

        #  ==== update dual variable y from aggregates ==== 
        if not self.helper_thread is None:
            utils.join_thread(self.helper_thread)
            _, flatten_params = self.get_buffer(self.param_groups, self.param_names, "params")
            _, flatten_param_prev = self.get_buffer(self.param_groups, self.param_names, "params_prev")
            surplus_y, flatten_surplus_y = self.get_buffer(self.param_groups, self.param_names, "surplus_y")
            flatten_surplus_y.buffer = self.di_cs_gossip(self.compressor.active_neighbors, 
                                                        self.sync_buffer["edge_result"], 
                                                        flatten_surplus_y).buffer
            flatten_surplus_y.buffer -= flatten_params.buffer - flatten_param_prev.buffer
            flatten_surplus_y.unpack(surplus_y)

            if self.it % self.B == 0:
                for groups in self.param_groups:
                    for y_mem, y in zip(groups["surplus_y_mem"], groups["surplus_y"]):
                        y_mem.data = y.data.detach().clone()
        
            # ==== SVRG update for grad_tracker ====
            if self.svrg:
                for nei in self.compressor.active_neighbors:
                    self.B_round_active_neighbours.add(nei)
                
                if self.it % self.B == self.B - 1:
                    grad_tracker, flatten_grad_tracker = self.get_buffer(self.param_groups, self.param_names, "grad_tracker")
                    # aggregate grad_tracker on B_round_active_neighbours (uncompressed)
                    self.n_bits += get_n_bits(flatten_grad_tracker.buffer) * len(self.B_round_active_neighbours)
                    flatten_grad_tracker.buffer = self.aggregator._agg_custom_neighbor(self.B_round_active_neighbours, flatten_grad_tracker.buffer, "weighted")
                    flatten_grad_tracker.unpack(grad_tracker)
                    self.B_round_active_neighbours = set()

                    # draw gradient for momentum v
                    self.update_momentum_v(kwargs["input"], kwargs["target"], kwargs["criterion"])

                    # apply momentum v
                    for groups in self.param_groups:
                        for g, v, vp in zip(groups["grad_tracker"], groups["momentum_v"], groups["momentum_vp"]):
                            g.data += v.data.detach().clone() - vp.data.detach().clone()
                    
            self.it += 1

        # ======================= CONSIDER THIS AS START OF NEW ITERATION =======================
        if self.svrg and self.it % self.T == 0:
            self.full_grad_handler(**kwargs)
        elif not self.svrg and self.it % self.T == 0:
            apply_true_gradient = True
            if apply_true_gradient:
                self.full_grad_handler(**kwargs)
            else:
                # apply weight decay
                utils.apply_gradient(
                    self.param_groups, self.state, apply_grad_to_model=False, set_model_prm_as_grad=False
                )
                for groups in self.param_groups:
                    for g, prm in zip(groups["grad_tracker"], groups["params"]):
                        g.data = prm.grad.data.detach().clone()

        params, flatten_params = self.get_buffer(self.param_groups, self.param_names, "params")

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

        for groups in self.param_groups:
            for prm, prm_prev in zip(groups["params"], groups["params_prev"]):
                prm_prev.data = prm.data.detach().clone()

        #  ==== update model parameters from aggregates ==== 
        
        flatten_params.buffer = self.di_cs_gossip(self.compressor.active_neighbors, 
                                                    self.sync_buffer["edge_result"], 
                                                    flatten_params).buffer
        flatten_params.unpack(params)

        if self.it % self.B == self.B - 1:
            # apply + gamma * y - alpha * g
            for groups in self.param_groups:
                for y, g, prm in zip(groups["surplus_y_mem"], groups["grad_tracker"], groups["params"]):
                    prm.data += self.gamma * y.data.detach().clone() - self.alpha * g.data.detach().clone()
            
        # swap edge mask 
        self.compressor.set_edge_mask_from_edge_result( self.sync_buffer["edge_result"],
                                                        self.sync_buffer["recv_selected_shapes"],
                                                        self.sync_buffer["original_shapes"] )

        #  ==== do aggregate for surplus variable ==== 
        surplus_y, flatten_surplus_y = self.get_buffer(self.param_groups, self.param_names, "surplus_y")

        self.n_bits += self.sync_buffer.get("n_bits", 0)
        # start compress/sync.
        self.sync_buffer = {
            "original_shapes": self.shapes,
            "flatten_params": flatten_surplus_y,
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

        # threads behind are aggregating for surplus_y update while this function returns to continue subsequent gradient calculation
        return self.n_bits

class Di_CS_DirectedGraph_Sparsifier(object):
    def __init__(
        self,
        aggregator,
        comm_device,
        compress_ratio,
        is_biased,
        backend,
        use_ipc,
        **kwargs,
    ):
        # assign the common hyper-parameters
        self.aggregator_fn = aggregator
        self.comm_device = comm_device
        self.compress_ratio = compress_ratio
        self.is_biased = is_biased
        self.backend = backend
        self.use_ipc = use_ipc
        self.kwargs = kwargs
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

        self.active_neighbors = [nei for nei in edge_activation if edge_activation[nei] <= self.kwargs["edge_fraction"]]
        self.edge_masks = {nei: self.prepare_compress(flatten_params) for nei in self.active_neighbors}
        
    def set_edge_mask_from_edge_result(self, edge_result, recv_selected_shapes, original_shapes):
        for nei in self.active_neighbors:
            if nei in edge_result:
                _, indices_buffer = edge_result[nei]
                self.edge_masks[nei] = self.buffer_idx_to_shaped_idx(indices_buffer, recv_selected_shapes[nei], original_shapes)
                

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
        sync_buffer["recv_selected_shapes"] = {}
        for nei in self.active_neighbors:
            selected_indices = self.edge_masks[nei]
            selected_values = []

            for param, _selected_indices in zip(sync_buffer["flatten_params"], selected_indices):
                _selected_values = param.view(-1)[_selected_indices]
                selected_values.append(_selected_values)
            
            selected_shapes = torch.tensor([len(_value) for _value in selected_values])

            flatten_selected_values = TensorBuffer(selected_values)
            flatten_selected_indices = TensorBuffer(selected_indices)
            

            sync_buffer["send_dict"][nei] = torch.cat(
                [flatten_selected_values.buffer, 
                flatten_selected_indices.buffer, selected_shapes]
            )

            sync_buffer["selected_shapes"][nei] = selected_shapes
            if self.comm_device == "cpu":
                sync_buffer["send_dict"][nei] = sync_buffer["send_dict"][nei].cpu().pin_memory()
            
            # get n_bits to transmit.
            n_bits = get_n_bits(flatten_selected_values.buffer) + get_n_bits(
                flatten_selected_indices.buffer
            )
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
            message_size = int((sync_buffer["sycned_message_size"][rank] - len(sync_buffer["original_shapes"]) )/ 2)

            # recover values/indices to the correct device.
            q_values, q_indices, nei_selected_shape = self._uncompress_helper(
                sync_buffer["flatten_params"].buffer.device,
                rank,
                sync_buffer["synced_message"],
                message_size,
                sync_buffer["selected_shapes"],
                sync_buffer["original_shapes"],
            )

            # have (rank)-neighbour sparse param here
            sync_buffer["edge_result"][rank] = (q_values, q_indices) # can be used directly on buffer
            sync_buffer["recv_selected_shapes"][rank] = nei_selected_shape

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
        indices = _message[sycned_message_size:-len(original_shapes)]
        nei_selected_shape = _message[-len(original_shapes):]

        # deal with unbalanced values/indieces
        q_values, q_indices = self.compressor_fn.uncompress(
            values, indices, selected_shapes[_rank], original_shapes
        )
        return q_values, q_indices, nei_selected_shape
    
    def buffer_idx_to_shaped_idx(self, q_indices, selected_shapes, original_shapes):
        # apply each param.
        sync_pointer = 0
        shape_sum = 0

        shaped_indices = []
        for idx, n_sparse_value in enumerate(selected_shapes):
            indices = q_indices[sync_pointer : sync_pointer + int(n_sparse_value)] - shape_sum
            shaped_indices += [indices.long()]

            sync_pointer += int(n_sparse_value)
            shape_sum += original_shapes[idx][1]

        return shaped_indices
