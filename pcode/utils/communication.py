# -*- coding: utf-8 -*-
import torch
import torch.distributed as dist
import copy

"""some auxiliary functions for communication."""


def global_average(sum, count, on_cuda=True):
    def helper(array):
        array = torch.FloatTensor(array)
        array = array.cuda() if on_cuda else array
        dist.all_reduce(array, op=dist.ReduceOp.SUM)
        all_sum, all_count = array
        if all_count == 0:
            return 0
        else:
            return all_sum / all_count

    avg = helper([sum, count])
    return avg

def global_tensor_average(_sum, world_size, on_cuda=True):
    def helper(array):
        array = torch.FloatTensor(array)
        array = array.cuda() if on_cuda else array
        dist.all_reduce(array, op=dist.ReduceOp.SUM)
        return array / world_size

    avg = helper(_sum)
    return avg

def global_tensor_sum(_sum, on_cuda=True):
    def helper(array):
        array = torch.FloatTensor(array)
        array = array.cuda() if on_cuda else array
        dist.all_reduce(array, op=dist.ReduceOp.SUM)
        return array

    _sum = helper(_sum)
    return _sum

def elementwise_min(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return tensor


def broadcast(tensor, src):
    return dist.broadcast(tensor, src=src)


"""some aggregation functions."""


def _get_data(param_groups, idx, is_get_grad, get_name):
    # Define the function to get the data.
    # when we create the param_group, each group only has one param.
    if is_get_grad:
        return param_groups[idx]["params"][0].grad
    return param_groups[idx][get_name][0]


def _get_shape(param_groups, idx):
    return param_groups[idx]["param_size"], param_groups[idx]["nelement"]


def get_data(param_groups, param_names, is_get_grad=True, is_get_grad_tracker=False, get_name=None):
    data, shapes = [], []
    if get_name is None and is_get_grad_tracker:
        get_name = "grad_tracker"
    elif get_name is None:
        get_name = "params"
    for idx, _ in param_names:
        _data = _get_data(param_groups, idx, is_get_grad, get_name)
        if _data is not None:
            data.append(_data)
            shapes.append(_get_shape(param_groups, idx))
    return data, shapes


def flatten(tensors, shapes=None, use_cuda=True):
    # init and recover the shapes vec.
    pointers = [0]
    if shapes is not None:
        for shape in shapes:
            pointers.append(pointers[-1] + shape[1])
    else:
        for tensor in tensors:
            pointers.append(pointers[-1] + tensor.nelement())

    # flattening.
    vec = torch.empty(
        pointers[-1],
        device=tensors[0].device if tensors[0].is_cuda and use_cuda else "cpu",
    )

    for tensor, start_idx, end_idx in zip(tensors, pointers[:-1], pointers[1:]):
        vec[start_idx:end_idx] = tensor.data.view(-1)
    return vec


def unflatten(tensors, synced_tensors, shapes):
    pointer = 0

    for tensor, shape in zip(tensors, shapes):
        param_size, nelement = shape
        tensor.data[:] = synced_tensors[pointer : pointer + nelement].view(param_size)
        pointer += nelement


"""auxiliary."""


def recover_device(data, device=None):
    if device is not None:
        return data.to(device)
    else:
        return data


"""main aggregators."""


class Aggregation(object):
    """Aggregate udpates / models from different processes."""

    def _agg(self, data, op):
        """Aggregate data using `op` operation.
        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        raise NotImplementedError

    def agg_model(self, model, op, communication_scheme):
        """Aggregate models by model weight.
        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
            communication_scheme (str): all_reduce/all_gather/reduce
        """
        # Aggregate layer by layer
        for _, param in enumerate(model.parameters()):
            param.data = self._agg(param.data, op=op, communication_scheme=communication_scheme)
    
 
    def agg_grad(self, model, op, communication_scheme, **kwargs):
        """Aggregate models gradients.
        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
            communication_scheme (str): all_reduce/all_gather/reduce
        """
        # Aggregate layer by layer
        for _, param in enumerate(model.parameters()):
            grad = self._agg(param.grad.data, op=op, communication_scheme=communication_scheme, **kwargs)
            param.grad.data = grad


    def all_gather_model(self, recver_model_list, model):
        # output all agents model to recver_model_list, and return the averaged model
        layers_models_list = []
        avg_model = copy.deepcopy(
            model.module if "DataParallel" == model.__class__.__name__ else model
        )
        for _, param in enumerate(avg_model.parameters()):
            param.data = torch.zeros_like(param.data)

        for _, param in enumerate(model.parameters()):
            layer_list = self._agg(param.data, communication_scheme="all_gather")
            layers_models_list.append(layer_list)

        for i, recver_model in enumerate(recver_model_list): # the ith agent's model
            for recver_param, recved_model_param_list in zip(recver_model.parameters(), layers_models_list): # layer by layer
                recver_param.data = recved_model_param_list[i]
        
        for recved_model_param_list, avg_model_param in zip(layers_models_list, avg_model.parameters()): # layer by layer
            layer_param_avg = sum([param.data for param in recved_model_param_list]) / len(recved_model_param_list)
            avg_model_param.data = layer_param_avg
        
        return avg_model

    


class CentralizedAggregation(Aggregation):
    """Aggregate udpates / models from different processes."""

    def __init__(self, rank, world, neighbors_info):
        # init
        self.rank = rank

        # define the dist group.
        neighbor_ranks = list(neighbors_info.keys())
        if len(neighbor_ranks) == 0:
            self.group = None
        else:
            self.group = dist.new_group(neighbor_ranks)

        # get the world size from the view of the current rank.
        self.world_size = float(len(neighbor_ranks))

    def _agg(
        self,
        data,
        op=None,
        distributed=True,
        communication_scheme="all_reduce",
        async_op=False,
        **kargs,
    ):
        """Aggregate data using `op` operation.
        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        if not distributed:
            return data

        # do the real sync.
        if communication_scheme == "all_reduce":
            if op == "avg":
                req = dist.all_reduce(
                    data, op=dist.ReduceOp.SUM, group=self.group, async_op=async_op
                )
            elif op == "sum":
                req = dist.all_reduce(
                    data, op=dist.ReduceOp.SUM, group=self.group, async_op=async_op
                )
            else:
                raise NotImplementedError

            if async_op:
                # it would be dangerous to use `avg` operation with async.
                return data, req
            else:
                if op == "avg":
                    return data / self.world_size
                else:
                    return data
        elif communication_scheme == "reduce":
            if op == "sum" or op == "avg":
                req = dist.reduce(
                    data,
                    dst=kargs["dst_rank"],
                    op=dist.ReduceOp.SUM,
                    group=self.group,
                    async_op=async_op,
                )
            else:
                raise NotImplementedError

            if async_op:
                return data, req
            else:
                if op == "sum":
                    return data
                elif op == "avg":
                    return data / self.world_size
                else:
                    raise NotImplementedError
        elif communication_scheme == "all_gather":
            gathered_list = [
                torch.empty_like(data) for _ in range(int(self.world_size))
            ]
            req = dist.all_gather(
                gathered_list, data, group=self.group, async_op=async_op
            )
            if async_op:
                return gathered_list, req
            else:
                return gathered_list
        else:
            raise NotImplementedError

    def complete_wait(self, req):
        req.wait()


class DecentralizedAggregation(Aggregation):
    """Aggregate updates in a decentralized manner."""

    def __init__(self, rank, neighbors_info):
        # init
        self.rank = rank
        self.neighbors_info = neighbors_info
        self.neighbor_ranks = [
            neighbor_rank
            for neighbor_rank in neighbors_info.keys()
            if neighbor_rank != rank
        ]

        # get the world size from the view of the current rank.
        self.world_size = float(len(self.neighbor_ranks) + 1)

    def _agg(self, data, op, force_wait=True):
        """Aggregate data using `op` operation.
        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, `weighted`, etc.
        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        # Create some tensors to host the values from neighborhood.
        local_data = {i: torch.empty_like(data) for i in self.neighbor_ranks}
        local_data[self.rank] = data

        # async send data.
        reqs = []
        for node_rank in self.neighbor_ranks:
            reqs.append(dist.isend(tensor=local_data[self.rank], dst=node_rank))
            reqs.append(dist.irecv(tensor=local_data[node_rank], src=node_rank))

        # wait until finish.
        if force_wait:
            self.complete_wait(reqs)

            # Aggregate local_data
            if op == "avg":
                output = sum(local_data.values()) / self.world_size
            elif op == "weighted":
                output = sum(
                    [
                        tensor * self.neighbors_info[rank]
                        for rank, tensor in local_data.items()
                    ]
                )
            elif op == "get_raw_sync_data":
                output = local_data
            else:
                raise NotImplementedError("op {} is not supported yet.".format(op))
            return output
        else:
            if op == "get_raw_sync_data":
                return reqs, local_data
            else:
                raise NotImplementedError("op {} is not supported yet.".format(op))
    
    def _agg_custom_neighbor(self, neighbor_set, data, op):
        # record original network
        rec_neighbors_info = self.neighbors_info
        rec_neighbor_ranks = self.neighbor_ranks

        # load custom network and reweight
        self.neighbor_ranks = list(neighbor_set)
        reweight_sum = sum([rec_neighbors_info[nei] for nei in self.neighbor_ranks])
        reweight_sum += rec_neighbors_info[self.rank]
        self.neighbors_info = {nei: rec_neighbors_info[nei] / reweight_sum for nei in self.neighbor_ranks}
        self.neighbors_info[self.rank] = rec_neighbors_info[self.rank] / reweight_sum

        agg_data = self._agg(data, op, force_wait=True)

        # restore original network
        self.neighbors_info = rec_neighbors_info
        self.neighbor_ranks = rec_neighbor_ranks
        return agg_data
    
    def one_way_sendrecv(self, data_dict, force_wait=True, **kwargs):
        # makes two agents agree on one value on each edge
        # node with higher rank is sender

        # async send data.
        reqs = []
        active_neighbors = kwargs.get("active_neighbors", self.neighbor_ranks)
        for node_rank in active_neighbors:
            if node_rank < self.rank:
                reqs.append(dist.isend(tensor=data_dict[node_rank], dst=node_rank))
            else:
                reqs.append(dist.irecv(tensor=data_dict[node_rank], src=node_rank))
        if force_wait:
            self.complete_wait(reqs)
        return data_dict
    
    def two_way_sendrecv(self, send_dict, recv_dict, force_wait=True, **kwargs):
        # two agents exchange values on data_dict

        # async send data.
        reqs = []
        active_neighbors = kwargs.get("active_neighbors", self.neighbor_ranks)
        for node_rank in active_neighbors:
            reqs.append(dist.isend(tensor=send_dict[node_rank], dst=node_rank))
            reqs.append(dist.irecv(tensor=recv_dict[node_rank], src=node_rank))
        if force_wait:
            self.complete_wait(reqs)
        return reqs, recv_dict

    def complete_wait(self, reqs):
        for req in reqs:
            req.wait()


class EfficientDecentralizedAggregation(Aggregation):
    """Aggregate updates in a decentralized manner."""

    def __init__(self, world, rank, neighbors_info, graph):
        # init
        self.rank = rank
        self.world = world
        self.graph = graph

        self.neighbors_info = neighbors_info
        self.neighbor_ranks = [
            neighbor_rank
            for neighbor_rank in neighbors_info.keys()
            if neighbor_rank != rank
        ]
        self.out_edges, self.in_edges = graph.get_edges()

    def _agg(self, data, op, force_wait=True):
        """Aggregate data using `op` operation.
        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, `weighted`, etc.
        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        data = data.detach().clone()
        self.in_buffer = {i: torch.empty_like(data) for i in self.neighbor_ranks}
        self.in_buffer[self.rank] = data

        # async send data.
        out_reqs, in_reqs = [], []
        for out_edge, in_edge in zip(self.out_edges, self.in_edges):
            out_req = dist.broadcast(
                tensor=self.in_buffer[self.rank],
                src=out_edge.src,
                group=out_edge.process_group,
                async_op=True,
            )
            out_reqs.append(out_req)
            in_reqs = []
            in_req = dist.broadcast(
                tensor=self.in_buffer[in_edge.src],
                src=in_edge.src,
                group=in_edge.process_group,
                async_op=True,
            )
            in_reqs.append(in_req)
        return [out_reqs, in_reqs], self.in_buffer

    def complete_wait(self, reqs):
        out_reqs, in_reqs = reqs

        while len(out_reqs) > 0:
            req = out_reqs.pop()
            req.wait()

        while len(in_reqs) > 0:
            req = in_reqs.pop()
            req.wait()


def get_aggregators(cur_rank, world, neighbors_info, aggregator_type, graph=None):
    if "centralized" == aggregator_type:
        return CentralizedAggregation(cur_rank, world, neighbors_info)
    elif "decentralized" == aggregator_type:
        return DecentralizedAggregation(cur_rank, neighbors_info)
    elif "efficient_decentralized" == aggregator_type:
        return EfficientDecentralizedAggregation(
            world=world, rank=cur_rank, neighbors_info=neighbors_info, graph=graph
        )
    else:
        raise NotImplementedError
