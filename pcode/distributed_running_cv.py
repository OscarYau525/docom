# -*- coding: utf-8 -*-
import gc
import copy
import numpy as np
import torch
from torch.autograd import grad
import torch.distributed as dist
import os.path as osp

from pcode.create_dataset import define_dataset, load_data_batch

from pcode.utils.checkpoint import save_to_checkpoint
from pcode.utils.logging import (
    display_training_stat,
    display_test_stat,
    dispaly_best_test_stat,
    display_consensus_distance,
    display_custom
)
from pcode.utils.stat_tracker import RuntimeTracker
import pcode.utils.error_handler as error_handler
import pcode.utils.auxiliary as auxiliary
from pcode.utils.gradient import gradient_norm, model_norm, log_gt_diff
from pcode.utils.communication import global_tensor_average, global_tensor_sum
from pcode.optim.sam import SAM


def train_and_validate(
    conf, model, criterion, scheduler, optimizer, metrics, data_loader
):
    print("=>>>> start training and validation.\n")

    # define runtime stat tracker and start the training.
    tracker_tr = RuntimeTracker(
        metrics_to_track=metrics.metric_names, on_cuda=conf.graph.on_cuda
    )

    # get the timer.
    timer = conf.timer
    if "global_train_loader" in data_loader:
        global_data_loader = {"train_loader": data_loader["global_train_loader"], 
                            "val_loader": data_loader["global_val_loader"]}
    # break until finish expected full epoch training.
    conf.logger.log_metric(
        name="hyperparameters",
        values={
            "rank": conf.graph.rank,
            "num_batches_train_per_device_per_epoch": conf.num_batches_train_per_device_per_epoch,
            "batch_size": conf.batch_size,
            "total_epochs": conf.num_epochs
        },
        tags={"type": "hyperparameters"},
        display=True
    )
    conf.logger.save_json()

    # create model holders
    global_models = [copy.deepcopy(
            model.module if "DataParallel" == model.__class__.__name__ else model
        ) for _ in range(conf.graph.n_nodes)]
    
    if conf.sam:
        sam_optimizer = SAM(model.parameters(), optimizer)
    else:
        sam_optimizer = None

    if optimizer.__class__.__name__ == "GtHsgd" or "DoCoM" in optimizer.__class__.__name__ or "BEER" in optimizer.__class__.__name__:
        # First step of GT-HSGD / DoCoM
        with timer("opt_init", epoch=0.0):
            if "BEER" in optimizer.__class__.__name__:
                optimizer.init_gradient_tracker(conf, criterion, data_loader["train_loader"], timer=timer)
            else:
                optimizer.init_momentum_tracker(conf, criterion, data_loader["train_loader"], timer=timer, sam_opt=sam_optimizer)
    
        # reshuffle the data.
        if conf.reshuffle_per_epoch and conf.data != "femnist": # femnist reshuffled by dataloader
            print("\nReshuffle the dataset.")
            del data_loader
            gc.collect()
            data_loader = define_dataset(conf)
    
    
    print("=>>>> enter the training.\n")
    dist.barrier()
    while True:
        # dist.barrier()

        # configure local step.
        for _input, _target in data_loader["train_loader"]:
            model.train()
            scheduler.step(optimizer, const_stepsize=conf.const_lr)

            # load data
            with timer("load_data", epoch=scheduler.epoch_):
                _input, _target = load_data_batch(conf, _input, _target)

            # inference and get current performance.
            with timer("forward_pass", epoch=scheduler.epoch_):
                optimizer.zero_grad()
                loss = inference(model, criterion, metrics, _input, _target, tracker_tr)

            with timer("backward_pass", epoch=scheduler.epoch_):
                loss.backward()
            
            if conf.sam:
                with timer("sam_loss", epoch=scheduler.epoch_):
                    # sam_optimizer.first_step(rho=0.05 * scheduler.get_lr() / conf.lr ,zero_grad=True) # climb to local maxima
                    sam_optimizer.first_step(zero_grad=True) # climb to local maxima
                    inference(model, criterion, metrics, _input, _target, tracker_tr).backward()
                    sam_optimizer.second_step(zero_grad=False) # revert model prm to the usual state, keep the SAM gradient for decentralized optimizer

            with timer("sync_complete", epoch=scheduler.epoch_):
                n_bits_to_transmit = optimizer.step(timer=timer, scheduler=scheduler, input=_input, target=_target, criterion=criterion,
                                                    model=model, epoch=scheduler.epoch_, conf=conf, dataloader=data_loader["train_loader"])

            # display the logging info.
            display_training_stat(conf, scheduler, tracker_tr, n_bits_to_transmit, display=conf.graph.rank==0)

            # display tracking time.
            if (
                conf.graph.rank == 0
                and conf.display_tracked_time
                and scheduler.local_index % conf.summary_freq == 0
            ):
                print(timer.summary())
            
            # display consensus distance, for tuning hyperparameters
            # if scheduler.local_index % 10 == 0:
            #     cal_consensus(conf, model, global_models, optimizer, scheduler)
            
            if tracker_tr.stat["loss"].avg > 1e3 or np.isnan(tracker_tr.stat["loss"].avg):
                print("\nThe process diverges!!!!!Early stop it. loss = {}".format(tracker_tr.stat["loss"].avg))
                error_handler.abort()

            # finish one epoch training and to decide if we want to val our model.
            # eval_dec = torch.tensor(scheduler.is_eval(), dtype=torch.int)
            # dist.all_reduce(eval_dec, op=dist.ReduceOp.SUM, async_op=False)
            # print(eval_dec, end="")
            # if eval_dec > conf.graph.n_nodes:
            if scheduler.epoch_ % 1 == 0:
                tracker_tr.reset()
            if scheduler.is_eval():
                # evaluate gradient tracker consensus
                if optimizer.__class__.__name__ in ["GNSD", "GtHsgd", "DeTAG", "ParallelDoCoM_V", "ParallelBEER_V"]:
                    local_gt = optimizer.get_gt()
                    avged_gt = global_tensor_average(local_gt, conf.graph.n_nodes, conf.on_cuda)
                    log_gt_diff(conf, scheduler, local_gt, avged_gt)
                
                # each worker finish one epoch training.
                if not conf.train_fast and not conf.skip_eval:
                    if not conf.clean_output or conf.graph.rank == 0:
                        conf.logger.log("epoch {}: eval the local model on local training data.".format(scheduler.epoch_)) # to decide the local best model (in real sceneario agent dont access global data)
                    do_validate(conf, model, optimizer, criterion, scheduler, metrics, data_loader, is_val=True)
                
                elif conf.save_epoch_models:
                    # save model at end of epoch
                    if not conf.clean_output or conf.graph.rank == 0:
                        conf.logger.log("epoch {}: save model".format(conf.epoch_))
                    torch.save(model, osp.join(conf.checkpoint_dir, "model{}_e{}.pt".format(conf.graph.rank, int(conf.epoch_))))

                # refresh the logging cache at the begining of each epoch.
                tracker_tr.reset()

                # evaluate (and only inference) on the whole training loader.
                if not conf.train_fast and not conf.skip_eval:

                    # evaluate on the local model.
                    if not conf.eval_consensus_only or (conf.eval_consensus_only and scheduler.is_stop()):
                        all_gather_models_and_local_eval_and_cal_consensus(
                            conf,
                            model,
                            optimizer,
                            criterion,
                            scheduler,
                            metrics,
                            data_loader=data_loader,
                            global_models=global_models
                        )

                    else:
                        consensus_distance(conf, model, optimizer, scheduler)

                # determine if the training is finished.
                if scheduler.is_stop():
                    # save json.
                    conf.logger.save_json()
                    # upload jsons to gcloud
                    if not conf.gcloud_bucket is None:
                        conf.logger.upload_gcloud(conf.gcloud_bucket)
                    # temporarily hack the exit parallelchoco
                    if optimizer.__class__.__name__ == "ParallelCHOCO" or optimizer.__class__.__name__ == "ParallelDoCoM":
                        error_handler.abort()
                    return

            

        # reshuffle the data.
        if conf.reshuffle_per_epoch and conf.data != "femnist" and conf.data != "tomshardware": # custom dataset reshuffled by dataloader
            print("\nReshuffle the dataset.")
            del data_loader
            gc.collect()
            data_loader = define_dataset(conf)


def inference(model, criterion, metrics, _input, _target, tracker=None, weight_decay=1e-4, backward=False):
    """Inference on the given model and get loss and accuracy."""
    output = model(_input)
    loss = criterion(output, _target)
    if backward:
        loss.backward()
    performance = metrics.evaluate(loss, output, _target)
    weight_decay_loss = weight_decay * model_norm(model)**2
    if tracker is not None:
        tracker.update_metrics([loss.item() + weight_decay_loss] + performance, n_samples=_input.size(0))
    return loss



def do_validate(conf, model, optimizer, criterion, scheduler, metrics, data_loader, is_val=True):
    """Evaluate the model on the test dataset and save to the checkpoint."""
    # wait until the whole group enters this function, and then evaluate.
    print("Enter validation phase.")
    performance = validate(
        conf, model, optimizer, criterion, scheduler, metrics, data_loader, is_val=is_val
    )

    # remember best performance and display the val info.
    scheduler.best_tracker.update(performance[0], scheduler.epoch_)
    dispaly_best_test_stat(conf, scheduler)

    # save to the checkpoint.
    if not conf.train_fast:
        save_to_checkpoint(
            conf,
            {
                "arch": conf.arch,
                "current_epoch": scheduler.epoch,
                "local_index": scheduler.local_index,
                "best_perf": scheduler.best_tracker.best_perf,
                "optimizer": optimizer.state_dict(),
                "state_dict": model.state_dict(),
            },
            scheduler.best_tracker.is_best,
            dirname=conf.checkpoint_dir,
            filename="checkpoint.pth.tar",
            save_all=conf.save_all_models,
        )
    print("Finished validation.")


def validate(
    conf,
    model,
    optimizer,
    criterion,
    scheduler,
    metrics,
    data_loader,
    label="local_model",
    is_val=True,
):
    """A function for model evaluation."""

    def _evaluate(_model, optimizer, label):
        # define stat.
        tracker_te = RuntimeTracker(
            metrics_to_track=metrics.metric_names, on_cuda=conf.graph.on_cuda
        )

        # switch to evaluation mode
        _model.eval()
        dloader = data_loader["val_loader"] if is_val else data_loader["train_loader"]
        for _input, _target in dloader:
            # load data and check performance.
            _input, _target = load_data_batch(conf, _input, _target)

            with torch.no_grad():
                inference(_model, criterion, metrics, _input, _target, tracker_te)

        # display the test stat.
        display_test_stat(conf, scheduler, tracker_te, label)

        # get global (mean) performance
        global_performance = tracker_te.evaluate_global_metrics()
        return global_performance

    # evaluate each local model on the validation dataset.
    global_performance = _evaluate(model, optimizer, label=label)
    return global_performance

def cal_consensus(conf, model, global_models, optimizer, scheduler):
    # all gather models
    my_copied_model = copy.deepcopy(
        model.module if "DataParallel" == model.__class__.__name__ else model
    )

    if not conf.clean_output or conf.graph.rank == 0:
        conf.logger.log("epoch {}: all gather models.".format(conf.epoch_))
    avg_model = optimizer.world_aggregator.all_gather_model(global_models, my_copied_model)

    # get the l2 distance of the local model to the averaged model
    consensus_dist = auxiliary.get_model_difference(model, avg_model)
    display_consensus_distance(conf, scheduler, consensus_dist)


def consensus_distance(conf, model, optimizer, scheduler):
    if (
        conf.graph_topology != "complete"
        and not conf.train_fast
    ):
        copied_model = copy.deepcopy(
            model.module if "DataParallel" == model.__class__.__name__ else model
        )
        optimizer.world_aggregator.agg_model(copied_model, op="avg", communication_scheme="all_reduce")

        # get the l2 distance of the local model to the averaged model
        conf.logger.log_metric(
            name="stat",
            values={
                "rank": conf.graph.rank,
                "epoch": scheduler.epoch_,
                "distance": auxiliary.get_model_difference(model, copied_model),
            },
            tags={"split": "test", "type": "averaged_model"},
            display=True
        )
        conf.logger.save_json()


def all_gather_models_and_local_eval_and_cal_consensus(  
    conf,
    model,
    optimizer,
    criterion,
    scheduler,
    metrics,
    data_loader,
    global_models
):
    """"Use centralized aggregator to get all models, eval all models on local dataset, 
    and aggregate the performance metrics (memory constrained approach)"""

    def _evaluate(_model, rank, is_val=False):
        # define stat.
        tracker_te = RuntimeTracker(
            metrics_to_track=metrics.metric_names, on_cuda=conf.graph.on_cuda
        )

        # switch to evaluation mode for logging grad
        _model._eval_layers()
        _model.zero_grad()

        dloader = data_loader["val_loader"] if is_val else data_loader["train_loader"]
        n_samples = 0
        for _input, _target in dloader:
            # load data and check performance.
            _input, _target = load_data_batch(conf, _input, _target)
            n_samples += _input.size(0)
            inference(_model, criterion, metrics, _input, _target, tracker_te, backward=True) 

        # aggregate gradient to get gradient of global function
        optimizer.world_aggregator.agg_grad(_model, op="sum", communication_scheme="reduce", dst_rank=rank)

        tracker_dict = tracker_te.get_sum()
        return tracker_dict["top1"][0], tracker_dict["loss"][0], n_samples
    
    def _eval_wrapper(conf, scheduler, global_models, label, is_val):
        performance_list = []
        for rank, agent_model in enumerate(global_models):
            agent_local_top1_sum, agent_local_loss_sum, n_samples = _evaluate(agent_model, rank, is_val)
            performance_list.append([agent_local_top1_sum, agent_local_loss_sum, n_samples])
        
        all_performances = global_tensor_sum(performance_list, conf.on_cuda) # averaged among all local datasets, allreduced all models

        global_n_samples = all_performances[conf.graph.rank][-1].item()
        perf_log = {"top1": all_performances[conf.graph.rank][0].item() / global_n_samples,
                    "loss": all_performances[conf.graph.rank][1].item() / global_n_samples,
                    "grad_norm": gradient_norm(global_models[conf.graph.rank], weight_decay=conf.weight_decay) / global_n_samples}
        
        display_custom(conf, scheduler, perf_log, label)
        

    if not conf.train_fast:  
        # all gather models
        my_copied_model = copy.deepcopy(
            model.module if "DataParallel" == model.__class__.__name__ else model
        )

        if not conf.clean_output or conf.graph.rank == 0:
            conf.logger.log("epoch {}: all gather models.".format(conf.epoch_))
        avg_model = optimizer.world_aggregator.all_gather_model(global_models, my_copied_model)

        # get the l2 distance of the local model to the averaged model
        consensus_dist = auxiliary.get_model_difference(model, avg_model)
        display_consensus_distance(conf, scheduler, consensus_dist)

        label_val = "eval_local_model_on_full_testing_data"
        label_train = "eval_local_model_on_full_training_data"

        if not conf.clean_output or conf.graph.rank == 0:
            conf.logger.log("epoch {}: eval all models on local training data".format(conf.epoch_))
        _eval_wrapper(conf, scheduler, global_models, label_train, False)
        if not conf.clean_output or conf.graph.rank == 0:
            conf.logger.log("epoch {}: eval all models on local testing data".format(conf.epoch_))
        _eval_wrapper(conf, scheduler, global_models, label_val, True)
