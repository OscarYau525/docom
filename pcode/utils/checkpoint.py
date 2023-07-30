# -*- coding: utf-8 -*-
import gc
import shutil
import time
from os.path import join, isfile

import torch

import pcode.utils.logging as logging
from pcode.utils.op_paths import build_dirs, remove_folder
from pcode.utils.op_files import write_pickle


def get_checkpoint_folder_name(conf):
    # get time_id
    time_id = str(int(time.time()))

    # get communication info.
    if conf.comm_op is None:
        comm_info = ""
    elif "compress" in conf.comm_op:
        comm_info = "{}-{:.2f}_".format(conf.comm_op, 1-conf.compress_ratio)
        # comm_info += "warmup_epochs-{}".format(conf.compress_warmup_epochs)
        comm_info += "_mask_momentum" if conf.mask_momentum else ""
        comm_info += (
            "_clip_grad-{}".format(conf.clip_grad_val) if conf.clip_grad else ""
        )
    elif conf.comm_op == "quantize_qsgd":
        comm_info = "{}-{}_".format(conf.comm_op, conf.quantize_level)
    elif conf.comm_op == "sign":
        comm_info = "{}_".format(conf.comm_op)
    else:
        comm_info = ""

    # get optimizer info.
    if "choco" in conf.optimizer or "beer" in conf.optimizer:
        optim_info = "{}_stepsize-{}".format(conf.optimizer, conf.consensus_stepsize)
        if not conf.gamma_decay is None:
            optim_info = "{}_gammadecay-{}".format(optim_info, conf.gamma_decay)
    elif "docom" in conf.optimizer:
        optim_info = "{}_stepsize-{}_beta-{}".format(conf.optimizer, conf.consensus_stepsize, conf.momentum_beta)
        if not conf.gamma_decay is None:
            optim_info = "{}_gammadecay-{}".format(optim_info, conf.gamma_decay)
        if not conf.beta_decay is None:
            optim_info = "{}_betasch-{}_betadecay-{}".format(optim_info, conf.beta_change_epochs, conf.beta_decay)
    elif "gt_hsgd" == conf.optimizer:
        optim_info = "{}_beta-{}_initb-{}".format(conf.optimizer, conf.momentum_beta, conf.initial_batch_num)
        if not conf.beta_decay is None:
            optim_info = "{}_betasch-{}_betadecay-{}".format(optim_info, conf.beta_change_epochs, conf.beta_decay)
    elif "detag" == conf.optimizer:
        optim_info = "{}_gossip_eta-{}_gossip_rounds-{}".format(conf.optimizer, conf.gossip_eta, conf.gossip_rounds)
    elif conf.optimizer == "pprox_spda":
        optim_info = "{}_rho-{}_gamma-{}_alpha-{}_edgefrac-{}".format(conf.optimizer, conf.rho, conf.gamma, conf.alpha, conf.edge_fraction)
    elif conf.optimizer == "fully_spda":
        optim_info = "{}_gamma-{}_alpha-{}_edgefrac-{}".format(conf.optimizer, conf.gamma, conf.alpha, conf.edge_fraction)
    elif conf.optimizer == "fsppd":
        optim_info = "{}_gamma-{}_edgefrac-{}".format(conf.optimizer, conf.gamma, conf.edge_fraction)
    elif conf.optimizer == "di_cs":
        opt_name = conf.optimizer
        if conf.SVRG:
            opt_name += "_svrg"
        else:
            opt_name += "_gd"
        optim_info = "{}_alpha-{}_gamma-{}_T-{}_B-{}_edgefrac-{}".format(opt_name, conf.alpha, conf.gamma, conf.outer_loop_T, conf.B_connected, conf.edge_fraction)
    elif conf.optimizer == "diging":
        optim_info = conf.optimizer + "_edgefrac-{}".format(conf.edge_fraction)
    else:
        optim_info = "{}".format(conf.optimizer)
    if conf.sam:
        optim_info += "_SAM"
    # concat them together.
    return (
        time_id
        + "_l2-{}_lr-{}_it-{}_epochs-{}_batchsize-{}_agents_{}_topo-{}_seed-{}_lrsch-{}_lrdecay-{}_optim-{}_comp-{}".format(
            conf.weight_decay,
            conf.lr,
            conf.num_iterations,
            conf.num_iterations / conf.eval_freq,
            conf.batch_size,
            conf.n_mpi_process,
            conf.graph_topology,
            conf.manual_seed,
            conf.lr_change_epochs,
            conf.lr_decay,
            optim_info,
            comm_info,
        )
    )


def init_checkpoint(conf):
    # init checkpoint dir.
    conf.checkpoint_root = join(
        conf.checkpoint,
        conf.data,
        conf.arch,
        conf.experiment if conf.experiment is not None else "",
        conf.timestamp,
    )
    conf.checkpoint_dir = join(conf.checkpoint_root, str(conf.graph.rank))
    if conf.save_some_models is not None:
        conf.save_some_models = conf.save_some_models.split(",")

    # if the directory does not exists, create them.
    build_dirs(conf.checkpoint_dir)


def _save_to_checkpoint(state, dirname, filename):
    checkpoint_path = join(dirname, filename)
    torch.save(state, checkpoint_path)
    return checkpoint_path


def save_arguments(conf):
    # save the configure file to the checkpoint.
    if conf.graph.rank == 0:
        write_pickle(conf, path=join(conf.checkpoint_root, "arguments.pickle"))


def save_to_checkpoint(conf, state, is_best, dirname, filename, save_all=False):
    # save full state.
    checkpoint_path = _save_to_checkpoint(state, dirname, filename)
    best_model_path = join(dirname, "model_best.pth.tar")
    if is_best:
        shutil.copyfile(checkpoint_path, best_model_path)
    if save_all:
        shutil.copyfile(
            checkpoint_path,
            join(dirname, "checkpoint_epoch_%s.pth.tar" % state["current_epoch"]),
        )
    elif conf.save_some_models is not None:
        if str(state["current_epoch"]) in conf.save_some_models:
            shutil.copyfile(
                checkpoint_path,
                join(dirname, "checkpoint_epoch_%s.pth.tar" % state["current_epoch"]),
            )


def maybe_resume_from_checkpoint(conf, model, optimizer, scheduler):
    if conf.resume:
        if conf.checkpoint_index is not None:
            # reload model from a specific checkpoint index.
            checkpoint_index = "_epoch_" + conf.checkpoint_index
        else:
            # reload model from the latest checkpoint.
            checkpoint_index = ""
        checkpoint_path = join(
            conf.resume,
            str(conf.graph.rank),
            "checkpoint{}.pth.tar".format(checkpoint_index),
        )
        print("try to load previous model from the path:{}".format(checkpoint_path))

        if isfile(checkpoint_path):
            print(
                "=> loading checkpoint {} for {}".format(conf.resume, conf.graph.rank)
            )

            # get checkpoint.
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # restore some run-time info.
            scheduler.update_from_checkpoint(checkpoint)

            # reset path for log.
            try:
                remove_folder(conf.checkpoint_root)
            except RuntimeError as e:
                print(f"ignore the error={e}")
            conf.checkpoint_root = conf.resume
            conf.checkpoint_dir = join(conf.resume, str(conf.graph.rank))
            # restore model.
            model.load_state_dict(checkpoint["state_dict"])
            # restore optimizer.
            optimizer.load_state_dict(checkpoint["optimizer"])
            # logging.
            print(
                "=> loaded model from path '{}' checkpointed at (epoch {})".format(
                    conf.resume, checkpoint["current_epoch"]
                )
            )
            # configure logger.
            conf.logger = logging.Logger(conf.checkpoint_dir)

            # try to solve memory issue.
            del checkpoint
            torch.cuda.empty_cache()
            gc.collect()
            return
        else:
            print("=> no checkpoint found at '{}'".format(conf.resume))
