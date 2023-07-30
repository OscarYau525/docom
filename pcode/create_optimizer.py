# -*- coding: utf-8 -*-
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F
import torch

from pcode.optim.sgd import SGD

from pcode.optim.dgc import DGC
from pcode.optim.parallel_choco import ParallelCHOCO
from pcode.optim.parallel_choco_v import ParallelCHOCO_V
from pcode.optim.ef_sign_sgd import EF_SignSGD
from pcode.optim.dcd_psgd import DCD_PSGD
from pcode.optim.ecd_psgd import ECD_PSGD
from pcode.optim.deep_squeeze import DeepSqueeze
from pcode.optim.gnsd import GNSD
from pcode.optim.detag import DeTAG
from pcode.optim.gt_hsgd import GtHsgd
from pcode.optim.parallel_docom_v import ParallelDoCoM_V
from pcode.optim.parallel_beer_v import ParallelBEER_V
from pcode.optim.pprox_spda import PProx_SPDA
from pcode.optim.fully_spda import Fully_SPDA
from pcode.optim.fsppd import FSPPD
from pcode.optim.di_cs import Di_CS
from pcode.optim.diging import DIGing


def define_optimizer(conf, model):
    # define the param to optimize.
    params = [
        {
            "params": [value],
            "name": key,
            "weight_decay": conf.weight_decay if "bn" not in key else 0.0,
            "param_size": value.size(),
            "nelement": value.nelement(),
        }
        for key, value in model.named_parameters()
    ]

    # define the optimizer.
    opt_dict = {
        "sgd": SGD,
        "dgc": DGC,
        "dcd_psgd": DCD_PSGD,
        "ecd_psgd": ECD_PSGD,
        "parallel_choco": ParallelCHOCO,
        "parallel_choco_v": ParallelCHOCO_V,
        "ef_sign_sgd": EF_SignSGD,
        "deep_squeeze": DeepSqueeze,
        "gnsd": GNSD,
        "detag": DeTAG,
        "gt_hsgd": GtHsgd,
        "docom_v": ParallelDoCoM_V,
        "beer_v": ParallelBEER_V,
        "pprox_spda": PProx_SPDA,
        "fully_spda": Fully_SPDA,
        "fsppd": FSPPD,
        "di_cs": Di_CS,
        "diging": DIGing
    }
    if conf.optimizer in opt_dict:
        optim_class = opt_dict[conf.optimizer]
    else:
        raise NotImplementedError

    return optim_class(
        params,
        lr=conf.lr,
        momentum=conf.momentum_factor,
        nesterov=conf.use_nesterov,
        conf=conf,
        model=model
    )
