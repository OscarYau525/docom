import json
import os.path as osp
import os
import numpy as np
import argparse
import wandb
from datetime import datetime
from math import log10
from copy import deepcopy

uncompressed_net_unit = {"choco":  2,
                        "gt_hsgd":  2 * 2,
                        "docom":  2 * 2,
                        "gnsd":  2 * 2,
                        "detag":  2 * 2,
                        "sgd":  2,
                        "csgd": 1,
                        "beer":  2 * 2}
init_full_batch_size = {"docom_aistats_lenet_femnist": 805263,
                        "docom_tmlr_femnist": 805263,
                        "docom_aistats_1layer_mnist": 60000,
                        "pprox_spda_ff_mnist": 60000,
                        "docom_cifar100_resnet20": 60000}

# grad_unit_batches = {"choco": 1, "gt_hsgd": 2, "docom": 2, "gnsd": 1, "detag": 1, "sgd": 1, "csgd": 1, "beer": 1}

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
def get_time_diff(start_str, end_str):
    start_time = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
    delta = end_time - start_time
    return delta.total_seconds()

def ls_cmd(ls):
    names = os.popen(ls).read().split("\n")
    names = names[:-1] if names[-1] == "" else names
    return names


def get_num_batches_train_per_device_per_epoch(results_dir):
    with open(osp.join(results_dir, "0", "log-1.json")) as f:
        logs = json.load(f)
        if "num_batches_train_per_device_per_epoch" in logs[0]:
            x = logs[0]["num_batches_train_per_device_per_epoch"]
        else:
            raise ValueError("check here: no number of batches")
    return x


def filter_logs(results_dir, tags_to_log, hypprms, log_eval=False):
    all_logs = []
    nodes_dir = ls_cmd("ls -d {}/*/".format(results_dir))
    

    for node_dir in nodes_dir:
        json_results = ls_cmd("ls -d {}*.json".format(node_dir))
        num_jsons = len(json_results)
        fil_log = {k: {m: [] for m in tags_to_log[k]} for k in tags_to_log} # one log for one agent
        
        if log_eval:
            n_points = 20
            log_cps = sorted(list(set([int(10**( i/n_points * log10(hypprms["it"]))) for i in range(n_points+1)])))
            fil_log["its"] = deepcopy(log_cps)
        else:
            fil_log["its"] = [int(hypprms["it"] / hypprms["epochs"] * i) for i in range(int(hypprms["epochs"]) + 1)]

        n_checkpoints = int(hypprms["epochs"] + 1) if not log_eval else n_points+1
        fil_log["network_usage_in_MBtyes"] = [0.0 for _ in range(n_checkpoints)]
        epoch_duration_in_sec = [0.0 for _ in range(n_checkpoints)]

        cp_ptr = 0
        for i in range(1, num_jsons+1): # keep the order of reading jsons
            fn = "{}log-{}.json".format(node_dir, i)
            if not fn in json_results:
                raise ValueError("file not found: {}".format(fn))
            with open(fn) as json_f:
                logs = json.load(json_f)
            for log in logs:
                if "epoch" in log:
                    if log_eval:
                        if log["epoch"] * hypprms["batch_per_epoch"] > log_cps[0]:
                            log_cps.pop(0)
                            cp_ptr += 1
                    else:
                        cp_ptr = int(log["epoch"])
                
                if log["measurement"] == "timer":
                    if log["event"] in ["load_data", "forward_pass", "backward_pass", "sync_complete"]:
                        epoch_duration_in_sec[cp_ptr] += log["value"]
                    elif log["event"] == "opt_init":
                        epoch_duration_in_sec[0] += log["value"]
                if log["measurement"] != "runtime" and log["measurement"] != "stat":
                    continue

                if "time" in log:
                    if fil_log.get("start_time", None) is None:
                        fil_log["start_time"] = log["time"]
                    else:
                        fil_log["end_time"] = log["time"]

                if "type" in log and log["type"] in tags_to_log:
                    for metric in tags_to_log[log["type"]]:
                        fil_log[log["type"]][metric].append(log[metric])
            
                if "n_bits_to_transmit" in log:
                    fil_log["network_usage_in_MBtyes"][cp_ptr] += log["n_bits_to_transmit"]
        if "start_time" in fil_log and "end_time" in fil_log:
            fil_log["total_duration_in_second"] = get_time_diff(fil_log["start_time"], fil_log["end_time"])
        fil_log["training_durations"] = np.cumsum(epoch_duration_in_sec[:-1])
        all_logs.append(fil_log)
    return all_logs


def reduce_logs(logs, reducer, tags_to_log, world_size):
    nps = {k: {} for k in reducer.keys()}
    for k in tags_to_log.keys():
        for tag in tags_to_log[k]:
            agg = [log[k][tag] for log in logs]
            nps[k][tag] = np.array(agg)
            assert len(nps[k][tag]) == world_size, "invalid world size found in logs: {}, {}, {}, {}".format(len(nps[k][tag]), k, tag ,nps[k][tag].shape)
    for k in reducer.keys():
        for tag, op in zip(tags_to_log[k], reducer[k]):
            if op == max:
                nps[k][tag] = np.max(nps[k][tag], axis=0)
            elif op == min:
                nps[k][tag] = np.min(nps[k][tag], axis=0)
            elif op == "avg":
                nps[k][tag] = np.mean(nps[k][tag], axis=0)
            else:
                raise ValueError("unsupported op {}".format(op))
    nps["network_usage_in_MBtyes"] = np.cumsum(np.sum([log["network_usage_in_MBtyes"][:-1] for log in logs], axis=0))
    nps["its"] = logs[0]["its"]
    return nps
 

def hypprms_parsing(name, results_dir, world_size, model_size, **kwargs):
    batch_per_epoch = get_num_batches_train_per_device_per_epoch(results_dir)
    name = name.split("_warmup_epochs-0")[0]
    key_names_type = [
        ("l2", "weight_decay", float),
        ("lr", "learning_rate", float),
        ("it", "it", int),
        ("epochs", "epochs", float),
        ("batchsize", "batchsize", int),
        ("stepsize", "consensus_stepsize", float),
        ("k", "k", float),
        ("qsgd", "quan_bits", int),
        ("seed", "seed", int),
        ("beta", "momentum_beta", float),
        ("rounds", "gossip_rounds", int),
        ("eta", "gossip_eta", float),
        ("lrsch", "lrschedule", str),
        ("lrdecay", "lrdecay", float),
        ("betasch", "betaschedule", str),
        ("betadecay", "betadecay", float),
        ("gammadecay", "gammadecay", float),
        ("initb", "initbatchnum", int),
        ("rho", "rho", float),
        ("gamma", "gamma", float),
        ("alpha", "alpha", float),
        ("edgefrac", "edgefrac", float),
        ("T", "T", int),
        ("B", "B", int),
        ]
    to_catch_keys = [tup[0] for tup in key_names_type]
    hypprms = {"name": name, "batch_per_epoch": batch_per_epoch, "proj_name": kwargs["proj_name"]}
    entries = name.split("_")
    for en in entries:
        if "-" in en:
            spl = en.split("-")
            k, v = spl[0], "-".join(spl[1:])
            if k in to_catch_keys:
                hypprms[k] = v
    for k, new_k, t in key_names_type:
        if k in hypprms:
            tmp = hypprms[k]
            hypprms.pop(k)
            hypprms[new_k] = t(tmp)
    return hypprms


def extra_props(hypprms, iterations, world_size, model_size):
    comp_k = hypprms.get("k", 1)
    if "choco" in hypprms["name"]:
        if "qsgd" in hypprms:
            net_use = np.cumsum([uncompressed_net_unit["choco"] * world_size * (64 + model_size * (1 + hypprms["quan_bits"])) for _ in range(int(hypprms["epochs"]))])
        else: # top-k
            net_use = np.cumsum([uncompressed_net_unit["choco"] * comp_k * world_size * model_size *64  for _ in range(int(hypprms["epochs"]))])
        sample_use = np.array([world_size * hypprms["batchsize"] * it for it in iterations])
    elif "gt_hsgd" in hypprms["name"]:
        net_use = np.cumsum([uncompressed_net_unit["gt_hsgd"] * world_size * model_size *64 for _ in range(int(hypprms["epochs"]))])
        first_step_grad = init_full_batch_size[hypprms["proj_name"]]
        sample_use = first_step_grad + np.array([world_size * hypprms["batchsize"] * it for it in iterations])
    elif "docom" in hypprms["name"]:
        net_use = None
        # if "qsgd" in hypprms:
        #     net_use = np.cumsum([uncompressed_net_unit["docom"] * world_size * (64 + model_size * (1 + hypprms["quan_bits"])) for _ in range(int(hypprms["epochs"]))])
        # else: # top-k
        #     net_use = np.cumsum([uncompressed_net_unit["docom"] * comp_k * world_size * model_size *64 for _ in range(int(hypprms["epochs"]))])
        first_step_grad = init_full_batch_size[hypprms["proj_name"]]
        sample_use = first_step_grad + np.array([world_size * hypprms["batchsize"] * it for it in iterations])
    elif "beer" in hypprms["name"]:
        if "qsgd" in hypprms:
            net_use = np.cumsum([uncompressed_net_unit["beer"] * world_size * (64 + model_size * (1 + hypprms["quan_bits"])) for _ in range(int(hypprms["epochs"]))])
        else: # top-k
            net_use = np.cumsum([uncompressed_net_unit["beer"] * comp_k * world_size * model_size*64 for _ in range(int(hypprms["epochs"]))])
        first_step_grad = init_full_batch_size[hypprms["proj_name"]]
        sample_use = first_step_grad + np.array([world_size * hypprms["batchsize"] * it for it in iterations])
    elif "gnsd" in hypprms["name"]:
        net_use = np.cumsum([uncompressed_net_unit["gnsd"] * world_size * model_size*64 for _ in range(int(hypprms["epochs"]))])
        sample_use = np.array([world_size * hypprms["batchsize"] * it for it in iterations])
    elif "detag" in hypprms["name"]:
        net_use = np.cumsum([uncompressed_net_unit["detag"] * world_size * model_size*64 for _ in range(int(hypprms["epochs"]))])
        sample_use = np.array([world_size * hypprms["batchsize"] * it for it in iterations])
    elif "sgd" in hypprms["name"] and "complete" in hypprms["name"]:
        net_use = np.cumsum([uncompressed_net_unit["csgd"]*64 for _ in range(int(hypprms["epochs"]))])
        sample_use = np.array([world_size * hypprms["batchsize"] * it for it in iterations])
    elif "sgd" in hypprms["name"] and not "complete" in hypprms["name"]:
        net_use = np.cumsum([uncompressed_net_unit["sgd"] * world_size * model_size*64 for _ in range(int(hypprms["epochs"]))])
        sample_use = np.array([world_size * hypprms["batchsize"] * it for it in iterations])
    elif "spda" in hypprms["name"] or "di_cs" in hypprms["name"] or "diging" in hypprms["name"] or "fsppd" in hypprms["name"]:
        net_use = None
        sample_use = np.array([world_size * hypprms["batchsize"] * it for it in iterations])
    else:
        raise ValueError("Unimplemented extra_props for {}".format(hypprms["name"]))
    if not net_use is None:
        hypprms["net_use"] = net_use
    hypprms["sample_use"] = sample_use

# wandb
def exp_template(exp, seed):
    if "choco" in exp["name"]:
        if "k" in exp:
            comp_name = "top" if "top_k" in exp["name"] else "random"
            name = "CHOCO-{}lr-{}gamma-{}-{}k-{}batchsize-{}l2".format(exp["learning_rate"], exp["consensus_stepsize"], comp_name, exp["k"], exp["batchsize"], exp["weight_decay"])
            hypprm = {"gamma": exp["consensus_stepsize"],
                    "k": exp["k"]}
        else:
            name = "CHOCO-{}lr-{}gamma-{}bit-{}batchsize-{}l2".format(exp["learning_rate"], exp["consensus_stepsize"], exp["quan_bits"], exp["batchsize"], exp["weight_decay"])
            hypprm = {"gamma": exp["consensus_stepsize"],
                    "quan_bits": exp["quan_bits"]}
    elif "gt_hsgd" in exp["name"]:
        name = "GT-HSGD-{}lr-{}beta-{}batchsize-{}initbatch-{}l2".format(exp["learning_rate"], exp["momentum_beta"], exp["batchsize"], exp["initbatchnum"], exp["weight_decay"])
        hypprm = {"beta": exp["momentum_beta"]}
    elif "docom" in exp["name"]:
        if "k" in exp:
            name = "DoCoM-{}lr-{}gamma-{}beta-{}k-{}batchsize-{}l2".format(exp["learning_rate"], exp["consensus_stepsize"], exp["momentum_beta"], exp["k"], exp["batchsize"], exp["weight_decay"])
            hypprm = {"gamma": exp["consensus_stepsize"], 
                    "beta": exp["momentum_beta"], 
                    "k": exp["k"]}
        else:
            name = "DoCoM-{}lr-{}gamma-{}beta-{}bits-{}batchsize-{}l2".format(exp["learning_rate"], exp["consensus_stepsize"], exp["momentum_beta"], exp["quan_bits"], exp["batchsize"], exp["weight_decay"])
            hypprm = {"gamma": exp["consensus_stepsize"], 
                    "beta": exp["momentum_beta"], 
                    "quan_bits": exp["quan_bits"]}
    elif "beer" in exp["name"]:
        comp_name = "top" if "top_k" in exp["name"] else "random"
        if "k" in exp:
            name = "BEER-{}lr-{}gamma-{}-{}k-{}batchsize-{}l2".format(exp["learning_rate"], exp["consensus_stepsize"], comp_name, exp["k"], exp["batchsize"], exp["weight_decay"])
            hypprm = {"gamma": exp["consensus_stepsize"],
                        "k": exp["k"]}
        else:
            name = "BEER-{}lr-{}gamma-{}bits-{}batchsize-{}l2".format(exp["learning_rate"], exp["consensus_stepsize"], exp["quan_bits"], exp["batchsize"], exp["weight_decay"])
            hypprm = {"gamma": exp["consensus_stepsize"],
                        "quan_bits": exp["quan_bits"]}
    elif "gnsd" in exp["name"]:
        name = "GNSD-{}lr-{}batchsize".format(exp["learning_rate"], exp["batchsize"])
        hypprm = {}
    elif "diging" in exp["name"]:
        name = "DIGing-{}lr-{}edge".format(exp["learning_rate"], exp["edgefrac"])
        hypprm = {}
    elif "detag" in exp["name"]:
        name = "DeTAG-{}lr-{}R-{}g_eta-{}batchsize".format(exp["learning_rate"], exp["gossip_rounds"], exp["gossip_eta"], exp["batchsize"])
        hypprm = {"gossip_eta": exp["gossip_eta"],
                "gossip_rounds": exp["gossip_rounds"]}
    elif "sgd" in exp["name"]:
        if "complete" in exp["name"]:
            name = "CSGD-{}lr-{}batchsize".format(exp["learning_rate"], exp["batchsize"])
        else:
            name = "DSGD-{}lr-{}batchsize".format(exp["learning_rate"], exp["batchsize"])
        hypprm = {}
    elif "pprox" in exp["name"]:
        name = "PProxSPDA-{}rho-{}gamma-{}alpha-{}edge-{}k".format(exp["rho"], exp["gamma"], exp["alpha"], exp["edgefrac"], exp["k"])
        hypprm = {"rho": exp["rho"], "gamma": exp["gamma"], "alpha": exp["alpha"],
                    "k": exp["k"], "edge_frac": exp["edgefrac"]}
    elif "fully_spda" in exp["name"]:
        name = "FullySPDA-{}gamma-{}alpha-{}edge-{}k".format(exp["gamma"], exp["alpha"], exp["edgefrac"], exp["k"])
        hypprm = {"gamma": exp["gamma"], "alpha": exp["alpha"],
                    "k": exp["k"], "edge_frac": exp["edgefrac"]}
    elif "fsppd" in exp["name"]:
        name = "FSPPD-{}lr-{}gamma-{}batchsize-{}edge-{}k".format(exp["learning_rate"], exp["gamma"], exp["batchsize"], exp["edgefrac"], exp["k"])
        hypprm = {"gamma": exp["gamma"], 
                    "k": exp["k"], "edge_frac": exp["edgefrac"]}
    elif "di_cs" in exp["name"]:
        if "di_cs_svrg" in exp["name"]:
            opt_name = "Di-CS-SVRG"
        elif "di_cs_gd" in exp["name"]:
            opt_name = "Di-CS-GD"
        else:
            opt_name = "Di-CS"
        name = opt_name + "-{}alpha-{}gamma-{}T-{}B-{}edge-{}k".format(exp["alpha"], exp["gamma"], exp["T"], exp["B"], exp["edgefrac"], exp["k"])
        hypprm = {"gamma": exp["gamma"], "alpha": exp["alpha"], "T": exp["T"], "B": exp["B"],
                    "k": exp["k"], "edge_frac": exp["edgefrac"]}
    else:
        raise ValueError("Unimplemented template for {}".format(exp["name"]))
    name = "{}-{}seed".format(name, seed)
    if "SAM" in exp["name"]:
        name = "SAM-" + name
    
    # if "lrschedule" in exp:
    #     name = "{}-{}lrschedule".format(name, exp["lrschedule"])
    # if "lrdecay" in exp:
    #     name = "{}-{}lrdecay".format(name, exp["lrdecay"])
    # if "gammadecay" in exp:
    #     name = "{}-{}gammadecay".format(name, exp["gammadecay"])
    return name, hypprm
  

def upload_to_wandb(exp, proj_name, entries_count):
    wandb.login()
    exp_name, exp_hypprm = exp_template(exp, exp["seed"])
    hyperparameters = {
                "_name": exp["name"],
                "epochs": exp["epochs"],
                "batch_size": exp["batchsize"],
                "seed": exp["seed"],
                "learning_rate": exp["learning_rate"],
                **exp_hypprm,
                }
    metrics = [("worst_true_grad_train_global", "eval_local_model_on_full_training_data", "grad_norm"),
                ("worst_loss_train_global", "eval_local_model_on_full_training_data", "loss"),
                ("worst_acc_train_global", "eval_local_model_on_full_training_data", "top1"), 
                ("worst_loss_test_global", "eval_local_model_on_full_testing_data", "loss"), 
                ("worst_acc_test_global", "eval_local_model_on_full_testing_data", "top1"),
                ("consensus_gap", "averaged_model", "distance")]
    log_length = {name: entries_count[a+"_"+b] for name,a,b in metrics}
    max_log_length = max([log_length[n] for n in log_length])
    with wandb.init(project=proj_name, name=exp_name, config=hyperparameters):
        for i in range(max_log_length):
            wandb_log = {
                        "iteration": exp["its"][i],
                        # "worst_true_grad_train_global": exp["eval_local_model_on_full_training_data"]["grad_norm"][i],
                        # "worst_loss_train_global": exp["eval_local_model_on_full_training_data"]["loss"][i],
                        # "worst_acc_train_global": exp["eval_local_model_on_full_training_data"]["top1"][i],
                        # "worst_loss_test_global": exp["eval_local_model_on_full_testing_data"]["loss"][i],
                        # "worst_acc_test_global": exp["eval_local_model_on_full_testing_data"]["top1"][i],
                        # "consensus_gap": exp["averaged_model"]["distance"][i],
                        # "network_bits_transmitted": exp.get("net_use", exp["network_usage_in_MBtyes"] * 8 * 2**20)[i],
                        "network_MBtyes_transmitted": exp["network_usage_in_MBtyes"][i],
                        "sample_accessed":exp["sample_use"][i], 
                        "wall_clock_training_time": exp["wall_clock_training_time"][i],
                        "total_program_time": exp["total_program_time"][i]}
            for name, a, b in metrics:
                if i < log_length[name]:
                    wandb_log[name] = exp[a][b][i]
            if "gnsd" in exp["name"] or "detag" in exp["name"] or "docom" in exp["name"] or "gt_hsgd" in exp["name"] or "beer" in exp["name"]:
                wandb_log = {**wandb_log, "grad_tracker_consensus_gap": exp["averaged_gt"]["distance"][i]}
            wandb.log(wandb_log, step=i)

def get_gt_consensus(saved_model_dir, world_size=36):
    gt_cons = [[] for _ in range(world_size)]
    for ag in range(world_size):
        json_results = ls_cmd("ls -d {}/*.json".format(osp.join(saved_model_dir, str(ag))))
        num_jsons = len(json_results)
        for j in range(1, 1+num_jsons):
            with open(osp.join(saved_model_dir, str(ag), "log-{}.json".format(j))) as f:
                logs = json.load(f)
                for log in logs:
                    if "grad_tracker_consensus_dist" in log:
                        gt_cons[ag].append(log["grad_tracker_consensus_dist"])
    
    gt_cons = np.array(gt_cons)
    gt_cons = np.max(gt_cons, axis=0)
    return gt_cons


def world_size(exp_name):
    return int(exp_name.split("agents_")[-1].split("_")[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--inference_dir', type=str, required=True)
    parser.add_argument("--log_eval", type=str2bool, default=False)
    parser.add_argument('--model_size', type=int, default=60850)
    parser.add_argument("--proj_name", type=str, default="leaf-femnist")
    conf = parser.parse_args()

    print("parsing log of {} and upload to wandb".format(conf.exp_name))
    conf.world_size = world_size(conf.exp_name)

    
    results_dir = osp.join(conf.inference_dir, conf.exp_name)

    tags_to_log = {"eval_local_model_on_full_training_data": ["loss", "grad_norm", "top1"], 
                "eval_local_model_on_full_testing_data": ["loss", "top1"],
                "averaged_model": ["distance"],
                "averaged_gt": ["distance"]}
    reducer = {"eval_local_model_on_full_training_data": [max, max, min], 
            "eval_local_model_on_full_testing_data": [max, min],
            "averaged_model": ["avg"],
            "averaged_gt": ["avg"]}
    hypprms = hypprms_parsing(conf.exp_name, results_dir, conf.world_size, conf.model_size, proj_name=conf.proj_name)
    all_logs = filter_logs(results_dir, tags_to_log, hypprms, log_eval=conf.log_eval)
    reduced_logs = reduce_logs(all_logs, reducer, tags_to_log, conf.world_size)
    extra_props(hypprms, reduced_logs["its"], conf.world_size, conf.model_size)
    entries_count = {}
    for k in tags_to_log:
        for metric in tags_to_log[k]:
            print("{} - {} entries: {}".format(k, metric, len(reduced_logs[k][metric])))
            entries_count[k+"_"+metric] = len(reduced_logs[k][metric])

    total_program_time =  [all_logs[0]["total_duration_in_second"] / int(hypprms["epochs"]) * (i+1) for i in range(int(hypprms["epochs"]))]
    reduced_logs = {**reduced_logs, **hypprms, "wall_clock_training_time": all_logs[0]["training_durations"], "total_program_time": total_program_time}
    # reduced_logs
    print("epochs in log: ", len(reduced_logs["eval_local_model_on_full_training_data"]["loss"]))
    upload_to_wandb(reduced_logs, conf.proj_name, entries_count)
