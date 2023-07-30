import torch

def gradient_norm(model, weight_decay=None):
    total_norm = 0
    for prm in model.parameters():
        total_norm += prm.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    # if not weight_decay is None:
        # total_norm += 2 * weight_decay * model_norm(model)
    return total_norm

def model_norm(model):
    total_norm = 0
    for prm in model.parameters():
        total_norm += prm.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def log_gt_diff(conf, scheduler, local_gt, avg_gt):
    local_gt = torch.FloatTensor(local_gt)
    diff = local_gt - avg_gt
    dist = diff.norm(2).item()
    conf.logger.log_metric(
        name="stat",
        values={
            "rank": conf.graph.rank,
            "epoch": scheduler.epoch_,
            "distance": dist,
            "type": "averaged_gt"
        },
        tags={"split": "train"},
        display=conf.graph.rank==0
    )

def log_comm_norm(conf, scheduler, comm_x_norm, comm_y_norm=None):
    v_dict = {
        "rank": conf.graph.rank,
        "epoch": scheduler.epoch_,
        "local_index": scheduler.local_index,
        "error_compensation_norm_x": comm_x_norm
    }
    if not comm_y_norm is None:
        v_dict = {**v_dict, "error_compensation_norm_y": comm_y_norm}
    conf.logger.log_metric(
        name="runtime",
        values=v_dict,
        tags={"split": "train"},
        display=conf.graph.rank==0
    )