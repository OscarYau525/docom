# DoCoM: Compressed Decentralized Optimization with Near-Optimal Sample Complexity
Extends from the PyTorch repo https://github.com/epfml/ChocoSGD.

## Installation
- Create Docker environment using `environments/docker/pytorch-mpi/Dockerfile`. Alternatively, you can setup the local environment according to the same file. Our experiments are reproducible from a single CPU server.

- Get the dataset FEMNIST as follows:
    ```
    git clone https://github.com/TalwalkarLab/leaf.git && cd leaf/data/femnist && ./preprocess.sh -s niid --sf 1.0 -k 0 -t sample
    ```

## Experiments
- Use `docom_experiments.ipynb`, `exps/feedforward-mnist.sh` and `exps/lenet-femnist.sh` to run the exact configurations shown in the paper.

- For PyTorch experiments, follow the example code below to use `parse_logs.py` that reads evaluation results and uploads data to [wandb](https://wandb.ai/) for visualization. `exp_name` can be found in `data/checkpoint/${DATASET}/${MODEL}/test`.

    ```
    python parse_logs.py \
        --exp_name 1670276052_l2-0.0001_lr-0.0001_it_epochs-20.0_batchsize-128_num_mpi_process_10_n_sub_process-1_topology-ring_seed-1_lrschedule-1000_lrdecay-10.0_optim-beer_v_stepsize-0.1_comm_info-compress_top_k-0.05_warmup_epochs-0 \
        --inference_dir data/checkpoint/femnist/lenet/test \
        --model_size 66126 \
        --proj_name docom_femnist
    ```
