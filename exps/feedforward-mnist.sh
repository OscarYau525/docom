# BEER top_k with random topology.
$HOME/conda/envs/pytorch-py3.7/bin/python run.py \
    --arch feedforward --optimizer beer_v \
    --experiment test \
    --data mnist --pin_memory True \
    --batch_size 16 --initial_batch_num 6000 --num_workers 0 \
    --num_iterations 1000 --partition_data sorted --reshuffle_per_epoch False --stop_criteria iteration --eval_freq 10 \
    --n_mpi_process 10 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cuda \
    --lr 0.01 --const_lr False --lr_schedule_scheme custom_multistep --lr_change_epochs 1000 --lr_decay 10 \
    --weight_decay 1e-4 \
    --comm_op compress_top_k --compress_ratio 0.95 --is_biased True --consensus_stepsize 0.2 \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --evaluate_avg True --evaluate_consensus True \
    --python_path $HOME/conda/envs/pytorch-py3.7/bin/python --mpi_path $HOME/.openmpi/ --manual_seed 1 --train_fast False

# DoCoM with ring topology.
$HOME/conda/envs/pytorch-py3.7/bin/python run.py \
    --arch feedforward --optimizer docom_v \
    --experiment test \
    --data mnist --pin_memory True \
    --batch_size 16 --initial_batch_num 6000 --num_workers 0 \
    --num_iterations 1000 --partition_data sorted --reshuffle_per_epoch False --stop_criteria iteration --eval_freq 10 \
    --n_mpi_process 10 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cuda \
    --lr 0.01 --const_lr False --lr_schedule_scheme custom_multistep --lr_change_epochs 1000 --lr_decay 10 \
    --weight_decay 1e-4 --momentum_beta 0.01 \
    --comm_op compress_top_k --compress_ratio 0.95 --is_biased True --consensus_stepsize 0.2 \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --evaluate_avg True --evaluate_consensus True \
    --python_path $HOME/conda/envs/pytorch-py3.7/bin/python --mpi_path $HOME/.openmpi/ --manual_seed 1 --train_fast False

# parallel_choco with compress_top_k for ring topology
$HOME/conda/envs/pytorch-py3.7/bin/python run.py \
    --arch feedforward --optimizer parallel_choco_v \
    --experiment test \
    --data mnist --pin_memory False \
    --batch_size 256 --num_workers 0 \
    --num_iterations 3000 --partition_data sorted --reshuffle_per_epoch False --stop_criteria iteration --eval_freq 10 \
    --n_mpi_process 10 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cuda \
    --lr 0.01 --const_lr False --lr_schedule_scheme custom_multistep --lr_change_epochs 1000 --lr_decay 10 \
    --weight_decay 1e-4 \
    --comm_op compress_top_k --compress_ratio 0.9 --is_biased True --consensus_stepsize 0.3 \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --evaluate_avg True --evaluate_consensus True \
    --python_path $HOME/conda/envs/pytorch-py3.7/bin/python --mpi_path $HOME/.openmpi/ --manual_seed 1 --train_fast False
