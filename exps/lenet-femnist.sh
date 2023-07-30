$HOME/conda/envs/pytorch-py3.7/bin/python run.py \
    --arch lenet --optimizer docom_v \
    --experiment test \
    --data femnist --pin_memory True \
    --batch_size 16 --initial_batch_num 10000 --num_workers 0 \
    --num_iterations 120000 --reshuffle_per_epoch True --stop_criteria iteration --eval_freq 200 \
    --n_mpi_process 36 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cuda \
    --lr 0.01 --const_lr False --lr_schedule_scheme custom_multistep --lr_change_epochs 20,200,400 --lr_decay 10 \
    --beta_change_epochs 50,200,400 --beta_decay 10 \
    --weight_decay 1e-4 --momentum_beta 0.3 \
    --comm_op compress_top_k --compress_ratio 0.95 --is_biased True --consensus_stepsize 0.2 \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/conda/envs/pytorch-py3.7/bin/python --mpi_path $HOME/.openmpi/ --manual_seed 1 --train_fast False --sam False --eval_consensus_only False --log_eval True

$HOME/conda/envs/pytorch-py3.7/bin/python run.py \
    --arch lenet --optimizer parallel_choco_v \
    --experiment test \
    --data femnist --pin_memory False \
    --batch_size 32 --num_workers 0 \
    --num_iterations 120000 --reshuffle_per_epoch True --stop_criteria iteration --eval_freq 200 \
    --n_mpi_process 36 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cuda \
    --lr 0.01 --const_lr False --lr_schedule_scheme custom_multistep --lr_change_epochs 20,200,400 --lr_decay 10 \
    --weight_decay 1e-4 \
    --comm_op compress_top_k --compress_ratio 0.9 --consensus_stepsize 0.25 \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/conda/envs/pytorch-py3.7/bin/python --mpi_path $HOME/.openmpi/ --manual_seed 1 --train_fast False --log_eval True

$HOME/conda/envs/pytorch-py3.7/bin/python run.py \
    --arch lenet --optimizer beer_v \
    --experiment test \
    --data femnist --pin_memory True \
    --batch_size 32 --initial_batch_num 10000 --num_workers 0 \
    --num_iterations 120000 --reshuffle_per_epoch True --stop_criteria iteration --eval_freq 200 \
    --n_mpi_process 36 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cuda \
    --lr 0.01 --const_lr False --lr_schedule_scheme custom_multistep --lr_change_epochs 20,200,400 --lr_decay 10 \
    --weight_decay 1e-4 \
    --comm_op compress_top_k --compress_ratio 0.95 --is_biased True --consensus_stepsize 0.2 \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/conda/envs/pytorch-py3.7/bin/python --mpi_path $HOME/.openmpi/ --manual_seed 1 --train_fast True --log_eval True

$HOME/conda/envs/pytorch-py3.7/bin/python run.py \
    --arch lenet --optimizer gnsd \
    --experiment test \
    --data femnist --pin_memory True \
    --batch_size 32 --num_workers 0 \
    --num_iterations 60000 --reshuffle_per_epoch True --stop_criteria iteration --eval_freq 200 \
    --n_mpi_process 36 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cuda \
    --lr 0.02 --const_lr False --lr_schedule_scheme custom_multistep --lr_change_epochs 20,100,175 --lr_decay 10 \
    --weight_decay 1e-4 \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/conda/envs/pytorch-py3.7/bin/python --mpi_path $HOME/.openmpi/ --manual_seed 1 --train_fast False --sam False --eval_consensus_only False --log_eval True

$HOME/conda/envs/pytorch-py3.7/bin/python run.py \
    --arch lenet --optimizer gt_hsgd \
    --experiment test \
    --data femnist --pin_memory True \
    --batch_size 16 --initial_batch_num 10000 --num_workers 0 \
    --num_iterations 60000 --reshuffle_per_epoch True --stop_criteria iteration --eval_freq 200 \
    --n_mpi_process 36 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cuda \
    --lr 0.02 --const_lr False --lr_schedule_scheme custom_multistep --lr_change_epochs 20,100,175 --lr_decay 10 \
    --beta_change_epochs 50,100,175 --beta_decay 10 \
    --weight_decay 1e-4 --momentum_beta 0.3 \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/conda/envs/pytorch-py3.7/bin/python --mpi_path $HOME/.openmpi/ --manual_seed 1 --train_fast False --log_eval True
