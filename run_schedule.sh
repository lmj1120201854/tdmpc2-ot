#!/bin/bash

# mw-assembly-sparse mw-basketball-sparse mw-bin-picking-sparse mw-box-close-sparse mw-coffee-pull-sparse 
# mw-coffee-push-sparse mw-hammer-sparse mw-peg-insert-side-sparse mw-push-wall-sparse mw-soccer-sparse mw-sweep-sparse mw-sweep-into-sparse


task=mw-assembly-sparse
exp_name=test-scale_scheduler

mkdir -p nohub_log/${exp_name}

seed=1
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py \
    task=${task} \
    steps=500000 \
    enable_reward_learning=false \
    demo_path=/home/burson/data/tdmpc_ot/expert_demos/${task}.pkl \
    n_demos=1 \
    exp_name=${exp_name} \
    ot_reward_shaping=true \
    render_device=9 \
    seed=${seed} > nohub_log/${exp_name}/${task}_${seed}.log &


seed=2
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py \
    task=${task} \
    steps=500000 \
    enable_reward_learning=false \
    demo_path=/home/burson/data/tdmpc_ot/expert_demos/${task}.pkl \
    n_demos=1 \
    exp_name=${exp_name} \
    ot_reward_shaping=true \
    render_device=9 \
    seed=${seed} > nohub_log/${exp_name}/${task}_${seed}.log &


seed=3
CUDA_VISIBLE_DEVICES=3 nohup python3 train.py \
    task=${task} \
    steps=500000 \
    enable_reward_learning=false \
    demo_path=/home/burson/data/tdmpc_ot/expert_demos/${task}.pkl \
    n_demos=1 \
    exp_name=${exp_name} \
    ot_reward_shaping=true \
    render_device=9 \
    seed=${seed} > nohub_log/${exp_name}/${task}_${seed}.log &


wait


task=mw-basketball-sparse
exp_name=test-scale_scheduler

mkdir -p nohub_log/${exp_name}

seed=1
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py \
    task=${task} \
    steps=500000 \
    enable_reward_learning=false \
    demo_path=/home/burson/data/tdmpc_ot/expert_demos/${task}.pkl \
    n_demos=1 \
    exp_name=${exp_name} \
    ot_reward_shaping=true \
    render_device=9 \
    seed=${seed} > nohub_log/${exp_name}/${task}_${seed}.log &


seed=2
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py \
    task=${task} \
    steps=500000 \
    enable_reward_learning=false \
    demo_path=/home/burson/data/tdmpc_ot/expert_demos/${task}.pkl \
    n_demos=1 \
    exp_name=${exp_name} \
    ot_reward_shaping=true \
    render_device=9 \
    seed=${seed} > nohub_log/${exp_name}/${task}_${seed}.log &


seed=3
CUDA_VISIBLE_DEVICES=3 nohup python3 train.py \
    task=${task} \
    steps=500000 \
    enable_reward_learning=false \
    demo_path=/home/burson/data/tdmpc_ot/expert_demos/${task}.pkl \
    n_demos=1 \
    exp_name=${exp_name} \
    ot_reward_shaping=true \
    render_device=9 \
    seed=${seed} > nohub_log/${exp_name}/${task}_${seed}.log &

wait

task=mw-bin-picking-sparse
exp_name=test-scale_scheduler

mkdir -p nohub_log/${exp_name}

seed=1
CUDA_VISIBLE_DEVICES=4 nohup python3 train.py \
    task=${task} \
    steps=500000 \
    enable_reward_learning=false \
    demo_path=/home/burson/data/tdmpc_ot/expert_demos/${task}.pkl \
    n_demos=1 \
    exp_name=${exp_name} \
    ot_reward_shaping=true \
    render_device=9 \
    seed=${seed} > nohub_log/${exp_name}/${task}_${seed}.log &


seed=2
CUDA_VISIBLE_DEVICES=5 nohup python3 train.py \
    task=${task} \
    steps=500000 \
    enable_reward_learning=false \
    demo_path=/home/burson/data/tdmpc_ot/expert_demos/${task}.pkl \
    n_demos=1 \
    exp_name=${exp_name} \
    ot_reward_shaping=true \
    render_device=9 \
    seed=${seed} > nohub_log/${exp_name}/${task}_${seed}.log &


seed=3
CUDA_VISIBLE_DEVICES=6 nohup python3 train.py \
    task=${task} \
    steps=500000 \
    enable_reward_learning=false \
    demo_path=/home/burson/data/tdmpc_ot/expert_demos/${task}.pkl \
    n_demos=1 \
    exp_name=${exp_name} \
    ot_reward_shaping=true \
    render_device=9 \
    seed=${seed} > nohub_log/${exp_name}/${task}_${seed}.log &

wait


task=mw-box-close-sparse
exp_name=test-scale_scheduler

mkdir -p nohub_log/${exp_name}

seed=1
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py \
    task=${task} \
    steps=500000 \
    enable_reward_learning=false \
    demo_path=/home/burson/data/tdmpc_ot/expert_demos/${task}.pkl \
    n_demos=1 \
    exp_name=${exp_name} \
    ot_reward_shaping=true \
    render_device=9 \
    seed=${seed} > nohub_log/${exp_name}/${task}_${seed}.log &


seed=2
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py \
    task=${task} \
    steps=500000 \
    enable_reward_learning=false \
    demo_path=/home/burson/data/tdmpc_ot/expert_demos/${task}.pkl \
    n_demos=1 \
    exp_name=${exp_name} \
    ot_reward_shaping=true \
    render_device=9 \
    seed=${seed} > nohub_log/${exp_name}/${task}_${seed}.log &


seed=3
CUDA_VISIBLE_DEVICES=3 nohup python3 train.py \
    task=${task} \
    steps=500000 \
    enable_reward_learning=false \
    demo_path=/home/burson/data/tdmpc_ot/expert_demos/${task}.pkl \
    n_demos=1 \
    exp_name=${exp_name} \
    ot_reward_shaping=true \
    render_device=9 \
    seed=${seed} > nohub_log/${exp_name}/${task}_${seed}.log &

wait

task=mw-coffee-pull-sparse
exp_name=test-scale_scheduler

mkdir -p nohub_log/${exp_name}

seed=1
CUDA_VISIBLE_DEVICES=4 nohup python3 train.py \
    task=${task} \
    steps=500000 \
    enable_reward_learning=false \
    demo_path=/home/burson/data/tdmpc_ot/expert_demos/${task}.pkl \
    n_demos=1 \
    exp_name=${exp_name} \
    ot_reward_shaping=true \
    render_device=9 \
    seed=${seed} > nohub_log/${exp_name}/${task}_${seed}.log &


seed=2
CUDA_VISIBLE_DEVICES=5 nohup python3 train.py \
    task=${task} \
    steps=500000 \
    enable_reward_learning=false \
    demo_path=/home/burson/data/tdmpc_ot/expert_demos/${task}.pkl \
    n_demos=1 \
    exp_name=${exp_name} \
    ot_reward_shaping=true \
    render_device=9 \
    seed=${seed} > nohub_log/${exp_name}/${task}_${seed}.log &


seed=3
CUDA_VISIBLE_DEVICES=6 nohup python3 train.py \
    task=${task} \
    steps=500000 \
    enable_reward_learning=false \
    demo_path=/home/burson/data/tdmpc_ot/expert_demos/${task}.pkl \
    n_demos=1 \
    exp_name=${exp_name} \
    ot_reward_shaping=true \
    render_device=9 \
    seed=${seed} > nohub_log/${exp_name}/${task}_${seed}.log &

wait

task=mw-soccer-sparse
exp_name=test-scale_scheduler

mkdir -p nohub_log/${exp_name}

seed=1
CUDA_VISIBLE_DEVICES=4 nohup python3 train.py \
    task=${task} \
    steps=500000 \
    enable_reward_learning=false \
    demo_path=/home/burson/data/tdmpc_ot/expert_demos/${task}.pkl \
    n_demos=1 \
    exp_name=${exp_name} \
    ot_reward_shaping=true \
    render_device=9 \
    seed=${seed} > nohub_log/${exp_name}/${task}_${seed}.log &


seed=2
CUDA_VISIBLE_DEVICES=5 nohup python3 train.py \
    task=${task} \
    steps=500000 \
    enable_reward_learning=false \
    demo_path=/home/burson/data/tdmpc_ot/expert_demos/${task}.pkl \
    n_demos=1 \
    exp_name=${exp_name} \
    ot_reward_shaping=true \
    render_device=9 \
    seed=${seed} > nohub_log/${exp_name}/${task}_${seed}.log &


seed=3
CUDA_VISIBLE_DEVICES=6 nohup python3 train.py \
    task=${task} \
    steps=500000 \
    enable_reward_learning=false \
    demo_path=/home/burson/data/tdmpc_ot/expert_demos/${task}.pkl \
    n_demos=1 \
    exp_name=${exp_name} \
    ot_reward_shaping=true \
    render_device=9 \
    seed=${seed} > nohub_log/${exp_name}/${task}_${seed}.log &