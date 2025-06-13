#!/bin/bash
#SBATCH --job-name=graphgpt-stage1
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:4                  # 使用 4 张 A100 GPU
#SBATCH --cpus-per-task=8                  
#SBATCH --mem=64G                         
#SBATCH --time=12:00:00                    
#SBATCH --account=
#SBATCH --output=graphgpt_stage1_train.out
#SBATCH --error=graphgpt_stage1_train.err

# ------------------- 环境变量设置 -------------------
module load anaconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate GraphGPT_env

export CUDA_VISIBLE_DEVICES=0,1,2,3        # 使用4张GPU

# ------------------- 训练参数设置 -------------------
model_path=
instruct_ds=
graph_data_path=
pretra_gnn=
output_model=

# ------------------- 启动训练 -------------------
wandb offline
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
    graphgpt/train/train_mem.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --graph_content ./arxiv_ti_ab.json \
    --graph_data_path ${graph_data_path} \
    --graph_tower ${pretra_gnn} \
    --tune_graph_mlp_adapter True \
    --graph_select_layer -2 \
    --use_graph_start_end \
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
