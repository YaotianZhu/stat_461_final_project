#!/bin/bash
#SBATCH --job-name=cora-7b-train
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:4                 
#SBATCH --cpus-per-task=8                 
#SBATCH --mem=64G                          
#SBATCH --time=10:00:00                   
#SBATCH --account=
#SBATCH --output=llama_cora_train.out
#SBATCH --error=llama_cora_train.err

# ------------------- 环境变量设置 -------------------
module load anaconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate InstructGLM_env

export CUDA_VISIBLE_DEVICES=0,1,2,3        # 分配2张GPU就写两个

# ------------------- 训练参数设置 -------------------
name=cora-7b
output=snap/$name

# ------------------- 启动训练 -------------------
# ------------------- 启动训练 -------------------
PYTHONPATH=$PYTHONPATH:./llama_cora_src \
/home/bzw9055/miniconda3/envs/InstructGLM_env/bin/python -m torch.distributed.launch \
    --nproc_per_node=4 \
    ./llama_cora_src/pretrain.py \
        --distributed --multiGPU \
        --seed 42 \
        --gradient_accumulation_steps 8 \
        --train Cora \
        --valid Cora \
        --batch_size 4 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --num_workers 8 \
        --clip_grad_norm 1.0 \
        --losses 'link,classification' \
        --backbone './7B/7B' \
        --output $output \
        --epoch 2 \
        --weight_decay 0 \
        --max_text_length 512 \
        --gen_max_length 64 \
        --lr 0.00008 \

