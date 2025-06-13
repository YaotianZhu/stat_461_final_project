#!/bin/bash
#SBATCH --job-name=arxiv-7b-train
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:4                 # 请求 4 张 A100 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --account=p32759
#SBATCH --output=llama_arxiv_train.out
#SBATCH --error=llama_arxiv_train.err

# ------------------- 环境变量设置 -------------------
module load anaconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate InstructGLM_env

export CUDA_VISIBLE_DEVICES=0,1,2,3      # 指定4张GPU

# ------------------- 训练参数设置 -------------------
name=arxiv-7b
output=snap/$name

# ------------------- 启动训练 -------------------
PYTHONPATH=$PYTHONPATH:./llama_arxiv_src \
/home/bzw9055/miniconda3/envs/InstructGLM_env/bin/python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=12321 \
    llama_arxiv_src/pretrain.py \
        --distributed --multiGPU \
        --seed 42 \
        --gradient_accumulation_steps 8 \
        --train Arxiv \
        --valid Arxiv \
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
        --max_text_length 2048 \
        --gen_max_length 64 \
        --lr 0.00008

