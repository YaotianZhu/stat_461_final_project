#!/bin/bash
#SBATCH --job-name=cora-7b-test           # 更改作业名称为测试
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:4                 # 与训练脚本保持一致，请求4张A100 GPU (如果测试需更少，请调整)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                         # 内存，可根据测试需求调整
#SBATCH --time=04:00:00                   # 调整测试所需时间 (例如1小时，按需修改)
#SBATCH --account=p32759
#SBATCH --output=llama_cora_test.out      # 更改输出日志文件名
#SBATCH --error=llama_cora_test.err 

module load anaconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate InstructGLM_env

export CUDA_VISIBLE_DEVICES=0,1,2,3  

name=cora-7b                              # 模型/数据集名称
output=snap/$name 

PYTHONPATH=$PYTHONPATH:./llama_cora_src \
/home/bzw9055/miniconda3/envs/InstructGLM_env/bin/python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=12323 \
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
        --losses 'classification' \
        --backbone './7B/7B' \
        --output $output \
        --epoch 2 \
        --weight_decay 0 \
        --max_text_length 512 \
        --gen_max_length 64 \
        --lr 0.00008 \
        --inference \