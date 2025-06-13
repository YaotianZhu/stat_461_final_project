#!/bin/bash
#SBATCH --job-name=graphgpt-eval
#SBATCH --partition=gengpu                 # 请确认分区名称是否正确
#SBATCH --gres=gpu:a100:4              # 根据你的命令，请求4张GPU
#SBATCH --cpus-per-task=8              # 你可以根据需要调整CPU核心数
#SBATCH --mem=64G                      # 你可以根据需要调整内存
#SBATCH --time=06:00:00                # 预计运行时间，请根据实际情况调整
#SBATCH --account=p32759               # 请替换成你的账户名
#SBATCH --output=graphgpt_eval_cot.out  # %j 会被替换为作业ID
#SBATCH --error=graphgpt_eval_cot.err   # %j 会被替换为作业ID

# ------------------- 环境变量设置 -------------------
# 确保你的 anaconda 或 miniconda 已正确配置
# 如果你的 conda 环境不是全局的，可能需要指定完整路径
module load anaconda3  # 或者根据你的集群环境加载正确的 anaconda/miniconda模块
source ~/miniconda3/etc/profile.d/conda.sh # 替换成你的 conda 初始化路径
module load cuda/11.4.0-gcc # Attempt to load a specific CUDA toolkit version found by module spider

# Deactivate and reactivate conda env to ensure it picks up CUDA paths
conda deactivate
conda activate graphgpt_py310 # 替换成你的 conda 环境名称

# Slurm 会自动处理 GPU 的分配，通常不需要手动设置 CUDA_VISIBLE_DEVICES
# 如果你的脚本或程序内部需要，可以保留，但通常SBATCH --gres已足够
export CUDA_VISIBLE_DEVICES=0,1,2,3 

# ------------------- 参数设置 -------------------
output_model=./GraphGPT-7B-mix-all
datapath=./GraphGPT-eval-instruction/arxiv_test_instruct_cot.json
graph_data_path=./All_pyg_graph_data/graph_data_all.pt
res_path=./output_stage_2_arxiv_nc
start_id=0
end_id=20000
num_gpus=4 # 这个参数在 sbatch 的 --gres 中已经指定，但你的脚本可能仍需要它

# 运行脚本
# 注意：你的原始命令是直接调用 python3.8，这里我们使用 conda 环境中的 python
# 如果你的脚本确实需要特定版本的 python3.8 且与 conda 环境中的不同，请确保路径正确
# 你的原始命令未使用 torch.distributed.launch，所以这里也直接运行
PYTHONPATH=$PYTHONPATH:. \
/home/bzw9055/miniconda3/envs/graphgpt_py310/bin/python ./graphgpt/eval/run_graphgpt.py \
    --model-name ${output_model} \
    --prompting_file ${datapath} \
    --graph_data_path ${graph_data_path} \
    --output_res_path ${res_path} \
    --start_id ${start_id} \
    --end_id ${end_id} \
    --num_gpus ${num_gpus}
