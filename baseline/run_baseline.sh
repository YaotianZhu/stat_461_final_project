#!/bin/bash
#SBATCH --job-name=GCN_Baseline       # 任务名称
#SBATCH --output=output.log           # 输出日志文件
#SBATCH --error=error.log             # 错误日志文件
#SBATCH --partition=job               # 使用的分区
#SBATCH --gres=gpu:1                  # 请求 1 块 GPU
#SBATCH --time=48:00:00              
#SBATCH --cpus-per-task=4            
#SBATCH --mem=80G                    

# 加载 Conda 环境
source /home/hhz6461/anaconda3/etc/profile.d/conda.sh
conda activate project1               # 激活你的 Conda 环境

echo "优化的GCN Baseline实验"
echo "======================="

# 安装依赖（如果需要）
echo "检查依赖包..."
pip install -r requirements.txt > /dev/null 2>&1

echo "开始运行实验..."

# 运行Cora数据集
echo ""
echo "📊 运行Cora数据集实验..."
echo "------------------------"
python main.py --dataset Cora --seeds 42 43 44

# 运行OGBN-Arxiv数据集
echo ""
echo "📊 运行OGBN-Arxiv数据集实验..."
echo "------------------------------"
python main.py --dataset ogbn-arxiv --seeds 42 43 44

# 运行PubMed数据集
echo ""
echo "📊 运行PubMed数据集实验..."
echo "--------------------------"
python main.py --dataset PubMed --seeds 42 43 44

# 运行Wiki-CS数据集
echo ""
echo "📊 运行Wiki-CS数据集实验..."
echo "--------------------------"
python main.py --dataset WikiCS --seeds 42 43 44

echo ""
echo "✅ 所有实验完成!"
echo "结果文件："
echo "  - results_cora/final_results.txt"
echo "  - results_ogbn-arxiv/final_results.txt"
echo "  - results_pubmed/final_results.txt"
echo "  - results_wikics/final_results.txt"