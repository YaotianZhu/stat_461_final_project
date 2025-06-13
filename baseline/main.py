#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的GCN Baseline主程序
实现深度GCN + BatchNorm + 残差连接 + TF-IDF特征
"""

import argparse
import torch
import numpy as np
import os
import sys

from data_loader import DataLoader
from model import OptimizedGCN
from train import run_multiple_seeds, Trainer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='优化的GCN Baseline')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='Cora', 
                    choices=['Cora', 'ogbn-arxiv', 'PubMed', 'WikiCS'],
                    help='数据集名称')
    parser.add_argument('--data_root', type=str, default='./data',
                    help='数据存储根目录')
    parser.add_argument('--max_features', type=int, default=5000,
                    help='TF-IDF最大特征数')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=128,
                    help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=4,
                    help='GCN层数')
    parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout概率')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.005,
                    help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='权重衰减')
    parser.add_argument('--max_epochs', type=int, default=200,
                    help='最大训练轮数')
    parser.add_argument('--patience', type=int, default=50,
                    help='早停耐心值')
    
    # 实验参数
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                    help='随机种子列表')
    parser.add_argument('--device', type=str, default='auto',
                    help='训练设备 (cuda/cpu/auto)')
    parser.add_argument('--verbose', action='store_true', default=True,
                    help='是否打印详细信息')
    
    return parser.parse_args()

def setup_device(device):
    """设置训练设备"""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def print_dataset_info(data, num_classes, dataset_name):
    """打印数据集信息"""
    print(f"\n数据集信息: {dataset_name}")
    print(f"{'='*30}")
    print(f"节点数量: {data.x.shape[0]:,}")
    print(f"边数量: {data.edge_index.shape[1]:,}")
    print(f"特征维度: {data.x.shape[1]:,}")
    print(f"类别数量: {num_classes}")
    
    if hasattr(data, 'train_mask'):
        train_nodes = data.train_mask.sum().item()
        val_nodes = data.val_mask.sum().item()
        test_nodes = data.test_mask.sum().item()
        print(f"训练集: {train_nodes:,} 节点")
        print(f"验证集: {val_nodes:,} 节点")
        print(f"测试集: {test_nodes:,} 节点")

def print_model_info(model):
    """打印模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型信息:")
    print(f"{'='*30}")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型层数: {model.num_layers}")
    print(f"隐藏维度: {model.convs[0].out_channels}")

def get_model_config(dataset_name, input_dim):
    """根据数据集返回对应的模型配置"""
    name = dataset_name.lower()
    if name == 'cora':
        return {'hidden_dim': 128, 'num_layers': 4, 'dropout': 0.5}
    elif name == 'ogbn-arxiv':
        return {'hidden_dim': 256, 'num_layers': 3, 'dropout': 0.3}
    elif name == 'pubmed':
        return {'hidden_dim': 128, 'num_layers': 4, 'dropout': 0.5}
    elif name == 'wikics':
        return {'hidden_dim': 256, 'num_layers': 3, 'dropout': 0.3}
    else:
        return {'hidden_dim': 128, 'num_layers': 4, 'dropout': 0.5}

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    print("优化的GCN Baseline")
    print("="*50)
    print("配置参数:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # 设置设备
    device = setup_device(args.device)
    
    # 加载数据
    print(f"\n正在加载数据集: {args.dataset}...")
    data_loader = DataLoader(
        dataset_name=args.dataset,
        root=args.data_root,
        max_features=args.max_features
    )
    
    try:
        data, num_classes = data_loader.load_data()
        print("数据加载成功!")
    except Exception as e:
        print(f"数据加载失败: {e}")
        sys.exit(1)
    
    # 打印数据集信息
    print_dataset_info(data, num_classes, args.dataset)
    
    # 获取数据集特定的配置
    model_config = get_model_config(args.dataset, data.x.shape[1])
    
    if len(args.seeds) > 1:
        # 多种子运行
        results = run_multiple_seeds(
            OptimizedGCN, data, num_classes, 
            args.seeds, device, **model_config
        )
        
        # 保存多种子结果
        results_dir = f"results_{args.dataset.lower()}"
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, "final_results.txt")
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"优化的GCN Baseline结果 - {args.dataset}\n")
            f.write("="*50 + "\n")
            f.write(f"测试准确率: {results['mean_acc']:.4f} ± {results['std_acc']:.4f}\n")
            f.write(f"测试F1分数: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}\n")
        
        # 预期效果检查（使用平均值）
        test_acc = results['mean_acc']
        
    else:
        # 单种子运行
        model = OptimizedGCN(
            input_dim=data.x.shape[1],
            num_classes=num_classes,
            **model_config
        )
        print_model_info(model)
        
        trainer = Trainer(model, data, device, dataset_name=args.dataset)
        result = trainer.train()
        
        # 保存单次结果
        results_dir = f"results_{args.dataset.lower()}"
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, "final_results.txt")
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"优化的GCN Baseline结果 - {args.dataset}\n")
            f.write("="*50 + "\n")
            f.write(f"测试准确率: {result['test_acc']:.4f}\n")
            f.write(f"测试F1分数: {result['test_f1']:.4f}\n")
        
        # 预期效果检查
        test_acc = result['test_acc']

    print(f"\n结果已保存到: {results_file}")

    # 预期效果检查（统一处理）
    expected_acc = 0.80 if args.dataset.lower() == 'cora' else 0.72
    if test_acc >= expected_acc:
        print(f"✅ 达到预期效果! (目标: {expected_acc:.2f}, 实际: {test_acc:.4f})")
    else:
        print(f"⚠️  未达到预期效果 (目标: {expected_acc:.2f}, 实际: {test_acc:.4f})")

if __name__ == '__main__':
    main() 