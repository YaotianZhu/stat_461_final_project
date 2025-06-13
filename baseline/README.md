# 优化的GCN Baseline

这是一个优化的图卷积网络（GCN）baseline实现，通过增加网络深度、宽度、归一化和残差连接等技术，在Cora和OGBN-Arxiv数据集上显著提升准确率。

## 🚀 核心特性

- **深度网络**: 4层GCN（比常规2层更深）
- **宽隐藏层**: 128维隐藏特征
- **批量归一化**: 每层后添加BatchNorm
- **残差连接**: 缓解深层网络过平滑问题
- **TF-IDF特征**: 5000维文本向量化
- **多种子评估**: 确保结果可重复性

## 📁 项目结构

```
baseline/
├── main.py           # 主执行文件
├── model.py          # GCN模型定义
├── data_loader.py    # 数据加载和预处理
├── train.py          # 训练逻辑和多种子运行
├── requirements.txt  # 依赖包
└── README.md         # 说明文档
```

## 🛠️ 安装依赖

```bash
pip install -r requirements.txt
```

## 📊 使用方法

### 基本运行

```bash
# 在Cora数据集上运行
python main.py --dataset Cora

# 在OGBN-Arxiv数据集上运行
python main.py --dataset ogbn-arxiv
```

### 自定义参数

```bash
python main.py \
    --dataset Cora \
    --hidden_dim 256 \
    --num_layers 6 \
    --lr 0.01 \
    --seeds 42 43 44 45 46
```

### 主要参数说明

- `--dataset`: 数据集名称 (`Cora` 或 `ogbn-arxiv`)
- `--hidden_dim`: 隐藏层维度 (默认: 128)
- `--num_layers`: GCN层数 (默认: 4)
- `--dropout`: Dropout概率 (默认: 0.5)
- `--lr`: 学习率 (默认: 0.005)
- `--max_epochs`: 最大训练轮数 (默认: 200)
- `--patience`: 早停耐心值 (默认: 50)
- `--seeds`: 随机种子列表 (默认: [42, 43, 44])

## 🎯 预期结果

- **Cora**: 准确率 ≈ 0.80-0.83，Macro-F1 相近
- **OGBN-Arxiv**: 准确率 ≈ 0.72-0.75，Macro-F1 ≈ 0.70-0.73

## 🏗️ 技术细节

### 网络架构
- 4层标准GCN
- 每层后接BatchNorm + ReLU + Dropout
- 第2层开始添加残差连接
- 最后一层输出log_softmax

### 数据预处理
- Cora: 基于原始特征生成伪文本
- OGBN-Arxiv: 模拟title+abstract文本
- TF-IDF向量化到5000维

### 训练策略
- Adam优化器，权重衰减5e-4
- Cosine学习率调度
- 早停机制防止过拟合
- 多种子运行确保可重复性

## 🔧 故障排除

1. **CUDA内存不足**: 减少`hidden_dim`或`max_features`
2. **收敛慢**: 调整学习率或增加`patience`
3. **结果不稳定**: 增加种子数量或调整正则化参数

## 📈 结果分析

程序会自动生成结果文件：
- `results_cora/final_results.txt`: Cora数据集结果
- `results_ogbn-arxiv/final_results.txt`: OGBN-Arxiv数据集结果

每个文件包含：
- 平均准确率和标准差
- 平均F1分数和标准差
- 各种子的详细结果 