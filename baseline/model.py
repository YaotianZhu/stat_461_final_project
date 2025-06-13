import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class OptimizedGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=7, num_layers=4, dropout=0.5):
        """
        优化的GCN模型
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_classes: 类别数
            num_layers: 网络层数
            dropout: Dropout概率
        """
        super(OptimizedGCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 第一层：input_dim -> hidden_dim
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 中间层：hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 最后一层：hidden_dim -> num_classes
        self.convs.append(GCNConv(hidden_dim, num_classes))
        
    def forward(self, x, edge_index):
        """
        前向传播
        Args:
            x: 节点特征矩阵 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
        """
        # 存储中间层输出用于残差连接
        layer_outputs = []
        
        # 第一层
        x = self.convs[0](x, edge_index)
        x = self.batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_outputs.append(x)
        
        # 中间层（带残差连接）
        for i in range(1, self.num_layers - 1):
            identity = x  # 保存残差连接的输入
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            
            # 残差连接：当前输出 + 上一层输出
            if i > 0:  # 从第二层开始添加残差连接
                x = x + identity
            
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)
        
        # 最后一层（无BatchNorm和ReLU）
        x = self.convs[-1](x, edge_index)
        
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, x, edge_index):
        """
        获取节点嵌入（去掉最后的分类层）
        """
        # 存储中间层输出用于残差连接
        layer_outputs = []
        
        # 第一层
        x = self.convs[0](x, edge_index)
        x = self.batch_norms[0](x)
        x = F.relu(x)
        layer_outputs.append(x)
        
        # 中间层（带残差连接）
        for i in range(1, self.num_layers - 1):
            identity = x
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            
            # 残差连接
            if i > 0:
                x = x + identity
            
            layer_outputs.append(x)
        
        return x  # 返回最后一个隐藏层的输出 