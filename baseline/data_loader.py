import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from torch_geometric.datasets import WikiCS

class DataLoader:
    def __init__(self, dataset_name='Cora', root='./data', max_features=5000):
        """
        数据加载器
        Args:
            dataset_name: 数据集名称 ('Cora' 或 'ogbn-arxiv')
            root: 数据存储根目录
            max_features: TF-IDF最大特征数
        """
        self.dataset_name = dataset_name
        self.root = root
        self.max_features = max_features
        
    def load_cora(self):
        """加载Cora数据集"""
        dataset = Planetoid(root=self.root, name='Cora')
        data = dataset[0]
        
        # 为Cora生成伪文本特征（因为原始特征是词袋模型）
        num_nodes = data.x.shape[0]
        pseudo_texts = []
        
        # 基于原始特征生成伪关键词
        for i in range(num_nodes):
            features = data.x[i].numpy()
            # 找到非零特征的索引作为"关键词ID"
            nonzero_indices = np.where(features > 0)[0]
            # 取前30个作为关键词，生成伪文本
            keywords = [f"keyword_{idx}" for idx in nonzero_indices[:30]]
            pseudo_text = " ".join(keywords)
            pseudo_texts.append(pseudo_text)
        
        # TF-IDF向量化
        vectorizer = TfidfVectorizer(max_features=self.max_features)
        tfidf_features = vectorizer.fit_transform(pseudo_texts)
        
        # 转换为dense tensor
        data.x = torch.FloatTensor(tfidf_features.toarray())
        
        return data, dataset.num_classes
    
    def load_ogbn_arxiv(self):
        """加载OGBN-Arxiv数据集"""
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=self.root)
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        
        # 修复标签维度问题
        if data.y.dim() > 1:
            data.y = data.y.squeeze()
        
        # 直接使用原始特征，不进行文本处理
        # OGBN-Arxiv已经有高质量的128维节点特征
        print(f"使用原始特征，维度: {data.x.shape}")
        
        # 添加分割索引
        data.train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        
        data.train_mask[split_idx['train']] = True
        data.val_mask[split_idx['valid']] = True
        data.test_mask[split_idx['test']] = True
        
        return data, dataset.num_classes
    
    def load_pubmed(self):
        """加载PubMed数据集"""
        dataset = Planetoid(root=self.root, name='PubMed')
        data = dataset[0]
        # PubMed原始特征是词袋，可以直接TF-IDF处理
        num_nodes = data.x.shape[0]
        pseudo_texts = []
        for i in range(num_nodes):
            features = data.x[i].numpy()
            nonzero_indices = np.where(features > 0)[0]
            keywords = [f"keyword_{idx}" for idx in nonzero_indices[:30]]
            pseudo_text = " ".join(keywords)
            pseudo_texts.append(pseudo_text)
        vectorizer = TfidfVectorizer(max_features=self.max_features)
        tfidf_features = vectorizer.fit_transform(pseudo_texts)
        data.x = torch.FloatTensor(tfidf_features.toarray())
        return data, dataset.num_classes

    def load_wikics(self):
        """加载Wiki-CS数据集"""
        dataset = WikiCS(root=self.root)
        data = dataset[0]
        print(f"train_mask shape: {data.train_mask.shape}")
        print(f"val_mask shape: {data.val_mask.shape}")
        print(f"test_mask shape: {data.test_mask.shape}")
        print(f"train_mask dim: {data.train_mask.dim()}")
        # 自动判断mask维度
        if data.train_mask.dim() == 2:
            print("检测到二维mask，自动选取第0个split。")
            data.train_mask = data.train_mask[:, 0]
            data.val_mask = data.val_mask[:, 0]
            # 关键修复：即使test_mask是一维，也强制用[:, 0]，防止PyG版本不一致
            if data.test_mask.dim() == 2:
                data.test_mask = data.test_mask[:, 0]
        else:
            print("检测到一维mask，直接使用。")
        print(f"最终train_mask shape: {data.train_mask.shape}")
        print(f"最终val_mask shape: {data.val_mask.shape}")
        print(f"最终test_mask shape: {data.test_mask.shape}")
        return data, dataset.num_classes

    def load_data(self):
        """统一的数据加载接口"""
        if self.dataset_name.lower() == 'cora':
            return self.load_cora()
        elif self.dataset_name.lower() == 'ogbn-arxiv':
            return self.load_ogbn_arxiv()
        elif self.dataset_name.lower() == 'pubmed':
            return self.load_pubmed()
        elif self.dataset_name.lower() == 'wikics':
            return self.load_wikics()
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}") 