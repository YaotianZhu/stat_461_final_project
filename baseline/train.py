import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
from tqdm import tqdm

class Trainer:
    def __init__(self, model, data, device, dataset_name='Cora', **kwargs):
        # 根据数据集选择不同的训练参数
        if dataset_name.lower() == 'cora':
            lr = kwargs.get('lr', 0.005)
            weight_decay = kwargs.get('weight_decay', 5e-4)
            patience = kwargs.get('patience', 50)
            max_epochs = 200
        elif dataset_name.lower() == 'ogbn-arxiv':
            lr = kwargs.get('lr', 0.01)        # 更高学习率
            weight_decay = kwargs.get('weight_decay', 1e-3)  # 更强正则化
            patience = kwargs.get('patience', 100)  # 更大耐心
            max_epochs = 500  # 更多训练轮数
        
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.patience = patience
        self.max_epochs = max_epochs
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # 学习率调度器 - 为OGBN-Arxiv使用更温和的调度
        if dataset_name.lower() == 'ogbn-arxiv':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=max_epochs, 
                eta_min=1e-5
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=200, 
                eta_min=1e-6
            )
        
        # 早停相关
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.best_model_state = None
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        out = self.model(self.data.x, self.data.edge_index)
        
        # 计算损失
        loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self):
        """验证"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            
            # 验证集损失
            val_loss = F.nll_loss(out[self.data.val_mask], self.data.y[self.data.val_mask])
            
            # 验证集准确率
            pred = out[self.data.val_mask].max(1)[1]
            val_acc = accuracy_score(
                self.data.y[self.data.val_mask].cpu().numpy(),
                pred.cpu().numpy()
            )
            
        return val_loss.item(), val_acc
    
    def test(self):
        """测试"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            
            # 测试集预测
            pred = out[self.data.test_mask].max(1)[1]
            y_true = self.data.y[self.data.test_mask].cpu().numpy()
            y_pred = pred.cpu().numpy()
            
            # 计算指标
            test_acc = accuracy_score(y_true, y_pred)
            test_f1 = f1_score(y_true, y_pred, average='macro')
            
        return test_acc, test_f1
    
    def train(self, verbose=True):
        """
        完整训练过程
        Args:
            verbose: 是否打印训练过程
        """
        if verbose:
            print("开始训练...")
            
        train_losses = []
        val_losses = []
        val_accs = []
        
        max_epochs = self.max_epochs  # 使用__init__中设置的值
        
        for epoch in range(max_epochs):
            # 训练
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # 验证
            val_loss, val_acc = self.validate()
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # 学习率调度
            self.scheduler.step()
            
            # 早停检查
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # 保存最佳模型
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
            
            # 打印进度
            if verbose and (epoch + 1) % 20 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"LR: {current_lr:.6f}")
            
            # 早停
            if self.patience_counter >= self.patience:
                if verbose:
                    print(f"早停触发，停止训练于第 {epoch+1} 轮")
                break
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # 最终测试
        test_acc, test_f1 = self.test()
        
        if verbose:
            print(f"训练完成!")
            print(f"最佳验证准确率: {self.best_val_acc:.4f}")
            print(f"测试准确率: {test_acc:.4f}")
            print(f"测试F1分数: {test_f1:.4f}")
        
        return {
            'test_acc': test_acc,
            'test_f1': test_f1,
            'best_val_acc': self.best_val_acc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accs': val_accs
        }

def run_multiple_seeds(model_class, data, num_classes, seeds=[42, 43, 44], device='cuda', **kwargs):
    """
    多种子运行
    Args:
        model_class: 模型类
        data: 数据
        num_classes: 类别数
        seeds: 随机种子列表
        device: 训练设备
        **kwargs: 其他参数
    """
    results = []
    
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"运行种子: {seed}")
        print(f"{'='*50}")
        
        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # 创建模型
        model = model_class(
            input_dim=data.x.shape[1],
            num_classes=num_classes,
            **kwargs
        )
        
        # 训练
        trainer = Trainer(model, data, device)
        result = trainer.train()
        results.append(result)
    
    # 统计结果
    test_accs = [r['test_acc'] for r in results]
    test_f1s = [r['test_f1'] for r in results]
    
    print(f"\n{'='*50}")
    print("最终结果统计:")
    print(f"{'='*50}")
    print(f"测试准确率: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    print(f"测试F1分数: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print(f"各种子详细结果:")
    for i, seed in enumerate(seeds):
        print(f"  种子 {seed}: Acc={test_accs[i]:.4f}, F1={test_f1s[i]:.4f}")
    
    return {
        'mean_acc': np.mean(test_accs),
        'std_acc': np.std(test_accs),
        'mean_f1': np.mean(test_f1s),
        'std_f1': np.std(test_f1s),
        'all_results': results
    } 