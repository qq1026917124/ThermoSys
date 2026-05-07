"""
热力学世界模型 (ThermoWorldModel)
预测：给定当前ThermoState + Agent Action，下一时刻的ThermoState
使用JEPA（Joint Embedding Predictive Architecture）架构
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Optional
from collections import deque


class ThermoWorldModel(nn.Module):
    """
    热力学动力学网络（模拟 dT/dt = f(T, A)）
    
    输入：当前状态 + 行动
    输出：下一状态预测 + 预测不确定性
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # 行动编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # 动力学网络
        layers = []
        input_dim = hidden_dim + hidden_dim // 2
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
        self.dynamics = nn.Sequential(*layers)
        
        # 预测头（残差预测 ΔState）
        self.delta_head = nn.Linear(hidden_dim, state_dim)
        
        # 不确定性头（异方差）
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, state_dim)
        )
        
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: [batch, state_dim]
            action: [batch, action_dim]
            
        Returns:
            next_state_pred: [batch, state_dim] 预测下一状态
            uncertainty: [batch, state_dim] 预测不确定性（方差）
        """
        s_emb = self.state_encoder(state)
        a_emb = self.action_encoder(action)
        
        combined = torch.cat([s_emb, a_emb], dim=-1)
        hidden = self.dynamics(combined)
        
        # 残差预测
        delta = self.delta_head(hidden)
        next_state = state + delta
        
        # 不确定性（确保为正）
        log_var = self.uncertainty_head(hidden)
        uncertainty = torch.exp(log_var)
        
        return next_state, uncertainty
    
    def compute_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state_true: torch.Tensor,
        reward_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        计算负对数似然损失（考虑异方差）
        
        Loss = Σ[(pred - true)^2 / (2*var) + 0.5*log(var)]
        """
        pred, var = self.forward(state, action)
        
        # 负对数似然
        nll = torch.mean((pred - next_state_true)**2 / (2 * var + 1e-8) + 
                        0.5 * torch.log(var + 1e-8))
        
        # MSE辅助损失
        mse = torch.mean((pred - next_state_true)**2)
        
        # 总损失
        total_loss = nll + reward_weight * mse
        
        return {
            'total': total_loss,
            'nll': nll,
            'mse': mse
        }
    
    def predict_with_uncertainty(
        self,
        state: np.ndarray,
        action: np.ndarray,
        num_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用MC Dropout进行不确定性估计
        
        Returns:
            (mean_pred, std_pred, epistemic_uncertainty)
        """
        self.train()  # 启用dropout
        
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred, _ = self.forward(state_t, action_t)
                predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        # 认知不确定性（预测方差）
        epistemic = np.var(predictions, axis=0)
        
        self.eval()
        return mean_pred, std_pred, epistemic


class WorldModelTrainer:
    """
    世界模型训练器
    支持在线学习和批量训练
    """
    
    def __init__(
        self,
        model: ThermoWorldModel,
        lr: float = 1e-4,
        batch_size: int = 256,
        buffer_size: int = 10000,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.buffer = deque(maxlen=buffer_size)
        
    def add_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray
    ):
        """添加经验到缓冲区"""
        self.buffer.append((state, action, next_state))
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """执行一步训练"""
        if len(self.buffer) < self.batch_size:
            return None
        
        # 采样
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32, device=self.device)
        
        # 前向传播和损失计算
        self.optimizer.zero_grad()
        losses = self.model.compute_loss(states, actions, next_states)
        losses['total'].backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def online_update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        n_steps: int = 5
    ) -> Dict[str, float]:
        """
        在线更新（用于每日收盘后快速适应）
        """
        self.add_experience(state, action, next_state)
        
        losses = []
        for _ in range(n_steps):
            loss = self.train_step()
            if loss is not None:
                losses.append(loss)
        
        if not losses:
            return {'total': 0, 'nll': 0, 'mse': 0}
        
        # 返回平均损失
        return {
            k: np.mean([l[k] for l in losses])
            for k in losses[0].keys()
        }
    
    def evaluate_prediction_accuracy(
        self,
        test_states: np.ndarray,
        test_actions: np.ndarray,
        test_next_states: np.ndarray
    ) -> Dict[str, float]:
        """
        评估预测精度
        """
        self.model.eval()
        
        with torch.no_grad():
            states_t = torch.tensor(test_states, dtype=torch.float32, device=self.device)
            actions_t = torch.tensor(test_actions, dtype=torch.float32, device=self.device)
            next_states_t = torch.tensor(test_next_states, dtype=torch.float32, device=self.device)
            
            pred, var = self.model(states_t, actions_t)
            
            mse = torch.mean((pred - next_states_t)**2).item()
            mae = torch.mean(torch.abs(pred - next_states_t)).item()
            
            # 校准误差（预测方差 vs 实际误差）
            actual_error = (pred - next_states_t)**2
            predicted_var = var
            calibration = torch.mean(torch.abs(actual_error - predicted_var)).item()
        
        self.model.train()
        
        return {
            'mse': mse,
            'mae': mae,
            'calibration': calibration,
            'rmse': np.sqrt(mse)
        }
