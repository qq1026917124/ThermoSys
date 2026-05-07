"""
元学习控制器 (MetaController)
MAML-based元学习，动态适应市场体制变化
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, List
from collections import defaultdict


class MetaController(nn.Module):
    """
    元学习控制器
    
    功能：
    1. 根据近期热力学状态序列识别市场Regime
    2. 输出该Regime下的快速适应参数
    """
    
    def __init__(
        self,
        state_dim: int,
        n_regimes: int = 4,
        hidden_dim: int = 64,
        adaptation_lr: float = 1e-3
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_regimes = n_regimes
        self.adaptation_lr = adaptation_lr
        
        # Regime编码器（LSTM处理时序）
        self.regime_encoder = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Regime分类器
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_regimes)
        )
        
        # 每个Regime对应一组快速适应参数的外星参数
        # 这里简化为策略网络的bias适配
        self.fast_weights = nn.Parameter(
            torch.randn(n_regimes, hidden_dim, hidden_dim) * 0.01
        )
        
        # Regime名称映射
        self.regime_names = {
            0: 'bull_trend',
            1: 'bear_trend',
            2: 'oscillation',
            3: 'transition'
        }
        
    def forward(
        self,
        state_history: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state_history: [batch, seq_len, state_dim] 过去T个时刻的热力学状态
            
        Returns:
            regime_id: 当前市场体制
            regime_logits: 体制分类logits
            fast_weight: 该体制下的快速适应参数
        """
        _, (h_n, _) = self.regime_encoder(state_history)
        
        # 使用最后一层隐状态
        last_hidden = h_n[-1]  # [batch, hidden_dim]
        
        regime_logits = self.regime_classifier(last_hidden)
        regime_id = torch.argmax(regime_logits, dim=-1)
        
        # 获取对应fast weight（批量处理时取第一个）
        if regime_id.shape[0] == 1:
            fw = self.fast_weights[regime_id[0]]
        else:
            fw = self.fast_weights[regime_id[0]]  # 简化，取第一个
        
        return regime_id[0].item(), regime_logits, fw
    
    def identify_regime(self, state_history: np.ndarray) -> Dict[str, any]:
        """
        识别当前市场体制
        
        Args:
            state_history: [seq_len, state_dim] numpy数组
            
        Returns:
            包含regime_id, regime_name, confidence的字典
        """
        self.eval()
        with torch.no_grad():
            state_t = torch.tensor(
                state_history, dtype=torch.float32
            ).unsqueeze(0)  # [1, seq_len, state_dim]
            
            regime_id, logits, _ = self.forward(state_t)
            probs = torch.softmax(logits, dim=-1)[0]
            confidence = probs[regime_id].item()
        
        return {
            'regime_id': regime_id,
            'regime_name': self.regime_names.get(regime_id, 'unknown'),
            'confidence': confidence,
            'probabilities': probs.numpy()
        }
    
    def adapt_policy_parameters(
        self,
        policy_params: Dict[str, torch.Tensor],
        support_set: List[Tuple[np.ndarray, np.ndarray, float]],
        n_steps: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        MAML内循环：快速适应策略参数到特定Regime
        
        Args:
            policy_params: 当前策略参数
            support_set: 支持集 [(state, action, reward), ...]
            n_steps: 内循环步数
            
        Returns:
            适应后的参数
        """
        adapted_params = {k: v.clone() for k, v in policy_params.items()}
        
        for _ in range(n_steps):
            # 计算支持集上的损失
            total_loss = 0
            for state, action, reward in support_set:
                state_t = torch.tensor(state, dtype=torch.float32)
                action_t = torch.tensor(action, dtype=torch.float32)
                reward_t = torch.tensor(reward, dtype=torch.float32)
                
                # 简化的策略梯度损失
                # 实际应使用策略网络的前向传播
                predicted_reward = torch.sum(state_t * action_t)
                loss = (predicted_reward - reward_t) ** 2
                total_loss += loss
            
            # 内循环梯度下降
            if len(support_set) > 0:
                total_loss /= len(support_set)
                
                # 手动计算梯度并更新
                for param in adapted_params.values():
                    if param.requires_grad:
                        # 这里简化处理，实际应通过autograd
                        pass
        
        return adapted_params


class RegimeTracker:
    """
    市场体制追踪器
    记录体制历史，检测体制切换
    """
    
    def __init__(self, switch_threshold: int = 3):
        self.switch_threshold = switch_threshold
        self.history = []
        self.current_regime = None
        self.regime_duration = 0
        
    def update(self, regime_info: Dict[str, any]):
        """更新体制记录"""
        regime_id = regime_info['regime_id']
        
        if regime_id == self.current_regime:
            self.regime_duration += 1
        else:
            # 记录切换
            if self.current_regime is not None:
                self.history.append({
                    'regime': self.current_regime,
                    'duration': self.regime_duration,
                    'end_regime': regime_id
                })
            
            self.current_regime = regime_id
            self.regime_duration = 1
    
    def get_switch_frequency(self, window: int = 50) -> float:
        """计算近期体制切换频率"""
        if len(self.history) < 2:
            return 0.0
        
        recent = self.history[-window:]
        total_duration = sum(h['duration'] for h in recent)
        n_switches = len(recent)
        
        return n_switches / max(total_duration, 1)
    
    def get_regime_statistics(self) -> pd.DataFrame:
        """获取各体制的统计信息"""
        if not self.history:
            return pd.DataFrame()
        
        stats = defaultdict(lambda: {'count': 0, 'total_duration': 0, 'avg_duration': 0})
        
        for h in self.history:
            regime = h['regime']
            stats[regime]['count'] += 1
            stats[regime]['total_duration'] += h['duration']
        
        for regime in stats:
            stats[regime]['avg_duration'] = (
                stats[regime]['total_duration'] / stats[regime]['count']
            )
        
        df = pd.DataFrame(stats).T
        df.index.name = 'regime_id'
        return df
