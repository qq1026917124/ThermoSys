"""
热力学状态统一接口
定义市场热力学状态向量，作为Agent的统一观测空间
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


@dataclass
class ThermoState:
    """
    市场热力学状态向量
    
    Attributes:
        ipv: 信息传播速度 (Information Propagation Velocity)
        clarity: 传播路径清晰度
        coherence: Kuramoto序参量 r(t)
        sector_temperatures: 各板块信息密度 ρ_i
        group_phases: 各投资者群体相位角
        entropy: 观点香农熵 S(t)
        entropy_change: 熵减率 dS/dt
        rsi: 散户情绪指数 Retail Sentiment Index
        mts: 市场热力态指数 Market Thermodynamic State
        timestamp: 时间戳
    """
    ipv: float = 0.0
    clarity: float = 0.0
    coherence: float = 0.0
    sector_temperatures: Dict[str, float] = field(default_factory=dict)
    group_phases: Dict[str, float] = field(default_factory=dict)
    entropy: float = 1.0
    entropy_change: float = 0.0
    rsi: float = 0.0
    mts: float = 0.0
    timestamp: Optional[pd.Timestamp] = None
    
    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """转换为Agent可消费的统一张量"""
        base = torch.tensor([
            self.ipv, self.clarity, self.coherence,
            self.entropy, self.entropy_change, self.rsi, self.mts
        ], dtype=torch.float32, device=device)
        
        sector_temps = torch.tensor(
            list(self.sector_temperatures.values()),
            dtype=torch.float32, device=device
        ) if self.sector_temperatures else torch.tensor([], dtype=torch.float32, device=device)
        
        group_phases = torch.tensor(
            list(self.group_phases.values()),
            dtype=torch.float32, device=device
        ) if self.group_phases else torch.tensor([], dtype=torch.float32, device=device)
        
        return torch.cat([base, sector_temps, group_phases])
    
    @property
    def to_dict(self) -> Dict[str, any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp,
            'ipv': self.ipv,
            'clarity': self.clarity,
            'coherence': self.coherence,
            'entropy': self.entropy,
            'entropy_change': self.entropy_change,
            'rsi': self.rsi,
            'mts': self.mts,
            **{f'sector_{k}': v for k, v in self.sector_temperatures.items()},
            **{f'group_{k}': v for k, v in self.group_phases.items()},
        }
    
    def is_extreme_fear(self, threshold: float = -2.0) -> bool:
        """是否处于极端恐惧区"""
        return self.rsi < threshold
    
    def is_extreme_greed(self, threshold: float = 2.0) -> bool:
        """是否处于极端贪婪区"""
        return self.rsi > threshold
    
    def is_coherent(self, threshold: float = 0.6) -> bool:
        """市场是否形成合力"""
        return self.coherence > threshold
    
    def is_path_clear(self, threshold: float = 0.7) -> bool:
        """传播路径是否清晰"""
        return self.clarity > threshold


class ThermoStateEncoder(nn.Module):
    """
    热力学状态编码器：将原始异构数据编码为统一ThermoState
    这是感知层(Perception)的核心
    """
    
    def __init__(
        self,
        n_sectors: int = 10,
        n_groups: int = 4,
        text_dim: int = 64,
        flow_dim: int = 64,
        search_dim: int = 32,
        hidden_dim: int = 128,
        device: str = 'cpu'
    ):
        super().__init__()
        self.n_sectors = n_sectors
        self.n_groups = n_groups
        self.device = device
        
        # 多源数据编码分支
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=text_dim, nhead=4, dim_feedforward=256, batch_first=True
            ),
            num_layers=2
        )
        
        self.flow_encoder = nn.LSTM(
            input_size=10, hidden_size=flow_dim, num_layers=2,
            batch_first=True, dropout=0.1
        )
        
        self.search_encoder = nn.Sequential(
            nn.Linear(5, search_dim),
            nn.LayerNorm(search_dim),
            nn.GELU()
        )
        
        # 热力学特征融合
        input_dim = text_dim + flow_dim + search_dim
        output_dim = 7 + n_sectors + n_groups  # 7 global + sectors + groups
        
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(
        self,
        text_tokens: torch.Tensor,
        flow_seq: torch.Tensor,
        search_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播，输出热力学状态张量
        
        Args:
            text_tokens: [batch, seq_len, text_dim] 文本特征
            flow_seq: [batch, seq_len, 10] 资金流向序列
            search_vec: [batch, 5] 搜索指数
            
        Returns:
            [batch, output_dim] 热力学状态张量
        """
        # 各模态编码
        text_feat = self.text_encoder(text_tokens).mean(dim=1)  # [batch, text_dim]
        flow_feat, _ = self.flow_encoder(flow_seq)
        flow_feat = flow_feat[:, -1, :]  # [batch, flow_dim]
        search_feat = self.search_encoder(search_vec)  # [batch, search_dim]
        
        # 融合
        fused = torch.cat([text_feat, flow_feat, search_feat], dim=-1)
        output = self.fusion(fused)
        
        return output
    
    def decode_to_state(
        self,
        output: torch.Tensor,
        sector_names: List[str],
        group_names: List[str],
        timestamp: Optional[pd.Timestamp] = None
    ) -> ThermoState:
        """
        将网络输出解码为ThermoState对象
        """
        output = output.detach().cpu().numpy()
        
        return ThermoState(
            ipv=float(output[0]),
            clarity=float(output[1]),
            coherence=float(output[2]),
            entropy=float(output[3]),
            entropy_change=float(output[4]),
            rsi=float(output[5]),
            mts=float(output[6]),
            sector_temperatures=dict(zip(sector_names, output[7:7+self.n_sectors])),
            group_phases=dict(zip(group_names, output[7+self.n_sectors:7+self.n_sectors+self.n_groups])),
            timestamp=timestamp
        )
