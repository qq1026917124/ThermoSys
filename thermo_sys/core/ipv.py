"""
信息传播速度指标 (Information Propagation Velocity, IPV)
量化信息在异质网络中的有效扩散速率
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.signal import correlate
from thermo_sys.utils.math_utils import zscore, cross_correlation


class InformationPropagationVelocity:
    """
    信息传播速度计算引擎
    
    核心公式：
    v_i(t) = |∇ρ_i(t)| / ρ_i(t) * D * 1/(1 + τ_lag(t))
    
    其中：
    - ρ_i: 信息密度场
    - D: 扩散系数
    - τ_lag: 跨平台时滞
    """
    
    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
        diffusion_coefficient: float = 0.5,
        max_lag_hours: int = 24,
        r0_threshold: float = 2.0,
        velocity_percentile: int = 90
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.D = diffusion_coefficient
        self.max_lag = max_lag_hours
        self.r0_threshold = r0_threshold
        self.velocity_percentile = velocity_percentile
        
    def compute_info_density(
        self,
        mentions: pd.Series,
        engagement: pd.Series,
        volume_change: pd.Series
    ) -> pd.Series:
        """
        计算信息密度场 ρ(t)
        
        ρ(t) = α * Z(mentions) + β * Z(engagement) + γ * Z(Δvolume)
        """
        z_mentions = zscore(mentions, window=60)
        z_engagement = zscore(engagement, window=60)
        z_volume = zscore(volume_change, window=60)
        
        rho = (
            self.alpha * z_mentions +
            self.beta * z_engagement +
            self.gamma * z_volume
        )
        return rho
    
    def compute_density_gradient(
        self,
        rho_dict: Dict[str, pd.Series],
        target_sector: str,
        neighbor_sectors: List[str]
    ) -> pd.Series:
        """
        计算信息密度梯度 |∇ρ_i|
        
        |∇ρ_i| = sqrt(Σ_j (ρ_i - ρ_j)^2)
        """
        rho_i = rho_dict[target_sector]
        gradients = []
        
        for neighbor in neighbor_sectors:
            if neighbor in rho_dict:
                rho_j = rho_dict[neighbor]
                aligned_i, aligned_j = rho_i.align(rho_j, join='inner')
                gradient = (aligned_i - aligned_j) ** 2
                gradients.append(gradient)
        
        if not gradients:
            return pd.Series(0, index=rho_i.index)
        
        # 合并所有梯度
        grad_df = pd.concat(gradients, axis=1)
        grad_norm = np.sqrt(grad_df.sum(axis=1))
        
        return grad_norm.reindex(rho_i.index).fillna(0)
    
    def compute_cross_platform_lag(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        max_lag: Optional[int] = None
    ) -> int:
        """
        计算跨平台时滞 τ_lag
        
        Returns:
            最优时滞（小时），越小传播越快
        """
        if max_lag is None:
            max_lag = self.max_lag
        
        # 对齐并填充
        aligned_a, aligned_b = series_a.align(series_b, join='inner')
        
        if len(aligned_a) < max_lag * 2:
            return max_lag  # 数据不足，返回最大时滞
        
        arr_a = aligned_a.values
        arr_b = aligned_b.values
        
        return cross_correlation(arr_a, arr_b, max_lag)
    
    def compute_velocity(
        self,
        rho: pd.Series,
        gradient: pd.Series,
        lag: Optional[int] = None
    ) -> pd.Series:
        """
        计算传播速度 v(t)
        
        v(t) = |∇ρ| / ρ * D / (1 + τ_lag)
        """
        # 避免除零
        safe_rho = rho.replace(0, np.nan).fillna(rho.median())
        
        # 扩散项
        diffusion_term = gradient / np.abs(safe_rho) * self.D
        
        # 时滞衰减
        lag = lag or 0
        lag_decay = 1.0 / (1.0 + lag / 6.0)  # 归一化，6小时为基准
        
        velocity = diffusion_term * lag_decay
        
        return velocity
    
    def compute_r0(
        self,
        new_spreaders: pd.Series,
        susceptible: pd.Series,
        forgetting_rate: pd.Series
    ) -> pd.Series:
        """
        计算信息再生数 R_0 = β / γ
        
        R_0 > 2: 指数级扩散
        R_0 < 1: 无法形成有效传播
        """
        beta = new_spreaders / (susceptible + 1)
        gamma = forgetting_rate.replace(0, np.nan).fillna(0.1)
        
        r0 = beta / gamma
        return r0
    
    def get_signal(
        self,
        velocity: pd.Series,
        r0: pd.Series,
        lookback: int = 252
    ) -> pd.DataFrame:
        """
        基于传播速度生成信号
        
        Returns:
            DataFrame with columns: ['phase', 'intensity']
            phase: 'acceleration'(加速), 'saturation'(饱和), 'decay'(衰减), 'dormant'(休眠)
        """
        signals = pd.DataFrame(index=velocity.index)
        
        # 计算历史分位数
        v_high = velocity.rolling(lookback, min_periods=60).quantile(self.velocity_percentile / 100)
        v_low = velocity.rolling(lookback, min_periods=60).quantile(1 - self.velocity_percentile / 100)
        
        # 判断传播阶段
        conditions = [
            (velocity > v_high) & (r0 > self.r0_threshold),
            (velocity > v_high) & (r0 <= self.r0_threshold),
            (velocity < v_low),
            True  # default
        ]
        choices = ['acceleration', 'saturation', 'decay', 'dormant']
        
        signals['phase'] = np.select(conditions, choices, default='dormant')
        signals['intensity'] = np.clip(velocity / (v_high + 1e-8), 0, 1)
        
        return signals
