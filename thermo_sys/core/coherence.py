"""
合力形成指标 (Coherence Force, CF)
量化异质投资者从随机分布走向同步的过程
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.signal import hilbert
from thermo_sys.utils.math_utils import entropy, normalized_entropy, safe_correlation


class CoherenceForce:
    """
    合力形成分析引擎
    
    核心指标：
    1. 熵减率 ΔS_norm
    2. Kuramoto序参量 r(t)
    3. 资金-信息共振度
    """
    
    def __init__(
        self,
        entropy_window: int = 5,
        order_param_threshold: float = 0.6,
        resonance_window: int = 20,
        resonance_threshold: float = 0.5,
        n_sentiment_bins: int = 5
    ):
        self.entropy_window = entropy_window
        self.order_threshold = order_param_threshold
        self.resonance_window = resonance_window
        self.resonance_threshold = resonance_threshold
        self.n_bins = n_sentiment_bins
        
    def compute_entropy(
        self,
        sentiment_distribution: pd.DataFrame
    ) -> pd.Series:
        """
        计算观点分布的香农熵
        
        Args:
            sentiment_distribution: DataFrame, 列为各情感档, 行为时间
            
        Returns:
            熵序列
        """
        entropies = []
        
        for idx, row in sentiment_distribution.iterrows():
            probs = row.values
            # 归一化为概率分布
            probs = probs / (probs.sum() + 1e-8)
            ent = normalized_entropy(probs)
            entropies.append(ent)
        
        return pd.Series(entropies, index=sentiment_distribution.index)
    
    def compute_entropy_change(
        self,
        entropy_series: pd.Series,
        jump_window: int = 3
    ) -> pd.DataFrame:
        """
        计算熵减率，识别观点有序化过程
        
        Returns:
            DataFrame with columns: ['entropy', 'entropy_change', 'entropy_jump']
        """
        result = pd.DataFrame(index=entropy_series.index)
        result['entropy'] = entropy_series
        result['entropy_change'] = entropy_series.diff(self.entropy_window)
        
        # 熵跳变检测（3日内从<0.3跃升到>0.7）
        low_entropy = entropy_series < 0.3
        high_entropy = entropy_series > 0.7
        
        # 检测是否从低熵快速跳到高熵（注意：这里应该是从高熵快速跳到低熵才表示有序化）
        # 修正：熵越低表示观点越一致
        result['entropy_jump'] = 0
        for i in range(jump_window, len(entropy_series)):
            recent = entropy_series.iloc[i-jump_window:i]
            if recent.iloc[0] > 0.7 and entropy_series.iloc[i] < 0.3:
                result.iloc[i, result.columns.get_loc('entropy_jump')] = 1
        
        return result
    
    def compute_kuramoto_order(
        self,
        flows_dict: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        计算Kuramoto序参量 r(t)
        
        使用Hilbert变换从资金流向序列提取相位
        
        Args:
            flows_dict: 各投资者群体资金流向序列字典
            weights: 各群体资金权重
            
        Returns:
            序参量序列 [0, 1]
        """
        if not flows_dict:
            return pd.Series()
        
        # 默认等权重
        if weights is None:
            n = len(flows_dict)
            weights = {k: 1.0/n for k in flows_dict.keys()}
        
        # 对齐所有序列
        aligned = pd.concat(flows_dict.values(), axis=1)
        aligned.columns = flows_dict.keys()
        aligned = aligned.dropna()
        
        if len(aligned) < 10:
            return pd.Series(index=aligned.index, data=0.0)
        
        # 对每个序列进行Hilbert变换提取相位
        phases = {}
        for col in aligned.columns:
            signal = aligned[col].values
            # 去趋势
            signal = signal - np.mean(signal)
            # Hilbert变换
            analytic_signal = hilbert(signal)
            phase = np.angle(analytic_signal)
            phases[col] = phase
        
        # 计算序参量
        order_params = []
        for t in range(len(aligned)):
            complex_sum = 0
            total_weight = 0
            
            for group, phase_array in phases.items():
                w = weights.get(group, 1.0)
                complex_sum += w * np.exp(1j * phase_array[t])
                total_weight += w
            
            r = np.abs(complex_sum / total_weight) if total_weight > 0 else 0
            order_params.append(r)
        
        return pd.Series(order_params, index=aligned.index)
    
    def compute_resonance(
        self,
        retail_heat: pd.Series,
        small_flow_next: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        计算资金-信息共振度
        
        Resonance = Corr(散户言论热度, 次日小单净流入)
        
        Args:
            retail_heat: 散户言论热度变化
            small_flow_next: 次日小单净流入（已对齐到T+1）
            window: 滚动窗口
            
        Returns:
            共振度序列
        """
        if window is None:
            window = self.resonance_window
        
        # 计算滚动相关系数
        resonance = retail_heat.rolling(window, min_periods=window//2).corr(
            small_flow_next
        )
        
        return resonance.fillna(0)
    
    def detect_phase_transition(
        self,
        order_param: pd.Series,
        entropy_series: pd.Series,
        velocity: pd.Series
    ) -> pd.DataFrame:
        """
        检测相变临界点
        
        条件：
        1. r从0.3以下突破0.6
        2. 伴随IPV激增
        3. 熵快速下降
        
        Returns:
            DataFrame with columns: ['phase', 'confidence']
            phase: 'trend_formation'(趋势形成), 'chaos'(混沌), 'transition'(过渡)
        """
        result = pd.DataFrame(index=order_param.index)
        
        # 序参量状态
        r_low = order_param < 0.3
        r_high = order_param > 0.6
        r_mid = (~r_low) & (~r_high)
        
        # 速度状态（使用滚动90分位）
        v_high = velocity > velocity.rolling(252, min_periods=60).quantile(0.9)
        
        # 熵状态
        entropy_dropping = entropy_series.diff(3) < -0.2
        
        # 判定相态
        conditions = [
            r_high & v_high & entropy_dropping,   # 趋势形成
            r_low,                                   # 混沌
        ]
        choices = ['trend_formation', 'chaos']
        
        result['phase'] = np.select(conditions, choices, default='transition')
        
        # 置信度
        result['confidence'] = 0.5
        result.loc[r_high & v_high, 'confidence'] = 0.8
        result.loc[r_high & v_high & entropy_dropping, 'confidence'] = 0.95
        result.loc[r_low, 'confidence'] = 0.3
        
        return result
    
    def get_coherence_signal(
        self,
        order_param: pd.Series,
        resonance: pd.Series,
        entropy_change: pd.Series
    ) -> pd.DataFrame:
        """
        生成合力信号
        
        Returns:
            DataFrame with columns: ['signal', 'strength', 'regime']
        """
        result = pd.DataFrame(index=order_param.index)
        
        # 信号判断
        strong_coherence = (order_param > self.order_threshold) & \
                          (resonance > self.resonance_threshold)
        
        entropy_rapid_drop = entropy_change < -0.3
        
        # 趋势市：高序参量 + 高共振
        result['signal'] = 0
        result.loc[strong_coherence & entropy_rapid_drop, 'signal'] = 1
        result.loc[(order_param < 0.3) & (entropy_change > 0.2), 'signal'] = -1
        
        # 强度
        result['strength'] = (
            order_param * 0.4 +
            np.clip(resonance, 0, 1) * 0.3 +
            np.clip(-entropy_change, 0, 1) * 0.3
        )
        
        # 市场体制
        conditions = [
            order_param > 0.6,
            order_param < 0.3,
        ]
        choices = ['trend', 'chaos']
        result['regime'] = np.select(conditions, choices, default='transition')
        
        return result
