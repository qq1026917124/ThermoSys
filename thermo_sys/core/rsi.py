"""
散户情绪指数 (Retail Sentiment Index, RSI)
基于行为数据为主、言论数据为辅的客观情绪量化
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
from thermo_sys.utils.math_utils import zscore, rolling_zscore


@dataclass
class RSIWeights:
    """RSI子指标权重配置"""
    margin_balance_change: float = 0.35
    small_order_flow: float = 0.25
    new_account_growth: float = 0.20
    search_index_change: float = 0.15
    option_pcr: float = 0.05


class RetailSentimentIndex:
    """
    散户情绪指数计算引擎
    
    核心公式：
    RSI = 0.35 * Δ融资余额_Zscore 
        + 0.25 * 小单净流入_Zscore 
        + 0.20 * 新增开户数_Zscore 
        + 0.15 * 搜索指数变化_Zscore 
        + 0.05 * 期权PCR_Zscore
    """
    
    def __init__(
        self,
        weights: Optional[RSIWeights] = None,
        zscore_window: int = 20,
        long_term_window: int = 60,
        lookback_percentile: int = 500
    ):
        self.weights = weights or RSIWeights()
        self.zscore_window = zscore_window
        self.long_term_window = long_term_window
        self.lookback_percentile = lookback_percentile
        self._history: pd.Series = pd.Series(dtype=float)
        
    def compute(
        self,
        margin_balance: pd.Series,
        small_order_flow: pd.Series,
        new_accounts: pd.Series,
        search_index: pd.Series,
        option_pcr: pd.Series
    ) -> pd.Series:
        """
        计算散户情绪指数
        
        Args:
            margin_balance: 融资余额序列
            small_order_flow: 小单净流入序列（万元）
            new_accounts: 新增开户数序列
            search_index: 搜索指数序列
            option_pcr: 期权PCR序列
            
        Returns:
            RSI序列，值域约[-3, 3]
        """
        # 计算各子指标的Z-score
        # 1. 融资余额变化率（20日）
        margin_change = margin_balance.pct_change(20)
        z_margin = zscore(margin_change, self.zscore_window)
        
        # 2. 小单净流入5日累计
        small_flow_5d = small_order_flow.rolling(5).sum()
        z_small = zscore(small_flow_5d, self.zscore_window)
        
        # 3. 新增开户数环比
        account_change = new_accounts.pct_change(5)
        z_account = zscore(account_change, self.zscore_window)
        
        # 4. 搜索指数20日变化
        search_change = search_index.pct_change(20)
        z_search = zscore(search_change, self.zscore_window)
        
        # 5. 期权PCR（反向指标，高PCR=恐慌）
        z_pcr = -zscore(option_pcr, self.zscore_window)  # 取负，PCR越高情绪越低
        
        # 加权合成RSI
        rsi = (
            self.weights.margin_balance_change * z_margin +
            self.weights.small_order_flow * z_small +
            self.weights.new_account_growth * z_account +
            self.weights.search_index_change * z_search +
            self.weights.option_pcr * z_pcr
        )
        
        # 长周期平滑
        rsi_smooth = rsi.rolling(self.long_term_window, min_periods=10).mean()
        
        # 保存历史用于动态阈值
        self._history = rsi_smooth.dropna()
        
        return rsi_smooth
    
    def get_signal(
        self,
        rsi: pd.Series,
        method: str = 'rolling_percentile'
    ) -> pd.DataFrame:
        """
        生成择时信号
        
        Args:
            rsi: RSI序列
            method: 'fixed_threshold' 或 'rolling_percentile'
            
        Returns:
            DataFrame包含signal和strength列
        """
        signals = pd.DataFrame(index=rsi.index)
        signals['rsi'] = rsi
        signals['signal'] = 0  # 0: 中性, 1: 买入, -1: 卖出
        signals['strength'] = 0.0
        
        if method == 'fixed_threshold':
            # 固定阈值法
            buy_mask = rsi < -2.0
            sell_mask = rsi > 2.0
            
        elif method == 'rolling_percentile':
            # 滚动分位数法（适应市场结构变化）
            lower = rsi.rolling(self.lookback_percentile, min_periods=100).quantile(0.05)
            upper = rsi.rolling(self.lookback_percentile, min_periods=100).quantile(0.95)
            
            buy_mask = rsi < lower
            sell_mask = rsi > upper
        else:
            raise ValueError(f"Unknown method: {method}")
        
        signals.loc[buy_mask, 'signal'] = 1
        signals.loc[sell_mask, 'signal'] = -1
        
        # 信号强度 [0, 1]
        signals['strength'] = np.clip(np.abs(rsi) / 3.0, 0, 1)
        
        return signals
    
    def detect_divergence(
        self,
        price: pd.Series,
        rsi: pd.Series,
        window: int = 20
    ) -> pd.DataFrame:
        """
        检测情绪-价格背离
        
        Returns:
            DataFrame with columns: ['bullish_divergence', 'bearish_divergence']
        """
        result = pd.DataFrame(index=price.index)
        
        # 计算局部极值
        price_low = price.rolling(window, center=True).min()
        price_high = price.rolling(window, center=True).max()
        rsi_low = rsi.rolling(window, center=True).min()
        rsi_high = rsi.rolling(window, center=True).max()
        
        # 看涨背离：价格创新低，RSI未创新低
        result['bullish_divergence'] = (
            (price == price_low) & (rsi > rsi_low.shift(window//2))
        ).astype(int)
        
        # 看跌背离：价格创新高，RSI未创新高
        result['bearish_divergence'] = (
            (price == price_high) & (rsi < rsi_high.shift(window//2))
        ).astype(int)
        
        return result
    
    def get_dynamic_weights(
        self,
        df: pd.DataFrame,
        future_return: pd.Series,
        window: int = 252
    ) -> Dict[str, float]:
        """
        使用滚动回归动态校准权重（月度优化）
        
        Args:
            df: 各子指标Z-score的DataFrame
            future_return: 未来5日收益率
            window: 滚动窗口
            
        Returns:
            优化后的权重字典
        """
        from sklearn.linear_model import Ridge
        
        # 对齐数据
        aligned_df, aligned_ret = df.align(future_return, join='inner', axis=0)
        
        if len(aligned_df) < window:
            # 数据不足，返回默认权重
            return self.weights.__dict__
        
        # 滚动回归
        recent_df = aligned_df.iloc[-window:]
        recent_ret = aligned_ret.iloc[-window:]
        
        model = Ridge(alpha=1.0)
        model.fit(recent_df.fillna(0), recent_ret.fillna(0))
        
        # 归一化权重
        raw_weights = np.abs(model.coef_)
        normalized = raw_weights / raw_weights.sum()
        
        col_names = recent_df.columns.tolist()
        return {col: float(w) for col, w in zip(col_names, normalized)}
