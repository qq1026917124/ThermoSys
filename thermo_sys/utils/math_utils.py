"""数学工具函数"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Optional


def zscore(series: Union[np.ndarray, pd.Series], window: Optional[int] = None) -> Union[np.ndarray, pd.Series]:
    """
    计算Z-score标准化
    
    Args:
        series: 输入序列
        window: 滚动窗口，None则使用全序列
        
    Returns:
        Z-score标准化后的序列，限制在[-3, 3]区间
    """
    if isinstance(series, pd.Series):
        if window is not None:
            mean = series.rolling(window, min_periods=window//2).mean()
            std = series.rolling(window, min_periods=window//2).std()
        else:
            mean = series.mean()
            std = series.std()
        result = (series - mean) / (std + 1e-8)
        return result.clip(-3, 3)
    else:
        arr = np.asarray(series)
        if window is not None and len(arr) >= window:
            result = np.zeros_like(arr, dtype=float)
            for i in range(len(arr)):
                start = max(0, i - window + 1)
                window_data = arr[start:i+1]
                result[i] = (arr[i] - np.mean(window_data)) / (np.std(window_data) + 1e-8)
            return np.clip(result, -3, 3)
        else:
            return np.clip((arr - np.mean(arr)) / (np.std(arr) + 1e-8), -3, 3)


def rolling_zscore(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    对DataFrame所有列进行滚动Z-score标准化
    """
    return df.rolling(window, min_periods=window//2).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8)
    ).clip(-3, 3)


def entropy(probabilities: np.ndarray) -> float:
    """
    计算香农熵
    
    Args:
        probabilities: 概率分布数组，和为1
        
    Returns:
        香农熵值
    """
    probs = np.asarray(probabilities)
    probs = probs[probs > 0]  # 避免log(0)
    return -np.sum(probs * np.log2(probs))


def normalized_entropy(probabilities: np.ndarray) -> float:
    """
    归一化熵 [0, 1]
    """
    n = len(probabilities)
    if n <= 1:
        return 0.0
    max_entropy = np.log2(n)
    return entropy(probabilities) / max_entropy


def safe_correlation(x: np.ndarray, y: np.ndarray, max_lag: int = 0) -> float:
    """
    安全计算相关系数，处理常数序列等边界情况
    
    Args:
        x, y: 输入序列
        max_lag: 最大时滞（用于互相关）
        
    Returns:
        相关系数或最大互相关系数
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    
    if len(x_arr) != len(y_arr) or len(x_arr) < 3:
        return 0.0
    
    if np.std(x_arr) < 1e-8 or np.std(y_arr) < 1e-8:
        return 0.0
    
    if max_lag == 0:
        return np.corrcoef(x_arr, y_arr)[0, 1]
    else:
        # 计算最大互相关系数
        max_corr = 0.0
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = np.corrcoef(x_arr[:lag], y_arr[-lag:])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(x_arr[lag:], y_arr[:-lag])[0, 1]
            else:
                corr = np.corrcoef(x_arr, y_arr)[0, 1]
            if not np.isnan(corr) and abs(corr) > abs(max_corr):
                max_corr = corr
        return max_corr


def cross_correlation(series_a: np.ndarray, series_b: np.ndarray, max_lag: int = 24) -> int:
    """
    计算两个序列的最优时滞
    
    Returns:
        最优时滞（小时），越小表示传播越快
    """
    correlations = []
    for lag in range(max_lag + 1):
        if lag == 0:
            corr = np.corrcoef(series_a, series_b)[0, 1]
        else:
            corr = np.corrcoef(series_a[:-lag], series_b[lag:])[0, 1]
        correlations.append(0 if np.isnan(corr) else abs(corr))
    
    return int(np.argmax(correlations))


def granger_causality(x: pd.Series, y: pd.Series, max_lag: int = 5) -> float:
    """
    简化的Granger因果检验，返回F统计量p值
    较小的p值表示x Granger导致y
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        data = pd.concat([y, x], axis=1).dropna()
        if len(data) < max_lag + 10:
            return 1.0
        gc_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        # 返回最小p值
        p_values = [gc_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
        return min(p_values)
    except Exception:
        return 1.0


def mann_kendall_trend(series: np.ndarray) -> tuple:
    """
    Mann-Kendall趋势检验
    
    Returns:
        (tau, p_value)  tau>0表示上升趋势
    """
    from scipy.stats import kendalltau
    data = np.asarray(series)
    n = len(data)
    if n < 3:
        return 0.0, 1.0
    tau, p_value = kendalltau(np.arange(n), data)
    return tau, p_value
