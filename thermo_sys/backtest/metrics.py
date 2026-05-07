"""
绩效指标计算与统计检验
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.03) -> Dict[str, float]:
    """
    计算完整的绩效指标集
    
    Args:
        returns: 日收益率序列
        risk_free_rate: 无风险利率（年化）
        
    Returns:
        指标字典
    """
    metrics = {}
    
    # 基础统计
    metrics['mean_return'] = returns.mean()
    metrics['std_return'] = returns.std()
    metrics['skewness'] = returns.skew()
    metrics['kurtosis'] = returns.kurtosis()
    
    # 收益指标
    metrics['total_return'] = (1 + returns).prod() - 1
    metrics['annualized_return'] = (1 + metrics['total_return']) ** (252 / len(returns)) - 1
    metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
    
    # 风险调整收益
    excess_return = metrics['annualized_return'] - risk_free_rate
    metrics['sharpe_ratio'] = excess_return / (metrics['annualized_volatility'] + 1e-8)
    
    # 下行风险
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-8
    metrics['sortino_ratio'] = excess_return / downside_std
    
    # 最大回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    metrics['max_drawdown_duration'] = _max_drawdown_duration(drawdown)
    
    # Calmar
    metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'] + 1e-8)
    
    # 胜率与盈亏比
    positive = returns[returns > 0]
    negative = returns[returns < 0]
    metrics['win_rate'] = len(positive) / len(returns)
    metrics['profit_loss_ratio'] = (positive.mean() / abs(negative.mean())) if len(negative) > 0 else 0
    
    # 统计检验
    metrics['t_statistic'], metrics['p_value'] = stats.ttest_1samp(returns, 0)
    
    return metrics


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    """计算最大回撤持续天数"""
    is_drawdown = drawdown < 0
    if not is_drawdown.any():
        return 0
    
    # 找连续回撤的最长区间
    durations = []
    current_duration = 0
    
    for in_dd in is_drawdown:
        if in_dd:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
            current_duration = 0
    
    if current_duration > 0:
        durations.append(current_duration)
    
    return max(durations) if durations else 0


def granger_causality_test(
    cause_series: pd.Series,
    effect_series: pd.Series,
    max_lag: int = 5
) -> Dict[str, float]:
    """
    Granger因果检验
    
    Returns:
        包含p值和结论的字典
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        
        data = pd.concat([effect_series, cause_series], axis=1).dropna()
        
        if len(data) < max_lag + 10:
            return {'p_value': 1.0, 'has_causality': False, 'reason': 'insufficient_data'}
        
        gc_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        
        # 收集所有lag的p值
        p_values = {}
        for lag in range(1, max_lag + 1):
            p_val = gc_result[lag][0]['ssr_ftest'][1]
            p_values[f'lag_{lag}'] = p_val
        
        min_p = min(p_values.values())
        
        return {
            'p_values': p_values,
            'min_p_value': min_p,
            'has_causality': min_p < 0.05,
            'best_lag': min(p_values.keys(), key=lambda k: p_values[k])
        }
    
    except Exception as e:
        return {'p_value': 1.0, 'has_causality': False, 'error': str(e)}


def mann_kendall_trend_test(series: pd.Series) -> Dict[str, float]:
    """
    Mann-Kendall趋势检验
    
    Returns:
        tau, p_value, has_trend
    """
    from scipy.stats import kendalltau
    
    data = series.dropna().values
    n = len(data)
    
    if n < 3:
        return {'tau': 0.0, 'p_value': 1.0, 'has_trend': False}
    
    tau, p_value = kendalltau(np.arange(n), data)
    
    return {
        'tau': tau,
        'p_value': p_value,
        'has_trend': p_value < 0.05,
        'trend_direction': 'increasing' if tau > 0 else 'decreasing'
    }


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Bootstrap置信区间估计
    
    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    point_estimate = statistic_func(data)
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    return point_estimate, lower, upper


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    信息比率
    
    Args:
        returns: 策略收益率
        benchmark_returns: 基准收益率
        
    Returns:
        信息比率
    """
    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std() * np.sqrt(252)
    
    return active_returns.mean() * 252 / (tracking_error + 1e-8)
