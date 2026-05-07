"""数据工具函数"""
import os
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载YAML配置文件
    """
    if config_path is None:
        # 默认查找config目录下的system_config.yaml
        base_dir = Path(__file__).parent.parent.parent
        config_path = base_dir / "config" / "system_config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 递归转换数值字符串为浮点数
    def convert_numeric(obj):
        if isinstance(obj, dict):
            return {k: convert_numeric(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numeric(v) for v in obj]
        elif isinstance(obj, str):
            # 尝试转换为数值
            try:
                if '.' in obj or 'e' in obj.lower():
                    return float(obj)
                else:
                    return int(obj)
            except ValueError:
                return obj
        return obj
    
    return convert_numeric(config)


def time_align(*dataframes: pd.DataFrame, method: str = 'inner') -> list:
    """
    时间对齐多个DataFrame，避免未来函数
    
    Args:
        dataframes: 多个DataFrame，索引为日期
        method: 对齐方法，'inner'交集，'outer'并集
    """
    if not dataframes:
        return []
    
    # 统一转换为日期索引
    aligned_dfs = []
    for df in dataframes:
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        aligned_dfs.append(df)
    
    # 取日期交集/并集
    common_index = aligned_dfs[0].index
    for df in aligned_dfs[1:]:
        if method == 'inner':
            common_index = common_index.intersection(df.index)
        else:
            common_index = common_index.union(df.index)
    
    if method == 'inner':
        return [df.loc[common_index] for df in aligned_dfs]
    else:
        return [df.reindex(common_index) for df in aligned_dfs]


def resample_to_daily(df: pd.DataFrame, agg_func: str = 'last') -> pd.DataFrame:
    """
    将分钟/小时数据降采样为日频数据
    
    Args:
        df: 时间序列数据
        agg_func: 聚合函数
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    agg_map = {
        'last': 'last',
        'first': 'first',
        'mean': 'mean',
        'sum': 'sum',
        'max': 'max',
        'min': 'min'
    }
    
    return df.resample('D').agg(agg_map.get(agg_func, 'last')).dropna()


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """
    使用IQR方法检测异常值
    
    Returns:
        布尔序列，True表示异常值
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return (series < lower_bound) | (series > upper_bound)


def apply_seasonal_adjustment(series: pd.Series, period: int = 5) -> pd.Series:
    """
    简单季节性调整（去除交易日效应）
    
    Args:
        series: 时间序列
        period: 周期，默认5个交易日（周效应）
    """
    seasonal_mean = series.groupby(series.index.to_series().index % period).transform('mean')
    return series - seasonal_mean + series.mean()


def ensure_t_plus_1(signal_df: pd.DataFrame, execution_df: pd.DataFrame) -> pd.DataFrame:
    """
    确保信号使用T日数据，T+1日执行，避免未来函数
    
    Args:
        signal_df: 信号DataFrame，索引为日期
        execution_df: 执行价格DataFrame
        
    Returns:
        对齐后的执行数据
    """
    # 信号使用当日收盘数据计算，次日开盘执行
    signal_df = signal_df.copy()
    signal_df.index = signal_df.index + pd.Timedelta(days=1)
    
    # 取交集
    common_dates = signal_df.index.intersection(execution_df.index)
    return execution_df.loc[common_dates], signal_df.loc[common_dates]
