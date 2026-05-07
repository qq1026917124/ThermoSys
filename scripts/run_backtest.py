#!/usr/bin/env python
"""
回测运行脚本
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from thermo_sys.main import ThermoSystem, argparse


def run_quick_backtest():
    """快速回测示例"""
    system = ThermoSystem()
    
    import numpy as np
    import pandas as pd
    from datetime import datetime
    
    # 生成模拟数据
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='B')
    n = len(dates)
    
    returns = np.random.randn(n) * 0.015
    price = pd.Series(100 * (1 + returns).cumprod(), index=dates)
    
    mock_data = {
        'margin_balance': pd.Series(15000 + np.cumsum(np.random.randn(n)*100), index=dates),
        'money_flow': pd.DataFrame({
            'small_inflow': np.random.randn(n) * 5000,
            'main_inflow': np.random.randn(n) * 10000,
        }, index=dates),
        'new_accounts': pd.Series(30000 + np.random.randn(n) * 5000, index=dates),
        'option_pcr': pd.Series(0.9 + np.random.randn(n) * 0.1, index=dates),
        'posts': pd.DataFrame({
            'post_count': np.random.randint(100, 1000, n),
            'comment_count': np.random.randint(500, 5000, n),
        }, index=dates),
        'sentiment': pd.DataFrame(
            np.random.dirichlet([1,2,4,2,1], n),
            index=dates,
            columns=['extreme_bear', 'bear', 'neutral', 'bull', 'extreme_bull']
        ),
        'search_index': pd.DataFrame(
            np.random.randint(1000, 5000, (n, 6)),
            index=dates,
            columns=['股票开户', '牛市', '涨停', '割肉', '跌停', '清仓']
        )
    }
    
    print("Running backtest...")
    result = system.run_backtest(price, mock_data)
    
    print("\n=== Backtest Results ===")
    for key, value in result['metrics'].items():
        if isinstance(value, float):
            print(f"  {key:25s}: {value:>12.4f}")
    
    print(f"\n  Final Equity: {result['equity_curve'].iloc[-1]:,.2f}")
    print(f"  Total Return: {(result['equity_curve'].iloc[-1]/result['equity_curve'].iloc[0]-1)*100:.2f}%")
    
    return result


if __name__ == '__main__':
    run_quick_backtest()
