"""
回测模块单元测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from thermo_sys.backtest import BacktestEngine, calculate_metrics


class TestBacktestEngine:
    def test_simple_backtest(self):
        n = 100
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B')
        
        # 生成价格
        returns = np.random.randn(n) * 0.01
        price = 100 * (1 + returns).cumprod()
        price = pd.Series(price, index=dates)
        
        # 生成热力学状态
        thermo_df = pd.DataFrame({
            'rsi': np.random.randn(n),
            'coherence': np.random.uniform(0, 1, n),
            'clarity': np.random.uniform(0, 1, n),
            'entropy': np.random.uniform(0, 1, n),
            'entropy_change': np.random.randn(n) * 0.1,
            'mts': np.random.randn(n),
        }, index=dates)
        
        def signal_generator(thermo):
            signals = pd.DataFrame(index=thermo.index)
            signals['signal'] = np.where(thermo['rsi'] < -1.5, 1, 
                                       np.where(thermo['rsi'] > 1.5, -1, 0))
            signals['strength'] = 0.5
            return signals
        
        engine = BacktestEngine(initial_capital=1000000)
        result = engine.run(price, signal_generator, thermo_df)
        
        assert len(result.equity_curve) == n - 1  # T+1执行少一天
        assert len(result.returns) == n - 1
        assert 'sharpe_ratio' in result.metrics
        assert 'max_drawdown' in result.metrics
        
    def test_position_limits(self):
        n = 50
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B')
        price = pd.Series(100 + np.cumsum(np.random.randn(n)), index=dates)
        
        thermo_df = pd.DataFrame({
            'rsi': np.zeros(n),
            'coherence': np.ones(n) * 0.8,
            'clarity': np.ones(n) * 0.9,
            'entropy': np.zeros(n),
            'entropy_change': np.zeros(n),
            'mts': np.zeros(n),
        }, index=dates)
        
        def signal_generator(thermo):
            signals = pd.DataFrame(index=thermo.index)
            signals['signal'] = 1  # 一直买入
            signals['strength'] = 1.0
            return signals
        
        engine = BacktestEngine(position_limits=(-0.5, 0.5))
        result = engine.run(price, signal_generator, thermo_df)
        
        # 检查仓位是否被限制
        assert result.positions.abs().max() <= 0.5 + 1e-6


class TestMetrics:
    def test_calculate_metrics(self):
        returns = pd.Series(np.random.randn(100) * 0.01)
        metrics = calculate_metrics(returns)
        
        assert 'sharpe_ratio' in metrics
        assert 'total_return' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert metrics['sharpe_ratio'] is not None
        
    def test_mann_kendall(self):
        from thermo_sys.utils.math_utils import mann_kendall_trend
        
        increasing = np.arange(100)
        tau, p = mann_kendall_trend(increasing)
        assert tau > 0.9  # 强上升趋势
        
        decreasing = np.arange(100, 0, -1)
        tau, p = mann_kendall_trend(decreasing)
        assert tau < -0.9  # 强下降趋势


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
