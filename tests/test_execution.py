"""
测试交易执行模块
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from thermo_sys.execution.manual_executor import (
    ManualTradeExecutor,
    TradeSignal,
    WeeklyStrategy,
    ActionType
)
from thermo_sys.execution.weekly_backtest import WeeklyStrategyBacktest, BacktestConfig


class TestManualTradeExecutor:
    """测试手动交易执行器"""
    
    def test_creation(self):
        executor = ManualTradeExecutor()
        assert executor is not None
    
    def test_generate_weekly_plan(self):
        executor = ManualTradeExecutor()
        market_analysis = {
            'rsi': -2.0,
            'coherence': 0.7,
            'clarity': 0.8,
            'ipv': 1.5
        }
        stock_pool = ['000001', '000002', '600000']
        
        plan = executor.generate_weekly_plan(market_analysis, stock_pool)
        assert isinstance(plan, WeeklyStrategy)
        assert len(plan.target_stocks) > 0
    
    def test_generate_signals(self):
        executor = ManualTradeExecutor()
        
        mock_states = {
            '000001': {'rsi': -2.3, 'coherence': 0.75, 'clarity': 0.8, 'entropy': 0.3, 'ipv': 1.8},
            '000002': {'rsi': 1.8, 'coherence': 0.2, 'clarity': 0.3, 'entropy': 0.9, 'ipv': 0.5}
        }
        current_pos = {'000001': 0.2}
        
        signals = executor.generate_daily_signals(mock_states, current_pos)
        assert len(signals) == 2
        assert all(isinstance(s, TradeSignal) for s in signals)
    
    def test_signal_reasoning(self):
        executor = ManualTradeExecutor()
        
        # 测试极端恐惧买入
        state = {'rsi': -2.3, 'coherence': 0.75, 'clarity': 0.8, 'entropy': 0.3, 'ipv': 1.8}
        signal = executor._generate_signal_for_stock('000001', state, 0.0)
        assert signal.action == ActionType.BUY
        assert signal.confidence > 0.8
        
        # 测试极端贪婪卖出
        state = {'rsi': 2.3, 'coherence': 0.2, 'clarity': 0.3, 'entropy': 0.9, 'ipv': 0.5}
        signal = executor._generate_signal_for_stock('000001', state, 0.5)
        assert signal.action == ActionType.SELL
    
    def test_daily_report(self):
        executor = ManualTradeExecutor()
        report = executor.generate_daily_report()
        assert isinstance(report, str)
        assert "每日交易报告" in report


class TestWeeklyBacktest:
    """测试周内策略回测"""
    
    def test_creation(self):
        config = BacktestConfig()
        backtest = WeeklyStrategyBacktest(config)
        assert backtest is not None
        assert backtest.capital == config.initial_capital
    
    def test_backtest_run(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='B')
        n = len(dates)
        
        # 模拟价格数据
        price_data = pd.DataFrame(index=dates)
        for stock in ['000001', '000002']:
            returns = np.random.randn(n) * 0.02
            price_data[stock] = 100 * (1 + returns).cumprod()
        
        # 模拟热力学数据
        thermo_data = {}
        for stock in ['000001', '000002']:
            thermo_data[stock] = pd.DataFrame({
                'rsi': np.random.randn(n),
                'coherence': np.random.rand(n),
                'clarity': np.random.rand(n),
                'entropy': np.random.rand(n),
                'ipv': np.random.randn(n) * 2
            }, index=dates)
        
        config = BacktestConfig()
        backtest = WeeklyStrategyBacktest(config)
        result = backtest.run(price_data, thermo_data)
        
        assert 'total_return' in result
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result
    
    def test_stop_loss(self):
        config = BacktestConfig(stop_loss=-0.05)
        backtest = WeeklyStrategyBacktest(config)
        
        # 添加持仓
        backtest.positions['000001'] = {
            'position': 0.3,
            'entry_price': 100,
            'entry_date': datetime.now(),
            'value': 300000,
            'volume': 3000
        }
        
        # 价格跌至触发止损
        dates = pd.date_range('2024-01-01', periods=1)
        price_data = pd.DataFrame({'000001': [94.0]}, index=dates)  # -6%
        
        backtest._check_stop_loss_take_profit(dates[0], price_data)
        
        # 持仓应该被清空
        assert '000001' not in backtest.positions


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
