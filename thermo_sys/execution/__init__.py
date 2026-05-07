"""
交易执行模块
"""
from .manual_executor import ManualTradeExecutor, TradeSignal, WeeklyStrategy, ActionType
from .weekly_backtest import WeeklyStrategyBacktest, BacktestConfig
from .auto_backtest import AutoBacktestEngine
from .strategy_optimizer import StrategyOptimizer, StrategyParameters
from .end_to_end_loop import AutoEvolutionLoop

__all__ = [
    "ManualTradeExecutor",
    "TradeSignal", 
    "WeeklyStrategy",
    "ActionType",
    "WeeklyStrategyBacktest",
    "BacktestConfig",
    "AutoBacktestEngine",
    "StrategyOptimizer",
    "StrategyParameters",
    "AutoEvolutionLoop"
]