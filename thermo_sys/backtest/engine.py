"""
回测引擎
支持热力学Agent的完整回测，包括信号生成、执行、绩效分析
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TradeRecord:
    """交易记录"""
    timestamp: datetime
    action: str  # 'buy', 'sell', 'hold'
    price: float
    size: float
    position_before: float
    position_after: float
    signal_strength: float
    thermo_state: Dict[str, float]


@dataclass
class BacktestResult:
    """回测结果"""
    equity_curve: pd.Series
    trades: List[TradeRecord]
    returns: pd.Series
    positions: pd.Series
    signals: pd.DataFrame
    metrics: Dict[str, float]
    thermo_history: pd.DataFrame


class BacktestEngine:
    """
    热力学回测引擎
    
    特点：
    1. T+1执行，避免未来函数
    2. 支持热力学状态驱动的仓位管理
    3. 详细的交易记录和状态追踪
    """
    
    def __init__(
        self,
        initial_capital: float = 10_000_000,
        commission: float = 0.0003,
        slippage: float = 0.001,
        position_limits: tuple = (-1.0, 1.0)  # 最小/最大仓位
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.min_position, self.max_position = position_limits
        
    def run(
        self,
        price: pd.Series,
        signal_generator: Callable[[pd.DataFrame], pd.DataFrame],
        thermo_states: Optional[pd.DataFrame] = None,
        position_sizer: Optional[Callable[[float, Dict], float]] = None
    ) -> BacktestResult:
        """
        运行回测
        
        Args:
            price: 价格序列（收盘价用于计算净值，开盘价用于执行）
            signal_generator: 信号生成函数，输入thermo_states，输出signal DataFrame
            thermo_states: 热力学状态历史
            position_sizer: 仓位管理函数，输入信号强度和热力学状态，输出目标仓位
            
        Returns:
            BacktestResult
        """
        if thermo_states is None:
            raise ValueError("thermo_states is required for thermodynamic backtest")
        
        # 生成信号
        signals = signal_generator(thermo_states)
        
        # 对齐数据（T日信号，T+1执行）
        aligned_price, aligned_signals = self._align_for_execution(price, signals)
        aligned_thermo = thermo_states.loc[aligned_signals.index]
        
        # 初始化
        capital = self.initial_capital
        position = 0.0
        equity = []
        trades = []
        positions = []
        returns = []
        
        for date in aligned_signals.index:
            current_price = aligned_price.loc[date]
            signal = aligned_signals.loc[date]
            thermo = aligned_thermo.loc[date].to_dict() if date in aligned_thermo.index else {}
            
            # 计算目标仓位
            if position_sizer is not None:
                target_position = position_sizer(signal, thermo)
            else:
                target_position = self._default_position_sizer(signal, thermo)
            
            # 限制仓位
            target_position = np.clip(target_position, self.min_position, self.max_position)
            
            # 执行交易（如果需要调仓）
            if abs(target_position - position) > 0.05:  # 最小调仓阈值
                trade_size = target_position - position
                
                # 滑点
                execution_price = current_price * (1 + self.slippage * np.sign(trade_size))
                
                # 佣金
                trade_value = abs(trade_size) * execution_price * self.initial_capital
                commission_cost = trade_value * self.commission
                
                # 更新资金
                capital -= commission_cost
                
                trades.append(TradeRecord(
                    timestamp=date,
                    action='buy' if trade_size > 0 else 'sell',
                    price=execution_price,
                    size=trade_size,
                    position_before=position,
                    position_after=target_position,
                    signal_strength=signal.get('strength', abs(target_position)),
                    thermo_state=thermo
                ))
                
                position = target_position
            
            # 计算当日净值
            position_value = position * current_price * self.initial_capital / price.iloc[0]
            total_value = capital + position_value
            equity.append(total_value)
            positions.append(position)
            
            # 计算日收益率
            if len(equity) > 1:
                daily_return = (equity[-1] / equity[-2]) - 1
                returns.append(daily_return)
            else:
                returns.append(0)
        
        # 构建结果
        equity_series = pd.Series(equity, index=aligned_signals.index)
        returns_series = pd.Series(returns, index=aligned_signals.index)
        positions_series = pd.Series(positions, index=aligned_signals.index)
        
        # 计算绩效指标
        metrics = self._calculate_metrics(returns_series, equity_series)
        
        return BacktestResult(
            equity_curve=equity_series,
            trades=trades,
            returns=returns_series,
            positions=positions_series,
            signals=signals,
            metrics=metrics,
            thermo_history=aligned_thermo
        )
    
    def _align_for_execution(
        self,
        price: pd.Series,
        signals: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        T日信号，T+1日开盘执行
        确保不对齐到未来数据
        """
        # 信号使用T日收盘数据，次日开盘执行
        execution_price = price.shift(-1).dropna()  # T+1开盘（近似用T+1收盘）
        
        # 取交集
        common_idx = price.index.intersection(signals.index).intersection(execution_price.index)
        
        return execution_price.loc[common_idx], signals.loc[common_idx]
    
    def _default_position_sizer(
        self,
        signal: pd.Series,
        thermo: Dict[str, float]
    ) -> float:
        """
        默认仓位管理：基于RSI和合力信号
        
        - 极端恐惧(-2)+趋势形成: 满仓
        - 极端贪婪(+2)+趋势瓦解: 空仓
        - 混沌态: 低仓位
        """
        rsi = thermo.get('rsi', 0)
        coherence = thermo.get('coherence', 0)
        clarity = thermo.get('clarity', 0.5)
        
        # 基础信号
        if rsi < -2.0 and coherence > 0.6:
            base_position = 1.0
        elif rsi > 2.0 and coherence < 0.3:
            base_position = -0.5  # 可以做空或空仓
        elif rsi < -1.5:
            base_position = 0.5
        elif rsi > 1.5:
            base_position = 0.0
        else:
            base_position = 0.3
        
        # 清晰度衰减（路径不清晰时降低仓位）
        clarity_multiplier = min(clarity / 0.3, 1.0) if clarity < 0.3 else 1.0
        
        # 熵惩罚（高熵时降低仓位）
        entropy = thermo.get('entropy', 1.0)
        entropy_multiplier = max(1.0 - entropy, 0.2)
        
        position = base_position * clarity_multiplier * entropy_multiplier
        return np.clip(position, -1.0, 1.0)
    
    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity: pd.Series
    ) -> Dict[str, float]:
        """计算回测绩效指标"""
        metrics = {}
        
        # 基础收益指标
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        metrics['total_return'] = total_return
        metrics['annualized_return'] = (1 + total_return) ** (252 / len(returns)) - 1
        
        # 风险指标
        metrics['volatility'] = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = metrics['annualized_return'] / (metrics['volatility'] + 1e-8)
        
        # 最大回撤
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        metrics['max_drawdown'] = drawdown.min()
        
        # Calmar比率
        metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'] + 1e-8)
        
        # 胜率
        positive_returns = returns[returns > 0]
        metrics['win_rate'] = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        
        # 盈亏比
        avg_gain = positive_returns.mean() if len(positive_returns) > 0 else 0
        negative_returns = returns[returns < 0]
        avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 1
        metrics['profit_loss_ratio'] = avg_gain / avg_loss if avg_loss > 0 else 0
        
        # 索提诺比率（下行波动率）
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-8
        metrics['sortino_ratio'] = metrics['annualized_return'] / downside_std
        
        # 热力学特异性指标
        metrics['num_trading_days'] = len(returns)
        
        return metrics


class WalkForwardValidator:
    """
    滚动前向验证
    避免过拟合，验证系统稳健性
    """
    
    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 63,
        step_size: int = 63
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        
    def run(
        self,
        price: pd.Series,
        thermo_states: pd.DataFrame,
        signal_generator_factory: Callable,
        n_splits: int = 5
    ) -> List[Dict[str, Any]]:
        """
        执行滚动前向验证
        
        Returns:
            每折的验证结果列表
        """
        results = []
        n = len(price)
        
        for i in range(n_splits):
            # 划分训练/测试集
            test_start = n - (i + 1) * self.step_size - self.test_window
            test_end = test_start + self.test_window
            train_start = max(0, test_start - self.train_window)
            
            if test_start < 0:
                break
            
            train_price = price.iloc[train_start:test_start]
            test_price = price.iloc[test_start:test_end]
            train_thermo = thermo_states.iloc[train_start:test_start]
            test_thermo = thermo_states.iloc[test_start:test_end]
            
            # 在训练集上训练信号生成器
            signal_gen = signal_generator_factory(train_thermo)
            
            # 在测试集上回测
            engine = BacktestEngine()
            result = engine.run(test_price, signal_gen, test_thermo)
            
            results.append({
                'split': i + 1,
                'train_period': (train_price.index[0], train_price.index[-1]),
                'test_period': (test_price.index[0], test_price.index[-1]),
                'metrics': result.metrics,
                'sharpe': result.metrics['sharpe_ratio']
            })
        
        return results
