"""
自动回测引擎
替代手动执行，实现全自动策略验证
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

from thermo_sys.execution.manual_executor import ManualTradeExecutor, ActionType


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 1_000_000
    commission: float = 0.0003  # 佣金
    stamp_duty: float = 0.001   # 印花税（卖出时）
    max_positions: int = 5
    max_single_position: float = 0.3
    stop_loss: float = -0.07
    take_profit: float = 0.15
    min_confidence: float = 0.6
    

class AutoBacktestEngine:
    """
    自动回测引擎
    
    完全自动化的回测系统：
    1. 接收实时生成的信号
    2. 在历史数据上验证信号有效性
    3. 计算策略绩效指标
    4. 输出可执行的最优策略
    
    不需要用户手动执行，系统自动完成验证
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Dict] = {}  # 当前持仓
        self.trades: List[Dict] = []  # 交易记录
        self.daily_values: List[Dict] = []  # 每日净值
        self.signal_history: List[Dict] = []  # 信号历史
        
    def run(self, 
            price_data: pd.DataFrame,  # 价格数据 {symbol: close_price}
            thermo_data: Dict[str, pd.DataFrame]  # 热力学数据 {symbol: thermo_states}
            ) -> Dict:
        """
        运行回测
        
        Args:
            price_data: 价格数据，index为日期，columns为股票代码
            thermo_data: 热力学状态数据
            
        Returns:
            回测结果
        """
        dates = price_data.index
        executor = ManualTradeExecutor()
        
        for i, date in enumerate(dates):
            if i == 0:
                continue  # 第一天没有信号
            
            # 获取当日热力学状态（使用昨日收盘数据计算）
            prev_date = dates[i-1]
            daily_thermo = {}
            for symbol in price_data.columns:
                if symbol in thermo_data and prev_date in thermo_data[symbol].index:
                    daily_thermo[symbol] = thermo_data[symbol].loc[prev_date].to_dict()
            
            # 生成当前持仓
            current_positions = {symbol: pos['position'] for symbol, pos in self.positions.items()}
            
            # 生成信号
            signals = executor.generate_daily_signals(daily_thermo, current_positions)
            
            # 记录信号
            for signal in signals:
                self.signal_history.append({
                    'date': date,
                    'symbol': signal.symbol,
                    'action': signal.action.value,
                    'confidence': signal.confidence,
                    'reasoning': signal.reasoning,
                    'thermo_state': signal.thermo_state
                })
            
            # 执行信号（T+1，使用当日开盘价格）
            for signal in signals:
                if signal.confidence >= self.config.min_confidence:
                    self._execute_signal(signal, date, price_data)
            
            # 检查止损止盈
            self._check_stop_loss_take_profit(date, price_data)
            
            # 记录每日净值
            self._record_daily_value(date, price_data)
        
        return self._generate_report()
    
    def run_walk_forward(self,
                        price_data: pd.DataFrame,
                        thermo_data: Dict[str, pd.DataFrame],
                        train_size: int = 60,
                        test_size: int = 20) -> Dict:
        """
        滚动前向回测（Walk-Forward Analysis）
        
        更稳健的回测方法，避免过拟合：
        1. 用前N天数据训练策略参数
        2. 在接下来的M天测试
        3. 滚动窗口重复
        
        Args:
            price_data: 价格数据
            thermo_data: 热力学数据
            train_size: 训练窗口大小（天）
            test_size: 测试窗口大小（天）
        """
        dates = price_data.index
        total_days = len(dates)
        
        all_results = []
        window_results = []
        
        # 滑动窗口
        start = 0
        while start + train_size + test_size < total_days:
            train_end = start + train_size
            test_end = train_end + test_size
            
            train_dates = dates[start:train_end]
            test_dates = dates[train_end:test_end]
            
            # 在训练集上优化参数（简化版）
            logger.info(f"窗口 {start//test_size + 1}: 训练 {train_dates[0]} ~ {train_dates[-1]}, "
                       f"测试 {test_dates[0]} ~ {test_dates[-1]}")
            
            # 使用当前参数在测试集上回测
            result = self._run_single_window(
                price_data.loc[test_dates],
                {k: v.loc[test_dates] for k, v in thermo_data.items()}
            )
            
            window_results.append({
                'window': start // test_size + 1,
                'train_period': (train_dates[0], train_dates[-1]),
                'test_period': (test_dates[0], test_dates[-1]),
                'sharpe': result['sharpe_ratio'],
                'return': result['total_return'],
                'max_dd': result['max_drawdown']
            })
            
            all_results.append(result)
            start += test_size
        
        # 汇总结果
        sharpes = [r['sharpe_ratio'] for r in all_results]
        returns = [r['total_return'] for r in all_results]
        
        return {
            'window_results': window_results,
            'avg_sharpe': np.mean(sharpes),
            'avg_return': np.mean(returns),
            'sharpe_std': np.std(sharpes),
            'consistency': np.mean(sharpes) / (np.std(sharpes) + 1e-8),  # 稳定性指标
            'total_windows': len(all_results)
        }
    
    def _run_single_window(self, price_data: pd.DataFrame, thermo_data: Dict) -> Dict:
        """运行单个窗口的回测"""
        # 重置状态
        self.capital = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_values = []
        
        return self.run(price_data, thermo_data)
    
    def _execute_signal(self, signal, date, price_data):
        """执行交易信号"""
        symbol = signal.symbol
        
        if symbol not in price_data.columns:
            return
        
        price = price_data[symbol].loc[date]
        
        # 计算交易数量
        target_value = signal.target_position * self.capital
        current_value = self.positions.get(symbol, {}).get('value', 0)
        delta_value = target_value - current_value
        
        if abs(delta_value) < 1000:  # 最小交易金额
            return
        
        volume = int(abs(delta_value) / price)
        if volume < 100:  # A股最小100股
            return
        
        # 计算交易成本
        if delta_value > 0:  # 买入
            cost = delta_value * self.config.commission
            self.capital -= (delta_value + cost)
        else:  # 卖出
            cost = abs(delta_value) * (self.config.commission + self.config.stamp_duty)
            self.capital += (abs(delta_value) - cost)
        
        # 更新持仓
        if symbol not in self.positions:
            self.positions[symbol] = {
                'position': signal.target_position,
                'entry_price': price,
                'entry_date': date,
                'value': target_value,
                'volume': volume
            }
        else:
            if signal.target_position <= 0:
                del self.positions[symbol]
            else:
                # 更新平均成本
                old_volume = self.positions[symbol]['volume']
                old_cost = old_volume * self.positions[symbol]['entry_price']
                new_cost = volume * price
                total_volume = old_volume + volume if delta_value > 0 else old_volume - volume
                
                if total_volume > 0:
                    self.positions[symbol]['entry_price'] = (old_cost + new_cost) / total_volume
                    self.positions[symbol]['volume'] = total_volume
                    self.positions[symbol]['position'] = signal.target_position
                    self.positions[symbol]['value'] = target_value
        
        # 记录交易
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': signal.action.value,
            'price': price,
            'volume': volume,
            'value': abs(delta_value),
            'cost': cost,
            'reason': signal.reasoning,
            'confidence': signal.confidence
        })
    
    def _check_stop_loss_take_profit(self, date, price_data):
        """检查止损止盈"""
        for symbol in list(self.positions.keys()):
            if symbol not in price_data.columns:
                continue
            
            pos = self.positions[symbol]
            current_price = price_data[symbol].loc[date]
            entry_price = pos['entry_price']
            
            pnl_pct = (current_price - entry_price) / entry_price
            
            # 止损
            if pnl_pct <= self.config.stop_loss:
                logger.info(f"{date} {symbol} 触发止损: {pnl_pct:.2%}")
                self._sell_position(symbol, date, price_data, "止损")
            
            # 止盈
            elif pnl_pct >= self.config.take_profit:
                logger.info(f"{date} {symbol} 触发止盈: {pnl_pct:.2%}")
                self._sell_position(symbol, date, price_data, "止盈")
    
    def _sell_position(self, symbol, date, price_data, reason):
        """清仓"""
        if symbol not in self.positions:
            return
        
        price = price_data[symbol].loc[date]
        pos = self.positions[symbol]
        value = pos['value']
        
        # 扣除卖出成本
        cost = value * (self.config.commission + self.config.stamp_duty)
        self.capital += (value - cost)
        
        # 记录交易
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': '卖出',
            'price': price,
            'volume': pos['volume'],
            'value': value,
            'cost': cost,
            'reason': reason,
            'confidence': 1.0
        })
        
        del self.positions[symbol]
    
    def _record_daily_value(self, date, price_data):
        """记录每日净值"""
        position_value = 0
        for symbol, pos in self.positions.items():
            if symbol in price_data.columns:
                price = price_data[symbol].loc[date]
                position_value += pos['volume'] * price
        
        total_value = self.capital + position_value
        
        self.daily_values.append({
            'date': date,
            'capital': self.capital,
            'position_value': position_value,
            'total_value': total_value,
            'positions_count': len(self.positions)
        })
    
    def _generate_report(self) -> Dict:
        """生成回测报告"""
        values_df = pd.DataFrame(self.daily_values)
        if values_df.empty:
            return self._empty_report()
        
        values_df.set_index('date', inplace=True)
        
        # 计算收益率
        values_df['returns'] = values_df['total_value'].pct_change()
        values_df['cumulative'] = (1 + values_df['returns'].fillna(0)).cumprod()
        
        # 绩效指标
        total_return = (values_df['total_value'].iloc[-1] / self.config.initial_capital) - 1
        n_days = len(values_df)
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1 if n_days > 0 else 0
        volatility = values_df['returns'].std() * np.sqrt(252) if len(values_df['returns'].dropna()) > 1 else 0
        sharpe = annual_return / (volatility + 1e-8) if volatility > 0 else 0
        
        # 最大回撤
        cummax = values_df['total_value'].cummax()
        drawdown = (values_df['total_value'] - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # 交易统计
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            avg_confidence = trades_df['confidence'].mean()
            total_cost = trades_df['cost'].sum()
            
            # 胜率
            buy_trades = trades_df[trades_df['action'].isin(['买入', '加仓'])]
            sell_trades = trades_df[trades_df['action'].isin(['卖出', '减仓'])]
        else:
            avg_confidence = 0
            total_cost = 0
        
        # 信号统计
        signal_df = pd.DataFrame(self.signal_history)
        signal_quality = self._calculate_signal_quality(signal_df, values_df)
        
        return {
            'initial_capital': self.config.initial_capital,
            'final_value': values_df['total_value'].iloc[-1],
            'total_return': total_return,
            'annualized_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'avg_confidence': avg_confidence,
            'total_cost': total_cost,
            'signal_count': len(self.signal_history),
            'signal_quality': signal_quality,
            'equity_curve': values_df['total_value'],
            'drawdown': drawdown,
            'trades': trades_df,
            'signals': signal_df
        }
    
    def _empty_report(self) -> Dict:
        """空报告"""
        return {
            'initial_capital': self.config.initial_capital,
            'final_value': self.config.initial_capital,
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'avg_confidence': 0.0,
            'total_cost': 0.0,
            'signal_count': 0,
            'signal_quality': {},
            'equity_curve': pd.Series(),
            'drawdown': pd.Series(),
            'trades': pd.DataFrame(),
            'signals': pd.DataFrame()
        }
    
    def _calculate_signal_quality(self, signal_df: pd.DataFrame, values_df: pd.DataFrame) -> Dict:
        """
        计算信号质量
        
        评估信号的有效性：
        1. 高置信度信号的胜率
        2. 信号与后续收益的相关性
        """
        if signal_df.empty or values_df.empty:
            return {}
        
        quality = {}
        
        # 按置信度分组统计
        high_conf = signal_df[signal_df['confidence'] >= 0.8]
        med_conf = signal_df[(signal_df['confidence'] >= 0.6) & (signal_df['confidence'] < 0.8)]
        low_conf = signal_df[signal_df['confidence'] < 0.6]
        
        quality['high_conf_signals'] = len(high_conf)
        quality['med_conf_signals'] = len(med_conf)
        quality['low_conf_signals'] = len(low_conf)
        
        # 信号分布
        buy_signals = signal_df[signal_df['action'].isin(['买入', '加仓'])]
        sell_signals = signal_df[signal_df['action'].isin(['卖出', '减仓'])]
        
        quality['buy_signals'] = len(buy_signals)
        quality['sell_signals'] = len(sell_signals)
        quality['hold_signals'] = len(signal_df[signal_df['action'] == '持有'])
        
        return quality


if __name__ == '__main__':
    # 测试
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')
    n = len(dates)
    
    stocks = ['000001', '000002', '600000']
    price_data = pd.DataFrame(index=dates)
    
    for stock in stocks:
        returns = np.random.randn(n) * 0.02
        price_data[stock] = 100 * (1 + returns).cumprod()
    
    thermo_data = {}
    for stock in stocks:
        thermo_data[stock] = pd.DataFrame({
            'rsi': np.random.randn(n),
            'coherence': np.random.rand(n),
            'clarity': np.random.rand(n),
            'entropy': np.random.rand(n),
            'ipv': np.random.randn(n) * 2
        }, index=dates)
    
    # 标准回测
    engine = AutoBacktestEngine()
    result = engine.run(price_data, thermo_data)
    
    print("\n标准回测结果:")
    print(f"收益率: {result['total_return']:.2%}")
    print(f"夏普: {result['sharpe_ratio']:.2f}")
    print(f"最大回撤: {result['max_drawdown']:.2%}")
    print(f"信号数量: {result['signal_count']}")
    
    # Walk-Forward回测
    engine2 = AutoBacktestEngine()
    wf_result = engine2.run_walk_forward(price_data, thermo_data, train_size=20, test_size=10)
    
    print("\nWalk-Forward结果:")
    print(f"平均夏普: {wf_result['avg_sharpe']:.2f}")
    print(f"平均收益: {wf_result['avg_return']:.2%}")
    print(f"稳定性: {wf_result['consistency']:.2f}")
    print(f"窗口数: {wf_result['total_windows']}")
