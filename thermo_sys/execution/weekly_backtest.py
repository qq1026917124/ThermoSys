"""
周内策略回测
验证基于热力学的周内交易策略
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger

from thermo_sys.core import ThermoState
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
    

class WeeklyStrategyBacktest:
    """
    周内策略回测引擎
    
    策略逻辑：
    1. 周一开盘前生成周计划
    2. 每日收盘后分析，生成次日信号
    3. T+1执行（信号次日开盘执行）
    4. 周五收盘后总结，调整下周计划
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Dict] = {}  # 当前持仓
        self.trades: List[Dict] = []  # 交易记录
        self.daily_values: List[Dict] = []  # 每日净值
        
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
            
            # 执行信号（T+1，使用当日开盘价格）
            for signal in signals:
                if signal.confidence >= self.config.min_confidence:
                    self._execute_signal(signal, date, price_data)
            
            # 检查止损止盈
            self._check_stop_loss_take_profit(date, price_data)
            
            # 记录每日净值
            self._record_daily_value(date, price_data)
        
        return self._generate_report()
    
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
        values_df.set_index('date', inplace=True)
        
        # 计算收益率
        values_df['returns'] = values_df['total_value'].pct_change()
        values_df['cumulative'] = (1 + values_df['returns'].fillna(0)).cumprod()
        
        # 绩效指标
        total_return = (values_df['total_value'].iloc[-1] / self.config.initial_capital) - 1
        annual_return = (1 + total_return) ** (252 / len(values_df)) - 1
        volatility = values_df['returns'].std() * np.sqrt(252)
        sharpe = annual_return / (volatility + 1e-8)
        
        # 最大回撤
        cummax = values_df['total_value'].cummax()
        drawdown = (values_df['total_value'] - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # 交易统计
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            buy_trades = trades_df[trades_df['action'].isin(['买入', '加仓'])]
            sell_trades = trades_df[trades_df['action'].isin(['卖出', '减仓'])]
            
            avg_confidence = trades_df['confidence'].mean()
            total_cost = trades_df['cost'].sum()
        else:
            avg_confidence = 0
            total_cost = 0
        
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
            'equity_curve': values_df['total_value'],
            'drawdown': drawdown,
            'trades': trades_df
        }


def run_weekly_backtest_example():
    """运行周内策略回测示例"""
    # 生成模拟数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='B')
    n = len(dates)
    
    # 模拟3只股票的价格
    stocks = ['000001', '000002', '600000']
    price_data = pd.DataFrame(index=dates)
    
    for stock in stocks:
        returns = np.random.randn(n) * 0.02
        price_data[stock] = 100 * (1 + returns).cumprod()
    
    # 模拟热力学数据
    thermo_data = {}
    for stock in stocks:
        thermo_data[stock] = pd.DataFrame({
            'rsi': np.random.randn(n),
            'coherence': np.random.rand(n),
            'clarity': np.random.rand(n),
            'entropy': np.random.rand(n),
            'ipv': np.random.randn(n) * 2
        }, index=dates)
    
    # 运行回测
    config = BacktestConfig()
    backtest = WeeklyStrategyBacktest(config)
    result = backtest.run(price_data, thermo_data)
    
    # 输出结果
    print("\n" + "="*60)
    print("周内策略回测结果")
    print("="*60)
    print(f"初始资金: {result['initial_capital']:,.0f}")
    print(f"最终资金: {result['final_value']:,.0f}")
    print(f"总收益率: {result['total_return']:.2%}")
    print(f"年化收益: {result['annualized_return']:.2%}")
    print(f"波动率: {result['volatility']:.2%}")
    print(f"夏普比率: {result['sharpe_ratio']:.2f}")
    print(f"最大回撤: {result['max_drawdown']:.2%}")
    print(f"交易次数: {result['total_trades']}")
    print(f"平均置信度: {result['avg_confidence']:.1%}")
    print(f"总交易成本: {result['total_cost']:,.2f}")
    
    return result


if __name__ == '__main__':
    run_weekly_backtest_example()
