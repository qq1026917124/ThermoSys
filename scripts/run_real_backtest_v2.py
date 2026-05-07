#!/usr/bin/env python
"""
使用ATrader真实历史数据进行回测 V2
- 2018-2024年：基于价格和成交量的衍生指标（换手率、波动率、价格位置等）
- 2025年10月后：叠加真实资金流向数据
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime

from thermo_sys.data.atradar_adapter import ATraderDataAdapter
from thermo_sys.core import (
    RetailSentimentIndex, InformationPropagationVelocity,
    HeatTransferNetwork, CoherenceForce
)
from thermo_sys.backtest import BacktestEngine
from thermo_sys.utils.math_utils import zscore


def build_proxy_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    从价格和成交量构建代理情绪指标
    
    原理：
    - 换手率激增 → 散户活跃度上升
    - 价格突破近期高点 → 贪婪
    - 价格跌破近期低点 → 恐惧
    - 波动率放大 → 情绪不稳定
    - 放量滞涨 → 主力出货（散户接盘）
    - 缩量下跌 → 散户割肉
    """
    df = price_df.copy()
    
    # 1. 换手率（用volume/流通股本近似，这里用volume标准化）
    df['turnover_rate'] = df['volume'] / df['volume'].rolling(60).mean()
    
    # 2. 价格位置（0-1之间，1表示近期最高）
    rolling_high = df['close'].rolling(60).max()
    rolling_low = df['close'].rolling(60).min()
    df['price_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-8)
    
    # 3. 波动率
    df['volatility'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
    
    # 4. 放量/缩量指标
    df['volume_surge'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # 5. 涨跌动能（RSI-like）
    returns = df['close'].pct_change()
    up = returns.where(returns > 0, 0).rolling(14).mean()
    down = (-returns.where(returns < 0, 0)).rolling(14).mean()
    df['momentum'] = up / (up + down + 1e-8)
    
    # 6. 散户情绪代理：高换手+低涨幅 = 散户高频交易
    df['retail_proxy'] = df['turnover_rate'] * (1 - df['momentum'])
    
    # 7. 融资余额代理（基于价格趋势和成交量）
    trend = df['close'].rolling(20).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
    df['margin_proxy'] = trend * df['volume_surge']
    
    return df


def compute_thermo_from_price(price_df: pd.DataFrame, money_flow_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    从价格数据计算热力学状态
    """
    features = build_proxy_indicators(price_df)
    dates = features.index
    
    # === RSI 子指标 ===
    # 用代理指标替代真实数据
    margin_balance = features['margin_proxy']
    small_flow = features['retail_proxy'] * 10000  # 缩放
    new_accounts = features['turnover_rate'] * 30000
    search_index = features['volume_surge'] * 1000
    option_pcr = 1.0 - features['price_position']  # 高位PCR低（贪婪），低位PCR高（恐惧）
    
    rsi_engine = RetailSentimentIndex()
    rsi = rsi_engine.compute(
        margin_balance, small_flow, new_accounts, search_index, option_pcr
    )
    
    # === IPV 子指标 ===
    mentions = features['volume_surge'] * 100
    engagement = features['turnover_rate'] * 500
    volume_change = features['volume'].pct_change()
    
    ipv_engine = InformationPropagationVelocity()
    rho = ipv_engine.compute_info_density(mentions, engagement, volume_change)
    
    # === 合力子指标 ===
    # 从价格数据构建情感分布
    sentiment_dist = pd.DataFrame(index=dates)
    pos = features['momentum']
    sentiment_dist['extreme_bear'] = (pos < 0.2) * 0.3
    sentiment_dist['bear'] = ((pos >= 0.2) & (pos < 0.4)) * 0.3
    sentiment_dist['neutral'] = ((pos >= 0.4) & (pos < 0.6)) * 0.4
    sentiment_dist['bull'] = ((pos >= 0.6) & (pos < 0.8)) * 0.3
    sentiment_dist['extreme_bull'] = (pos >= 0.8) * 0.3
    sentiment_dist = sentiment_dist.div(sentiment_dist.sum(axis=1), axis=0).fillna(0.2)
    
    coherence_engine = CoherenceForce()
    entropy_series = coherence_engine.compute_entropy(sentiment_dist)
    
    # Kuramoto序参量（基于价格动量和波动率相位）
    phase_momentum = np.sin(np.linspace(0, 4*np.pi, len(dates))) * features['momentum'].values
    phase_vol = np.cos(np.linspace(0, 4*np.pi, len(dates))) * (1 - features['volatility'].values / features['volatility'].max())
    
    flows = {
        'momentum': pd.Series(phase_momentum, index=dates),
        'volatility': pd.Series(phase_vol, index=dates)
    }
    order_param = coherence_engine.compute_kuramoto_order(flows)
    
    # === 整合热力学状态 ===
    thermo_df = pd.DataFrame(index=dates)
    thermo_df['rsi'] = rsi.reindex(dates).fillna(0)
    thermo_df['ipv'] = abs(rho.pct_change()).reindex(dates).fillna(0).clip(0, 10)
    thermo_df['entropy'] = entropy_series.reindex(dates).fillna(1)
    thermo_df['coherence'] = order_param.reindex(dates).fillna(0.5)
    
    # 路径清晰度：基于波动率和趋势一致性
    trend_consistency = 1 - abs(features['close'].pct_change().rolling(5).std())
    thermo_df['clarity'] = trend_consistency.reindex(dates).fillna(0.5).clip(0, 1)
    
    # MTS综合指数
    thermo_df['mts'] = (
        0.3 * zscore(thermo_df['rsi'].fillna(0), 20) +
        0.2 * zscore(thermo_df['ipv'].fillna(0), 20) +
        0.2 * thermo_df['clarity'] +
        0.3 * (2 * thermo_df['coherence'] - 1)
    )
    
    # 如有真实资金流向数据，修正信号
    if money_flow_df is not None and len(money_flow_df) > 0:
        mf = money_flow_df.reindex(dates)
        retail_net = mf['retail_net'].fillna(0)
        main_net = mf['main_net'].fillna(0)
        
        # 散户净流入且主力流出 → 警示信号（散户接盘）
        retail_alert = (retail_net > 0) & (main_net < 0)
        thermo_df.loc[retail_alert, 'rsi'] += 0.5  # 情绪过热修正
    
    return thermo_df


def generate_signals(thermo_df: pd.DataFrame) -> pd.DataFrame:
    """
    基于热力学状态生成交易信号
    """
    signals = pd.DataFrame(index=thermo_df.index)
    signals['signal'] = 0
    signals['strength'] = 0.0
    signals['regime'] = 'neutral'
    
    for idx, row in thermo_df.iterrows():
        rsi = row['rsi']
        coherence = row['coherence']
        clarity = row['clarity']
        mts = row['mts']
        entropy = row['entropy']
        
        # 信号逻辑（基于调研中的择时标准）
        if rsi < -1.5 and coherence > 0.5 and clarity > 0.4:
            # 极端恐惧 + 合力形成 + 路径清晰 → 建仓
            sig = 1
            regime = 'bottom_fishing'
        elif rsi > 1.5 and coherence < 0.4:
            # 极端贪婪 + 合力瓦解 → 减仓
            sig = -1
            regime = 'top_fleeing'
        elif mts > 1.5 and coherence > 0.6 and clarity > 0.6:
            # MTS过热共振态 → 持仓/追涨
            sig = 1
            regime = 'trend_acceleration'
        elif mts < -1.5 and entropy > 0.7:
            # MTS过冷离散态 → 空仓
            sig = -1
            regime = 'panic_wait'
        elif abs(mts) < 0.5:
            # 混沌态 → 低仓位
            sig = 0
            regime = 'chaos'
        else:
            sig = 0
            regime = 'neutral'
        
        signals.loc[idx, 'signal'] = sig
        signals.loc[idx, 'strength'] = min(abs(mts) / 3, 1.0)
        signals.loc[idx, 'regime'] = regime
    
    return signals


def run_backtest(symbol: str = '000001', start_date: str = '2019-01-01', end_date: str = '2024-12-31'):
    """运行真实数据回测"""
    print("="*70)
    print("ThermoSys 真实数据回测 V2")
    print("="*70)
    
    adapter = ATraderDataAdapter()
    
    # 检查数据
    avail = adapter.get_data_availability()
    print(f"\n[数据可用性]")
    print(f"  CSV股票文件: {avail['csv_stocks']} 个")
    print(f"  stock_prices DB: {avail.get('stock_prices_count', 0):,} 条")
    print(f"  money_flow DB: {avail.get('money_flow_count', 0):,} 条")
    print(f"  sentiment_records DB: {avail.get('sentiment_records_count', 0):,} 条")
    
    # 加载数据
    print(f"\n[加载数据] {symbol}")
    
    # 价格数据（CSV更完整）
    try:
        price_df = adapter.load_stock_price_from_csv(symbol, start_date, end_date)
        print(f"  价格数据(CSV): {len(price_df)} 条, {price_df.index[0].date()} ~ {price_df.index[-1].date()}")
    except FileNotFoundError:
        price_df = adapter.load_stock_price_from_db(symbol, start_date, end_date)
        print(f"  价格数据(DB): {len(price_df)} 条")
    
    if len(price_df) == 0:
        print("  错误: 无价格数据")
        return None
    
    # 资金流向（近期才有）
    money_flow = adapter.load_money_flow(symbol, start_date, end_date)
    if len(money_flow) > 0:
        print(f"  资金流向: {len(money_flow)} 条, {money_flow.index[0].date()} ~ {money_flow.index[-1].date()}")
    else:
        print(f"  资金流向: 无数据（将使用代理指标）")
    
    # 计算热力学状态
    print(f"\n[计算热力学状态]")
    thermo_df = compute_thermo_from_price(price_df, money_flow)
    print(f"  RSI范围: {thermo_df['rsi'].min():.2f} ~ {thermo_df['rsi'].max():.2f}")
    print(f"  Coherence范围: {thermo_df['coherence'].min():.2f} ~ {thermo_df['coherence'].max():.2f}")
    print(f"  MTS范围: {thermo_df['mts'].min():.2f} ~ {thermo_df['mts'].max():.2f}")
    
    # 生成信号
    signals = generate_signals(thermo_df)
    
    # 统计信号分布
    signal_counts = signals['signal'].value_counts()
    print(f"\n[信号统计]")
    print(f"  买入信号: {signal_counts.get(1, 0)} 天")
    print(f"  卖出信号: {signal_counts.get(-1, 0)} 天")
    print(f"  中性持仓: {signal_counts.get(0, 0)} 天")
    
    regime_counts = signals['regime'].value_counts()
    print(f"\n[市场体制分布]")
    for regime, count in regime_counts.head(5).items():
        print(f"  {regime}: {count} 天")
    
    # 运行回测
    print(f"\n[运行回测]")
    price = price_df['close']
    
    engine = BacktestEngine(
        initial_capital=1_000_000,
        commission=0.0003,
        slippage=0.001
    )
    
    def signal_generator(thermo):
        return generate_signals(thermo)
    
    result = engine.run(price, signal_generator, thermo_df)
    
    # 输出结果
    print("\n" + "="*70)
    print("回测结果")
    print("="*70)
    
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {key:25s}: {value:>12.4f}")
        else:
            print(f"  {key:25s}: {value}")
    
    initial = result.equity_curve.iloc[0]
    final = result.equity_curve.iloc[-1]
    total_return = (final / initial - 1) * 100
    
    print(f"\n  初始资金: {initial:,.2f}")
    print(f"  最终资金: {final:,.2f}")
    print(f"  总收益率: {total_return:.2f}%")
    print(f"  交易次数: {len([t for t in result.trades if t.action != 'hold'])}")
    
    # 保存结果
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存净值曲线
    result.equity_curve.to_csv(output_dir / f"equity_{symbol}_{timestamp}.csv", header=['equity'])
    
    # 保存热力学状态
    thermo_df.to_csv(output_dir / f"thermo_{symbol}_{timestamp}.csv")
    
    # 保存交易记录
    trades_df = pd.DataFrame([
        {
            'date': t.timestamp,
            'action': t.action,
            'price': t.price,
            'size': t.size,
            'position_before': t.position_before,
            'position_after': t.position_after
        }
        for t in result.trades
    ])
    if len(trades_df) > 0:
        trades_df.to_csv(output_dir / f"trades_{symbol}_{timestamp}.csv", index=False)
    
    print(f"\n[保存结果]")
    print(f"  backtest_results/equity_{symbol}_{timestamp}.csv")
    print(f"  backtest_results/thermo_{symbol}_{timestamp}.csv")
    print(f"  backtest_results/trades_{symbol}_{timestamp}.csv")
    
    adapter.close()
    
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='000001', help='股票代码')
    parser.add_argument('--start', default='2019-01-01', help='开始日期')
    parser.add_argument('--end', default='2024-12-31', help='结束日期')
    args = parser.parse_args()
    
    run_backtest(args.symbol, args.start, args.end)
