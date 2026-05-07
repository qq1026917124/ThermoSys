#!/usr/bin/env python
"""
使用ATrader真实历史数据进行回测
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
from thermo_sys.backtest import BacktestEngine, calculate_metrics
from thermo_sys.utils.math_utils import zscore


def prepare_thermo_features(adapter: ATraderDataAdapter, symbol: str, start_date: str, end_date: str):
    """
    从真实数据构建热力学特征
    """
    data = adapter.build_thermo_dataset(symbol, start_date=start_date, end_date=end_date)
    
    price = data['price']
    money_flow = data['money_flow']
    sentiment = data['sentiment']
    
    if len(price) == 0:
        raise ValueError(f"No price data for {symbol}")
    
    # 构建统一的日期索引
    dates = price.index
    
    # 1. 构建RSI需要的子指标
    # 融资余额（用零售净流入近似，或生成模拟数据）
    if len(money_flow) > 0:
        # 使用资金流向数据
        small_flow = money_flow['retail_net'].reindex(dates).fillna(0)
        main_flow = money_flow['main_net'].reindex(dates).fillna(0)
    else:
        small_flow = pd.Series(0, index=dates)
        main_flow = pd.Series(0, index=dates)
    
    # 模拟融资余额（基于价格趋势+波动）
    margin_balance = price['close'].rolling(20).mean() * 1000 + np.random.randn(len(price)) * 500
    
    # 模拟新增开户数（基于市场热度）
    returns = price['close'].pct_change()
    market_heat = returns.rolling(20).std() * 10000
    new_accounts = 30000 + market_heat.fillna(0) + np.random.randn(len(price)) * 3000
    
    # 搜索指数（从情感数据提取或模拟）
    if len(sentiment) > 0:
        search_index = sentiment['heat_score'].reindex(dates).fillna(method='ffill').fillna(50)
    else:
        search_index = pd.Series(50 + np.random.randn(len(price)) * 10, index=dates)
    
    # 期权PCR（模拟）
    option_pcr = pd.Series(0.9 + np.random.randn(len(price)) * 0.1, index=dates)
    
    # 2. 构建IPV需要的指标
    if len(sentiment) > 0:
        mentions = sentiment['news_count'].reindex(dates).fillna(0)
        engagement = sentiment['heat_score'].reindex(dates).fillna(0) * 10
    else:
        mentions = pd.Series(np.random.randint(50, 200, len(price)), index=dates)
        engagement = pd.Series(np.random.randint(200, 800, len(price)), index=dates)
    
    volume_change = price['volume'].pct_change().fillna(0)
    
    # 3. 构建情感分布
    sentiment_dist = pd.DataFrame(index=dates)
    if len(sentiment) > 0:
        score = sentiment['sentiment_score'].reindex(dates).fillna(0.5)
        # 将单一分数转为五档分布
        sentiment_dist['extreme_bear'] = (score < 0.2).astype(float) * 0.3
        sentiment_dist['bear'] = ((score >= 0.2) & (score < 0.4)).astype(float) * 0.3
        sentiment_dist['neutral'] = ((score >= 0.4) & (score < 0.6)).astype(float) * 0.4
        sentiment_dist['bull'] = ((score >= 0.6) & (score < 0.8)).astype(float) * 0.3
        sentiment_dist['extreme_bull'] = (score >= 0.8).astype(float) * 0.3
        # 归一化
        sentiment_dist = sentiment_dist.div(sentiment_dist.sum(axis=1), axis=0).fillna(0.2)
    else:
        # 随机分布
        probs = np.random.dirichlet([1, 2, 4, 2, 1], len(price))
        sentiment_dist = pd.DataFrame(probs, index=dates, 
                                      columns=['extreme_bear', 'bear', 'neutral', 'bull', 'extreme_bull'])
    
    return {
        'price': price,
        'margin_balance': margin_balance,
        'small_order_flow': small_flow,
        'new_accounts': new_accounts,
        'search_index': search_index,
        'option_pcr': option_pcr,
        'mentions': mentions,
        'engagement': engagement,
        'volume_change': volume_change,
        'sentiment_dist': sentiment_dist,
        'main_flow': main_flow,
    }


def compute_thermo_signals(features: dict):
    """
    计算热力学信号
    """
    # 1. RSI
    rsi_engine = RetailSentimentIndex()
    rsi = rsi_engine.compute(
        features['margin_balance'],
        features['small_order_flow'],
        features['new_accounts'],
        features['search_index'],
        features['option_pcr']
    )
    
    # 2. IPV
    ipv_engine = InformationPropagationVelocity()
    rho = ipv_engine.compute_info_density(
        features['mentions'],
        features['engagement'],
        features['volume_change']
    )
    
    # 3. 合力
    coherence_engine = CoherenceForce()
    entropy_series = coherence_engine.compute_entropy(features['sentiment_dist'])
    
    # 构建资金流字典用于Kuramoto
    flows = {
        'retail': features['small_order_flow'],
        'main': features['main_flow']
    }
    # 过滤掉全零序列
    flows = {k: v for k, v in flows.items() if v.abs().sum() > 0}
    
    if len(flows) >= 2:
        order_param = coherence_engine.compute_kuramoto_order(flows)
    else:
        order_param = pd.Series(0.5, index=features['price'].index)
    
    # 4. 构建热力学状态DataFrame
    thermo_df = pd.DataFrame(index=features['price'].index)
    thermo_df['rsi'] = rsi.reindex(thermo_df.index).fillna(0)
    thermo_df['ipv'] = abs(rho.pct_change()).reindex(thermo_df.index).fillna(0)
    thermo_df['entropy'] = entropy_series.reindex(thermo_df.index).fillna(1)
    thermo_df['coherence'] = order_param.reindex(thermo_df.index).fillna(0.5)
    thermo_df['clarity'] = np.random.uniform(0.3, 0.9, len(thermo_df))  # 简化
    thermo_df['mts'] = (
        0.4 * zscore(thermo_df['ipv'].fillna(0), 20) +
        0.3 * thermo_df['clarity'] +
        0.3 * (2 * thermo_df['coherence'] - 1)
    )
    
    return thermo_df


def run_backtest(symbol: str = '000001', start_date: str = '2020-01-01', end_date: str = '2024-12-31'):
    """运行真实数据回测"""
    print("="*70)
    print("ThermoSys 真实数据回测")
    print("="*70)
    
    # 1. 初始化数据适配器
    adapter = ATraderDataAdapter()
    
    # 检查数据可用性
    avail = adapter.get_data_availability()
    print(f"\n[数据可用性]")
    print(f"  CSV股票文件: {avail['csv_stocks']} 个")
    print(f"  数据库存在: {avail['db_path_exists']}")
    print(f"  stock_prices记录: {avail.get('stock_prices_count', 0):,}")
    print(f"  money_flow记录: {avail.get('money_flow_count', 0):,}")
    print(f"  sentiment_records记录: {avail.get('sentiment_records_count', 0):,}")
    
    # 2. 准备特征
    print(f"\n[构建特征] {symbol}")
    features = prepare_thermo_features(adapter, symbol, start_date, end_date)
    
    print(f"\n[计算热力学信号]")
    thermo_df = compute_thermo_signals(features)
    
    # 3. 定义信号生成函数
    def signal_generator(thermo):
        signals = pd.DataFrame(index=thermo.index)
        signals['signal'] = 0
        signals['strength'] = 0.0
        
        for idx, row in thermo.iterrows():
            rsi = row['rsi']
            coherence = row['coherence']
            mts = row['mts']
            
            # 信号逻辑
            if rsi < -1.5 and coherence > 0.5:
                sig = 1
            elif rsi > 1.5 and coherence < 0.4:
                sig = -1
            elif mts > 1.5:
                sig = 1
            elif mts < -1.5:
                sig = -1
            else:
                sig = 0
            
            signals.loc[idx, 'signal'] = sig
            signals.loc[idx, 'strength'] = min(abs(mts) / 3, 1.0)
        
        return signals
    
    # 4. 运行回测
    print(f"\n[运行回测]")
    price = features['price']['close']
    
    engine = BacktestEngine(
        initial_capital=1_000_000,
        commission=0.0003,
        slippage=0.001
    )
    
    result = engine.run(price, signal_generator, thermo_df)
    
    # 5. 输出结果
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
    
    # 6. 保存结果
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result.equity_curve.to_csv(output_dir / f"equity_{symbol}_{timestamp}.csv")
    
    print(f"\n[保存结果] backtest_results/equity_{symbol}_{timestamp}.csv")
    
    adapter.close()
    
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='000001', help='股票代码')
    parser.add_argument('--start', default='2020-01-01', help='开始日期')
    parser.add_argument('--end', default='2024-12-31', help='结束日期')
    args = parser.parse_args()
    
    run_backtest(args.symbol, args.start, args.end)
