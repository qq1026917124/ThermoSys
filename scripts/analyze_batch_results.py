#!/usr/bin/env python
"""
分析已有的批量回测结果
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from glob import glob


def analyze_equity_file(path: str) -> dict:
    """从净值曲线计算绩效指标"""
    df = pd.read_csv(path)
    df.columns = ['date', 'equity']
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    returns = df['equity'].pct_change().dropna()
    
    if len(returns) < 10:
        return None
    
    total_ret = (df['equity'].iloc[-1] / df['equity'].iloc[0] - 1) * 100
    ann_ret = (1 + total_ret/100) ** (252 / len(returns)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / (vol + 1e-8)
    
    cummax = df['equity'].cummax()
    dd = (df['equity'] - cummax) / cummax
    max_dd = dd.min() * 100
    
    win_rate = (returns > 0).mean() * 100
    
    # 从文件名提取symbol
    symbol = Path(path).stem.split('_')[1]
    
    return {
        'symbol': symbol,
        'total_return_%': total_ret,
        'annualized_return_%': ann_ret * 100,
        'sharpe': sharpe,
        'max_drawdown_%': max_dd,
        'win_rate_%': win_rate,
        'volatility_%': vol * 100,
        'trading_days': len(returns)
    }


def main():
    result_dir = Path("backtest_results")
    equity_files = sorted(result_dir.glob("equity_*.csv"))
    
    print("="*80)
    print("ThermoSys 批量回测结果分析")
    print("="*80)
    print(f"\n找到 {len(equity_files)} 条净值曲线\n")
    
    results = []
    for f in equity_files:
        r = analyze_equity_file(str(f))
        if r:
            results.append(r)
    
    if not results:
        print("无有效结果")
        return
    
    df = pd.DataFrame(results)
    
    # 去重（保留最新）
    df = df.drop_duplicates(subset=['symbol'], keep='last')
    
    print(df.to_string(index=False))
    
    print("\n" + "="*80)
    print("统计摘要")
    print("="*80)
    
    print(f"\n  样本数: {len(df)}")
    print(f"  平均收益率: {df['total_return_%'].mean():.2f}%")
    print(f"  中位数收益率: {df['total_return_%'].median():.2f}%")
    print(f"  正收益比例: {(df['total_return_%'] > 0).sum()}/{len(df)} ({(df['total_return_%'] > 0).mean()*100:.1f}%)")
    print(f"  平均夏普比: {df['sharpe'].mean():.4f}")
    print(f"  平均最大回撤: {df['max_drawdown_%'].mean():.2f}%")
    print(f"  平均胜率: {df['win_rate_%'].mean():.2f}%")
    
    # 最佳/最差
    best = df.loc[df['total_return_%'].idxmax()]
    worst = df.loc[df['total_return_%'].idxmin()]
    
    print(f"\n  最佳标的: {best['symbol']} (收益{best['total_return_%']:.2f}%, 夏普{best['sharpe']:.4f})")
    print(f"  最差标的: {worst['symbol']} (收益{worst['total_return_%']:.2f}%, 夏普{worst['sharpe']:.4f})")
    
    # 保存
    df.to_csv(result_dir / "batch_summary.csv", index=False)
    print(f"\n[保存] {result_dir / 'batch_summary.csv'}")


if __name__ == '__main__':
    main()
