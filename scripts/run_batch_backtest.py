#!/usr/bin/env python
"""
批量回测多只股票，验证策略稳健性
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime

from scripts.run_real_backtest_v2 import run_backtest


def main():
    # 选择不同行业的代表性股票
    symbols = [
        '000001',  # 平安银行（银行）
        '000002',  # 万科A（地产）
        '000063',  # 中兴通讯（通信）
        '000100',  # TCL科技（电子）
        '000333',  # 美的集团（家电）
        '000538',  # 云南白药（医药）
        '000568',  # 泸州老窖（白酒）
        '000725',  # 京东方A（面板）
        '000768',  # 中航西飞（军工）
        '002001',  # 新和成（化工）
    ]
    
    start_date = '2019-01-01'
    end_date = '2024-12-31'
    
    print("="*70)
    print("ThermoSys 批量回测")
    print("="*70)
    
    results = []
    
    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"回测: {symbol}")
        print('='*70)
        
        try:
            result = run_backtest(symbol, start_date, end_date)
            if result:
                metrics = result.metrics
                initial = result.equity_curve.iloc[0]
                final = result.equity_curve.iloc[-1]
                total_ret = (final / initial - 1) * 100
                
                results.append({
                    'symbol': symbol,
                    'total_return': total_ret,
                    'sharpe': metrics['sharpe_ratio'],
                    'max_dd': metrics['max_drawdown'] * 100,
                    'win_rate': metrics['win_rate'] * 100,
                    'volatility': metrics['volatility'] * 100,
                })
        except Exception as e:
            print(f"  错误: {e}")
    
    # 汇总
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*70)
        print("批量回测汇总")
        print("="*70)
        print(df.to_string(index=False))
        
        print(f"\n[统计摘要]")
        print(f"  平均收益率: {df['total_return'].mean():.2f}%")
        print(f"  平均夏普比: {df['sharpe'].mean():.4f}")
        print(f"  正收益股票: {(df['total_return'] > 0).sum()}/{len(df)}")
        print(f"  平均最大回撤: {df['max_dd'].mean():.2f}%")
        
        # 保存汇总
        output_dir = Path("backtest_results")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(output_dir / f"batch_summary_{timestamp}.csv", index=False)
        print(f"\n[保存] backtest_results/batch_summary_{timestamp}.csv")


if __name__ == '__main__':
    main()
