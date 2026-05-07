#!/usr/bin/env python
"""
探索ATrader数据可用性
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
import pandas as pd
from thermo_sys.data.atradar_adapter import ATraderDataAdapter


def explore_data():
    adapter = ATraderDataAdapter()
    
    print("="*60)
    print("ATrader 数据探索")
    print("="*60)
    
    # 1. 基本可用性
    avail = adapter.get_data_availability()
    print(f"\n[基本统计]")
    print(f"  CSV股票文件: {avail['csv_stocks']}")
    print(f"  stock_prices记录: {avail.get('stock_prices_count', 0):,}")
    print(f"  money_flow记录: {avail.get('money_flow_count', 0):,}")
    print(f"  sentiment_records记录: {avail.get('sentiment_records_count', 0):,}")
    print(f"  index_prices记录: {avail.get('index_prices_count', 0):,}")
    
    # 2. 查看资金流向数据覆盖的股票
    conn = sqlite3.connect(r'C:\Users\YUXUAN\Desktop\ATrader\aquant.db')
    
    print(f"\n[资金流向覆盖]")
    df = pd.read_sql_query("""
        SELECT symbol, COUNT(*) as cnt, MIN(trade_date) as start, MAX(trade_date) as end
        FROM money_flow
        GROUP BY symbol
        ORDER BY cnt DESC
        LIMIT 10
    """, conn)
    print(df.to_string(index=False))
    
    # 3. 查看情感数据
    print(f"\n[情感数据样本]")
    df = pd.read_sql_query("""
        SELECT * FROM sentiment_records LIMIT 5
    """, conn)
    print(df.to_string())
    
    # 4. 查看指数
    print(f"\n[可用指数]")
    df = pd.read_sql_query("""
        SELECT symbol, COUNT(*) as cnt, MIN(trade_date) as start, MAX(trade_date) as end
        FROM index_prices
        GROUP BY symbol
    """, conn)
    print(df.to_string(index=False))
    
    # 5. 查看某只股票的CSV数据范围
    print(f"\n[CSV数据样本: 000001]")
    csv_data = adapter.load_stock_price_from_csv('000001')
    print(f"  日期范围: {csv_data.index[0]} ~ {csv_data.index[-1]}")
    print(f"  总记录数: {len(csv_data)}")
    print(csv_data.head(3).to_string())
    
    # 6. 查看数据库中000001的数据范围
    print(f"\n[DB数据样本: 000001]")
    db_data = adapter.load_stock_price_from_db('000001')
    if len(db_data) > 0:
        print(f"  日期范围: {db_data.index[0]} ~ {db_data.index[-1]}")
        print(f"  总记录数: {len(db_data)}")
    else:
        print("  无数据")
    
    conn.close()
    adapter.close()


if __name__ == '__main__':
    explore_data()
