"""
交易记录工具
用于手动记录实际交易执行情况
"""
import json
import argparse
from datetime import datetime
from pathlib import Path


def record_trade(symbol: str, action: str, price: float, volume: int, notes: str = ""):
    """
    记录一笔交易
    
    Usage:
        python scripts/record_trade.py --symbol 000001 --action buy --price 10.5 --volume 1000
    """
    trade = {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'action': action,
        'price': price,
        'volume': volume,
        'amount': price * volume,
        'notes': notes
    }
    
    # 保存到文件
    data_dir = Path("data/trades")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    trades_file = data_dir / "executed_trades.json"
    
    trades = []
    if trades_file.exists():
        with open(trades_file, 'r', encoding='utf-8') as f:
            trades = json.load(f)
    
    trades.append(trade)
    
    with open(trades_file, 'w', encoding='utf-8') as f:
        json.dump(trades, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 交易已记录: {symbol} {action} {volume}股 @ {price}")
    print(f"  总金额: {trade['amount']:,.2f}")
    print(f"  文件: {trades_file}")


def main():
    parser = argparse.ArgumentParser(description='记录交易执行情况')
    parser.add_argument('--symbol', required=True, help='股票代码')
    parser.add_argument('--action', required=True, choices=['buy', 'sell'], help='操作')
    parser.add_argument('--price', required=True, type=float, help='成交价格')
    parser.add_argument('--volume', required=True, type=int, help='成交数量')
    parser.add_argument('--notes', default='', help='备注')
    
    args = parser.parse_args()
    record_trade(args.symbol, args.action, args.price, args.volume, args.notes)


if __name__ == '__main__':
    main()
