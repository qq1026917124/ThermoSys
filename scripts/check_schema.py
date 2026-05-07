import sqlite3

conn = sqlite3.connect('C:/Users/YUXUAN/Desktop/ATrader/aquant.db')
cursor = conn.cursor()

tables = ['stock_prices', 'money_flow', 'sentiment_records', 'stocks', 'index_prices']
for table in tables:
    print(f"\n=== {table} ===")
    cursor.execute(f"PRAGMA table_info({table})")
    for row in cursor.fetchall():
        print(f"  {row[1]} ({row[2]})")
    
    # 查看样本数据
    cursor.execute(f"SELECT * FROM {table} LIMIT 2")
    rows = cursor.fetchall()
    if rows:
        print(f"  Sample: {rows[0]}")

conn.close()
