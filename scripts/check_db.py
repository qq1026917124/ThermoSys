import sqlite3
conn = sqlite3.connect('C:/Users/YUXUAN/Desktop/ATrader/aquant.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
print("=== aquant.db tables ===")
for row in cursor.fetchall():
    print(row[0])
conn.close()

print()
conn2 = sqlite3.connect('C:/Users/YUXUAN/Desktop/ATrader/macro.db')
cursor2 = conn2.cursor()
cursor2.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
print("=== macro.db tables ===")
for row in cursor2.fetchall():
    print(row[0])
conn2.close()
