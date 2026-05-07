"""
ATrader 数据适配器
从 ATrader 项目的历史数据和数据库中读取真实A股数据
"""
import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta


class ATraderDataAdapter:
    """
    ATrader 数据适配器
    
    数据源：
    1. CSV文件: C:/Users/YUXUAN/Desktop/ATrader/data/history/stocks/*.csv
    2. SQLite数据库: C:/Users/YUXUAN/Desktop/ATrader/aquant.db
    3. SQLite数据库: C:/Users/YUXUAN/Desktop/ATrader/macro.db
    """
    
    def __init__(self, atrader_root: str = r"C:\Users\YUXUAN\Desktop\ATrader"):
        self.atrader_root = Path(atrader_root)
        self.history_dir = self.atrader_root / "data" / "history"
        self.db_path = self.atrader_root / "aquant.db"
        self.macro_db_path = self.atrader_root / "macro.db"
        
        # 缓存
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._db_conn: Optional[sqlite3.Connection] = None
        
    def _get_db_connection(self) -> sqlite3.Connection:
        """获取数据库连接（懒加载）"""
        if self._db_conn is None:
            self._db_conn = sqlite3.connect(str(self.db_path))
        return self._db_conn
    
    def load_stock_price_from_csv(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        从CSV加载个股历史日线数据
        
        Args:
            symbol: 股票代码，如 '000001'
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            
        Returns:
            DataFrame with columns: [trade_date, open, high, low, close, volume, amount]
        """
        if symbol in self._price_cache:
            df = self._price_cache[symbol].copy()
        else:
            csv_path = self.history_dir / "stocks" / f"{symbol}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Stock data not found: {csv_path}")
            
            df = pd.read_csv(csv_path)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').set_index('trade_date')
            self._price_cache[symbol] = df
        
        # 日期过滤
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        return df.copy()
    
    def load_stock_price_from_db(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        从SQLite数据库加载个股历史数据
        
        Returns:
            DataFrame with columns: [trade_date, open, high, low, close, volume, amount, turnover, ma5, ma20, ma60]
        """
        conn = self._get_db_connection()
        
        query = f"""
        SELECT trade_date, open, high, low, close, volume, amount, turnover, ma5, ma20, ma60
        FROM stock_prices
        WHERE symbol = '{symbol}'
        """
        
        if start_date:
            query += f" AND trade_date >= '{start_date}'"
        if end_date:
            query += f" AND trade_date <= '{end_date}'"
        
        query += " ORDER BY trade_date"
        
        df = pd.read_sql_query(query, conn)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date')
        
        return df
    
    def load_money_flow(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        加载资金流向数据
        
        Returns:
            DataFrame with columns: [trade_date, main_inflow, main_outflow, main_net, main_ratio, retail_inflow, retail_outflow, retail_net]
        """
        conn = self._get_db_connection()
        
        query = f"""
        SELECT trade_date, main_inflow, main_outflow, main_net, main_ratio,
               retail_inflow, retail_outflow, retail_net
        FROM money_flow
        WHERE symbol = '{symbol}'
        """
        
        if start_date:
            query += f" AND trade_date >= '{start_date}'"
        if end_date:
            query += f" AND trade_date <= '{end_date}'"
        
        query += " ORDER BY trade_date"
        
        df = pd.read_sql_query(query, conn)
        if df.empty:
            return pd.DataFrame()
        
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date')
        
        return df
    
    def load_sentiment(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        加载情感数据
        
        Returns:
            DataFrame with columns: [record_date, sentiment_score, sentiment_label, heat_score, news_count]
        """
        conn = self._get_db_connection()
        
        query = """
        SELECT record_date, sentiment_score, sentiment_label, heat_score, news_count
        FROM sentiment_records
        WHERE 1=1
        """
        
        if symbol:
            query += f" AND symbol = '{symbol}'"
        if start_date:
            query += f" AND record_date >= '{start_date}'"
        if end_date:
            query += f" AND record_date <= '{end_date}'"
        
        query += " ORDER BY record_date"
        
        df = pd.read_sql_query(query, conn)
        if df.empty:
            return pd.DataFrame()
        
        df['record_date'] = pd.to_datetime(df['record_date'])
        df = df.set_index('record_date')
        
        return df
    
    def load_index_price(
        self,
        symbol: str = '000001',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        加载指数数据（默认上证指数）
        
        Returns:
            DataFrame with columns: [trade_date, open, high, low, close, volume, amount]
        """
        conn = self._get_db_connection()
        
        query = f"""
        SELECT trade_date, open, high, low, close, volume, amount
        FROM index_prices
        WHERE symbol = '{symbol}'
        """
        
        if start_date:
            query += f" AND trade_date >= '{start_date}'"
        if end_date:
            query += f" AND trade_date <= '{end_date}'"
        
        query += " ORDER BY trade_date"
        
        df = pd.read_sql_query(query, conn)
        if df.empty:
            return pd.DataFrame()
        
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date')
        
        return df
    
    def load_all_symbols(self) -> List[str]:
        """获取所有可用的股票代码"""
        csv_dir = self.history_dir / "stocks"
        symbols = [f.stem for f in csv_dir.glob("*.csv")]
        return sorted(symbols)
    
    def get_data_availability(self) -> Dict[str, any]:
        """检查数据可用性"""
        result = {
            'csv_stocks': len(list((self.history_dir / "stocks").glob("*.csv"))),
            'db_path_exists': self.db_path.exists(),
            'macro_db_exists': self.macro_db_path.exists(),
            'db_tables': []
        }
        
        if self.db_path.exists():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            result['db_tables'] = [row[0] for row in cursor.fetchall()]
            
            # 统计各表记录数
            for table in ['stock_prices', 'money_flow', 'sentiment_records', 'index_prices']:
                if table in result['db_tables']:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    result[f'{table}_count'] = cursor.fetchone()[0]
            
            conn.close()
        
        return result
    
    def build_thermo_dataset(
        self,
        symbol: str = '000001',
        index_symbol: str = '000001',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        构建完整的热力学分析数据集
        
        Returns:
            {
                'price': 股票价格,
                'index': 指数价格,
                'money_flow': 资金流向,
                'sentiment': 情感数据
            }
        """
        print(f"[ATrader] 加载 {symbol} 数据...")
        
        # 价格数据（优先从CSV读取，数据更完整）
        try:
            price = self.load_stock_price_from_csv(symbol, start_date, end_date)
            print(f"  价格数据: {len(price)} 条 (CSV)")
        except FileNotFoundError:
            price = self.load_stock_price_from_db(symbol, start_date, end_date)
            print(f"  价格数据: {len(price)} 条 (DB)")
        
        # 指数数据
        index_price = self.load_index_price(index_symbol, start_date, end_date)
        print(f"  指数数据: {len(index_price)} 条")
        
        # 资金流向
        money_flow = self.load_money_flow(symbol, start_date, end_date)
        print(f"  资金流向: {len(money_flow)} 条")
        
        # 情感数据
        sentiment = self.load_sentiment(symbol, start_date, end_date)
        print(f"  情感数据: {len(sentiment)} 条")
        
        return {
            'price': price,
            'index': index_price,
            'money_flow': money_flow,
            'sentiment': sentiment
        }
    
    def close(self):
        """关闭数据库连接"""
        if self._db_conn:
            self._db_conn.close()
            self._db_conn = None


class MacroDataAdapter:
    """
    宏观数据适配器（从 macro.db 读取）
    """
    
    def __init__(self, macro_db_path: str = r"C:\Users\YUXUAN\Desktop\ATrader\macro.db"):
        self.macro_db_path = Path(macro_db_path)
        self._conn: Optional[sqlite3.Connection] = None
    
    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.macro_db_path))
        return self._conn
    
    def load_macro_events(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """加载宏观事件数据"""
        conn = self._get_connection()
        
        query = """
        SELECT event_date, event_type, event_title, sentiment_score, impact_level, sector_tags
        FROM macro_events
        WHERE 1=1
        """
        
        if start_date:
            query += f" AND event_date >= '{start_date}'"
        if end_date:
            query += f" AND event_date <= '{end_date}'"
        
        query += " ORDER BY event_date"
        
        df = pd.read_sql_query(query, conn)
        if not df.empty:
            df['event_date'] = pd.to_datetime(df['event_date'])
            df = df.set_index('event_date')
        
        return df
    
    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
