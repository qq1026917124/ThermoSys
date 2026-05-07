"""
数据抓取模块
支持多源异构数据的异步抓取
"""
import asyncio
import aiohttp
import requests
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json


class DataCollector(ABC):
    """数据抓取器基类"""
    
    def __init__(self, base_url: str, rate_limit: float = 1.0):
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def fetch(self, **kwargs) -> pd.DataFrame:
        """抓取数据"""
        pass
    
    async def _get(self, endpoint: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Any:
        """异步GET请求"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}/{endpoint}"
        async with self.session.get(url, params=params, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"HTTP {response.status}: {await response.text()}")


class EastmoneyCollector(DataCollector):
    """
    东方财富数据抓取器
    抓取：资金流向、融资余额、散户开户数等
    """
    
    def __init__(self):
        super().__init__("https://push2.eastmoney.com/api", rate_limit=10)
        
    async def fetch_money_flow(
        self,
        stock_code: str,
        market_type: str = '1',  # 1=沪深
        days: int = 60
    ) -> pd.DataFrame:
        """
        抓取个股资金流向
        
        Returns:
            DataFrame with columns: [date, main_inflow, small_inflow, ...]
        """
        # 这里使用模拟数据，实际应调用东方财富API
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        
        data = {
            'date': dates,
            'main_inflow': np.random.randn(days) * 10000,
            'small_inflow': np.random.randn(days) * 5000,  # 散户资金
            'medium_inflow': np.random.randn(days) * 8000,
            'large_inflow': np.random.randn(days) * 12000,
        }
        
        return pd.DataFrame(data).set_index('date')
    
    async def fetch_margin_balance(self, days: int = 60) -> pd.Series:
        """抓取融资余额"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        
        # 模拟融资余额数据（缓慢增长+波动）
        base = 15000  # 亿元
        trend = np.linspace(0, np.random.randn() * 1000, days)
        noise = np.random.randn(days) * 200
        values = base + trend + noise
        
        return pd.Series(values, index=dates, name='margin_balance')
    
    async def fetch_new_accounts(self, days: int = 60) -> pd.Series:
        """抓取新增开户数"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        
        # 模拟开户数（与市场情绪相关）
        base = 30000
        cycle = np.sin(np.linspace(0, 4*np.pi, days)) * 10000
        noise = np.random.randn(days) * 5000
        values = base + cycle + noise
        values = np.maximum(values, 10000)  # 最小值
        
        return pd.Series(values, index=dates, name='new_accounts')
    
    async def fetch_option_pcr(self, days: int = 60) -> pd.Series:
        """抓取期权PCR（Put/Call Ratio）"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        
        # PCR通常在0.5-1.5之间
        base = 0.9
        noise = np.random.randn(days) * 0.15
        values = base + noise
        values = np.clip(values, 0.4, 1.6)
        
        return pd.Series(values, index=dates, name='option_pcr')


class XueqiuCollector(DataCollector):
    """
    雪球数据抓取器
    抓取：帖子量、评论情感、互动率等
    """
    
    def __init__(self):
        super().__init__("https://stock.xueqiu.com/v5/stock", rate_limit=5)
        
    async def fetch_posts(self, stock_code: str, days: int = 60) -> pd.DataFrame:
        """
        抓取雪球帖子数据
        
        Returns:
            DataFrame with columns: [date, post_count, comment_count, like_count, ...]
        """
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        
        # 模拟帖子数据
        base_posts = 500
        base_comments = 2000
        base_likes = 5000
        
        data = {
            'date': dates,
            'post_count': np.maximum(base_posts + np.random.randn(days) * 200, 50),
            'comment_count': np.maximum(base_comments + np.random.randn(days) * 800, 200),
            'like_count': np.maximum(base_likes + np.random.randn(days) * 2000, 500),
            'share_count': np.maximum(100 + np.random.randn(days) * 50, 10),
        }
        
        return pd.DataFrame(data).set_index('date')
    
    async def fetch_sentiment_distribution(self, days: int = 60) -> pd.DataFrame:
        """
        抓取情感分布
        
        Returns:
            DataFrame with columns: [extreme_bear, bear, neutral, bull, extreme_bull]
        """
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        n = len(dates)
        
        # 生成合理的概率分布
        distributions = []
        for _ in range(n):
            raw = np.random.dirichlet([1, 2, 4, 2, 1])  # 中性偏多
            distributions.append(raw)
        
        df = pd.DataFrame(
            distributions,
            index=dates,
            columns=['extreme_bear', 'bear', 'neutral', 'bull', 'extreme_bull']
        )
        
        return df


class BaiduIndexCollector(DataCollector):
    """
    百度指数抓取器
    抓取：关键词搜索量
    """
    
    def __init__(self):
        super().__init__("", rate_limit=3)
        self.keywords = ['股票开户', '牛市', '涨停', '割肉', '跌停', '清仓']
        
    async def fetch_search_index(self, days: int = 60) -> pd.DataFrame:
        """
        抓取搜索指数
        
        Returns:
            DataFrame with columns per keyword
        """
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        data = {'date': dates}
        for keyword in self.keywords:
            base = np.random.randint(1000, 5000)
            trend = np.sin(np.linspace(0, 2*np.pi, days)) * base * 0.3
            noise = np.random.randn(days) * base * 0.1
            data[keyword] = np.maximum(base + trend + noise, 100)
        
        return pd.DataFrame(data).set_index('date')


class UnifiedDataPipeline:
    """
    统一数据管道
    协调多个数据源的抓取、对齐和清洗
    """
    
    def __init__(self):
        self.collectors = {
            'eastmoney': EastmoneyCollector(),
            'xueqiu': XueqiuCollector(),
            'baidu': BaiduIndexCollector()
        }
        
    async def fetch_all(self, stock_code: str = '000001', days: int = 60) -> Dict[str, pd.DataFrame]:
        """
        抓取所有数据源的数据
        
        Returns:
            各数据源的数据字典
        """
        results = {}
        
        async with aiohttp.ClientSession() as session:
            # 东方财富
            em = self.collectors['eastmoney']
            em.session = session
            results['money_flow'] = await em.fetch_money_flow(stock_code, days=days)
            results['margin_balance'] = await em.fetch_margin_balance(days=days)
            results['new_accounts'] = await em.fetch_new_accounts(days=days)
            results['option_pcr'] = await em.fetch_option_pcr(days=days)
            
            # 雪球
            xq = self.collectors['xueqiu']
            xq.session = session
            results['posts'] = await xq.fetch_posts(stock_code, days=days)
            results['sentiment'] = await xq.fetch_sentiment_distribution(days=days)
            
            # 百度
            bd = self.collectors['baidu']
            bd.session = session
            results['search_index'] = await bd.fetch_search_index(days=days)
        
        return results
    
    def align_and_clean(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        对齐所有数据到统一时间轴
        
        Returns:
            合并后的DataFrame
        """
        from thermo_sys.utils.data_utils import time_align
        
        # 转换为日频
        daily_data = {}
        for name, df in data_dict.items():
            if isinstance(df, pd.Series):
                df = df.to_frame()
            
            # 重采样为日频（取最后值）
            if df.index.freq != 'D':
                df = df.resample('D').last()
            
            # 重命名列避免冲突
            df = df.rename(columns=lambda x: f"{name}_{x}" if x != 'date' else x)
            daily_data[name] = df
        
        # 合并
        merged = pd.concat(daily_data.values(), axis=1)
        merged = merged.fillna(method='ffill').dropna()
        
        return merged
