"""
实盘数据接入模块
支持多源实时数据流，包括WebSocket和REST API
"""
import asyncio
import aiohttp
import websockets
import json
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque
import threading
import time
from loguru import logger


@dataclass
class LiveTick:
    """实时 tick 数据"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float
    ask: float
    bid_volume: int
    ask_volume: int
    
    @property
    def spread(self) -> float:
        """买卖价差"""
        return self.ask - self.bid


@dataclass
class LiveBar:
    """实时 K 线数据"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    period: str  # '1m', '5m', '1d', etc.


class LiveDataConnector(ABC):
    """实盘数据连接器基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_connected = False
        self.subscriptions: set = set()
        self.callbacks: List[Callable] = []
        self._buffer: deque = deque(maxlen=10000)
        self._task: Optional[asyncio.Task] = None
        
    @abstractmethod
    async def connect(self):
        """建立连接"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]):
        """订阅标的"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        pass
    
    def add_callback(self, callback: Callable):
        """添加数据回调"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """移除数据回调"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _on_data(self, data: Any):
        """内部数据分发"""
        self._buffer.append((datetime.now(), data))
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")


class EastmoneyLiveConnector(LiveDataConnector):
    """
    东方财富实时数据连接器
    支持：个股行情、资金流向、板块异动
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("eastmoney", config or {})
        self.ws_url = "wss://hq.push2.eastmoney.com/ws"
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
    async def connect(self):
        """建立WebSocket连接"""
        try:
            logger.info("Connecting to Eastmoney WebSocket...")
            self.ws = await websockets.connect(self.ws_url)
            self.is_connected = True
            
            # 启动心跳
            self._heartbeat_task = asyncio.create_task(self._heartbeat())
            
            # 启动接收循环
            self._task = asyncio.create_task(self._receive_loop())
            
            logger.info("Eastmoney connected successfully")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.is_connected = False
            raise
    
    async def disconnect(self):
        """断开连接"""
        self.is_connected = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        
        if self._task:
            self._task.cancel()
        
        if self.ws:
            await self.ws.close()
        
        logger.info("Eastmoney disconnected")
    
    async def subscribe(self, symbols: List[str]):
        """订阅股票代码"""
        if not self.is_connected:
            raise RuntimeError("Not connected")
        
        # 东方财富格式: 0.000001 (深市), 1.600000 (沪市)
        formatted_symbols = []
        for s in symbols:
            if s.startswith('6'):
                formatted_symbols.append(f"1.{s}")
            else:
                formatted_symbols.append(f"0.{s}")
        
        msg = {
            "cmd": "subscribe",
            "codes": formatted_symbols
        }
        
        await self.ws.send(json.dumps(msg))
        self.subscriptions.update(symbols)
        logger.info(f"Subscribed to {len(symbols)} symbols")
    
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        if not self.is_connected:
            return
        
        msg = {
            "cmd": "unsubscribe",
            "codes": symbols
        }
        
        await self.ws.send(json.dumps(msg))
        self.subscriptions.difference_update(symbols)
    
    async def _heartbeat(self):
        """心跳保活"""
        while self.is_connected:
            try:
                if self.ws and self.ws.open:
                    await self.ws.send('{"cmd":"ping"}')
                await asyncio.sleep(30)
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")
                break
    
    async def _receive_loop(self):
        """接收消息循环"""
        while self.is_connected:
            try:
                if not self.ws:
                    await asyncio.sleep(1)
                    continue
                
                message = await asyncio.wait_for(self.ws.recv(), timeout=60)
                data = json.loads(message)
                
                # 解析tick数据
                tick = self._parse_tick(data)
                if tick:
                    self._on_data(tick)
                    
            except asyncio.TimeoutError:
                logger.warning("Receive timeout, reconnecting...")
                await self._reconnect()
            except Exception as e:
                logger.error(f"Receive error: {e}")
                await asyncio.sleep(1)
    
    def _parse_tick(self, data: Dict) -> Optional[LiveTick]:
        """解析tick数据"""
        try:
            code = data.get('code', '')
            if not code:
                return None
            
            # 提取价格信息
            price = float(data.get('price', 0))
            volume = int(data.get('volume', 0))
            
            return LiveTick(
                symbol=code,
                timestamp=datetime.now(),
                price=price,
                volume=volume,
                bid=float(data.get('bid1', price * 0.999)),
                ask=float(data.get('ask1', price * 1.001)),
                bid_volume=int(data.get('bid1_volume', 0)),
                ask_volume=int(data.get('ask1_volume', 0))
            )
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return None
    
    async def _reconnect(self):
        """自动重连"""
        logger.info("Reconnecting...")
        await self.disconnect()
        await asyncio.sleep(5)
        await self.connect()
        
        # 恢复订阅
        if self.subscriptions:
            await self.subscribe(list(self.subscriptions))
    
    async def fetch_money_flow_realtime(self, stock_code: str) -> Dict[str, float]:
        """
        获取实时资金流向（REST API备用）
        """
        url = f"https://push2.eastmoney.com/api/qt/stock/get"
        params = {
            "secid": f"0.{stock_code}" if not stock_code.startswith('6') else f"1.{stock_code}",
            "fields": "f43,f44,f45,f46,f47,f48,f57,f58,f60"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # 解析字段
                    result = data.get('data', {})
                    return {
                        'price': result.get('f43', 0) / 100,
                        'change_percent': result.get('f44', 0) / 100,
                        'volume': result.get('f45', 0),
                        'amount': result.get('f46', 0),
                        'main_inflow': result.get('f47', 0),
                        'retail_inflow': result.get('f48', 0)
                    }
                else:
                    raise Exception(f"HTTP {resp.status}")


class XueqiuLiveConnector(LiveDataConnector):
    """
    雪球实时数据连接器
    支持：帖子流、情感分析、热股排行
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("xueqiu", config or {})
        self.base_url = "https://stock.xueqiu.com/v5/stock"
        self.token = config.get('token', '') if config else ''
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def connect(self):
        """建立会话"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Cookie': f'xq_a_token={self.token}' if self.token else ''
        }
        self._session = aiohttp.ClientSession(headers=headers)
        self.is_connected = True
        logger.info("Xueqiu session established")
    
    async def disconnect(self):
        """断开会话"""
        if self._session:
            await self._session.close()
        self.is_connected = False
        logger.info("Xueqiu disconnected")
    
    async def subscribe(self, symbols: List[str]):
        """订阅股票"""
        self.subscriptions.update(symbols)
        # 雪球通过轮询获取，不需要WebSocket订阅
        self._task = asyncio.create_task(self._poll_loop(symbols))
    
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        self.subscriptions.difference_update(symbols)
    
    async def _poll_loop(self, symbols: List[str]):
        """轮询循环"""
        while self.is_connected and self.subscriptions:
            try:
                for symbol in list(self.subscriptions):
                    data = await self._fetch_quote(symbol)
                    if data:
                        self._on_data(data)
                
                await asyncio.sleep(5)  # 5秒轮询间隔
            except Exception as e:
                logger.error(f"Poll error: {e}")
                await asyncio.sleep(10)
    
    async def _fetch_quote(self, symbol: str) -> Optional[Dict]:
        """获取行情数据"""
        if not self._session:
            return None
        
        url = f"{self.base_url}/app/stock/quotepage.json"
        params = {"symbol": symbol}
        
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    quote = data.get('data', {}).get('quote', {})
                    return {
                        'symbol': symbol,
                        'price': quote.get('current', 0),
                        'change_percent': quote.get('percent', 0),
                        'volume': quote.get('volume', 0),
                        'market_cap': quote.get('market_capital', 0),
                        'source': 'xueqiu'
                    }
        except Exception as e:
            logger.error(f"Fetch error for {symbol}: {e}")
        
        return None
    
    async def fetch_hot_posts(self, stock_code: str, count: int = 20) -> pd.DataFrame:
        """
        获取热门帖子
        """
        if not self._session:
            raise RuntimeError("Not connected")
        
        url = f"{self.base_url}/app/stock/quotepage.json"
        params = {
            "symbol": stock_code,
            "count": count
        }
        
        async with self._session.get(url, params=params) as resp:
            data = await resp.json()
            posts = data.get('data', {}).get('posts', [])
            
            records = []
            for post in posts:
                records.append({
                    'id': post.get('id'),
                    'title': post.get('title', ''),
                    'content': post.get('text', '')[:200],
                    'author': post.get('user', {}).get('screen_name', ''),
                    'likes': post.get('like_count', 0),
                    'comments': post.get('reply_count', 0),
                    'timestamp': post.get('created_at', '')
                })
            
            return pd.DataFrame(records)
    
    async def fetch_sentiment_realtime(self, stock_code: str) -> Dict[str, float]:
        """
        获取实时情感指标
        """
        posts = await self.fetch_hot_posts(stock_code, count=50)
        
        if posts.empty:
            return {'bull_ratio': 0.5, 'bear_ratio': 0.5, 'neutral_ratio': 0.0}
        
        # 简单情感分析（基于关键词）
        bullish_words = ['涨', '牛市', '看好', '买入', '抄底']
        bearish_words = ['跌', '熊市', '看空', '卖出', '割肉']
        
        bull_count = 0
        bear_count = 0
        
        for _, post in posts.iterrows():
            text = str(post['title']) + str(post['content'])
            if any(w in text for w in bullish_words):
                bull_count += 1
            elif any(w in text for w in bearish_words):
                bear_count += 1
        
        total = len(posts)
        return {
            'bull_ratio': bull_count / total,
            'bear_ratio': bear_count / total,
            'neutral_ratio': (total - bull_count - bear_count) / total
        }


class LiveDataManager:
    """
    统一实时数据管理器
    管理多个连接器，提供统一接口
    """
    
    def __init__(self):
        self.connectors: Dict[str, LiveDataConnector] = {}
        self._aggregator: Optional[Callable] = None
        self._running = False
        
    def register_connector(self, connector: LiveDataConnector):
        """注册连接器"""
        self.connectors[connector.name] = connector
        logger.info(f"Registered connector: {connector.name}")
    
    async def start_all(self):
        """启动所有连接器"""
        self._running = True
        tasks = []
        
        for name, connector in self.connectors.items():
            try:
                await connector.connect()
                logger.info(f"Started: {name}")
            except Exception as e:
                logger.error(f"Failed to start {name}: {e}")
        
        return tasks
    
    async def stop_all(self):
        """停止所有连接器"""
        self._running = False
        
        for name, connector in self.connectors.items():
            try:
                await connector.disconnect()
                logger.info(f"Stopped: {name}")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
    
    async def subscribe(self, symbols: List[str], sources: Optional[List[str]] = None):
        """
        订阅标的
        
        Args:
            symbols: 股票代码列表
            sources: 指定数据源，None表示所有
        """
        targets = sources or list(self.connectors.keys())
        
        for source in targets:
            if source in self.connectors:
                connector = self.connectors[source]
                if connector.is_connected:
                    await connector.subscribe(symbols)
    
    def add_global_callback(self, callback: Callable):
        """添加全局数据回调"""
        for connector in self.connectors.values():
            connector.add_callback(callback)
    
    def get_buffer(self, source: str) -> pd.DataFrame:
        """获取数据缓冲区"""
        if source not in self.connectors:
            return pd.DataFrame()
        
        buffer = self.connectors[source]._buffer
        if not buffer:
            return pd.DataFrame()
        
        records = []
        for ts, data in buffer:
            if isinstance(data, LiveTick):
                records.append({
                    'timestamp': ts,
                    'symbol': data.symbol,
                    'price': data.price,
                    'volume': data.volume,
                    'bid': data.bid,
                    'ask': data.ask
                })
        
        return pd.DataFrame(records)
    
    def get_latest_price(self, symbol: str, source: str = 'eastmoney') -> Optional[float]:
        """获取最新价格"""
        df = self.get_buffer(source)
        if df.empty:
            return None
        
        mask = df['symbol'] == symbol
        if mask.any():
            return df[mask].iloc[-1]['price']
        
        return None
    
    def get_spread(self, symbol: str, source: str = 'eastmoney') -> Optional[Dict[str, float]]:
        """获取买卖价差"""
        df = self.get_buffer(source)
        if df.empty:
            return None
        
        mask = df['symbol'] == symbol
        if mask.any():
            latest = df[mask].iloc[-1]
            return {
                'bid': latest['bid'],
                'ask': latest['ask'],
                'spread': latest['ask'] - latest['bid'],
                'spread_pct': (latest['ask'] - latest['bid']) / latest['price'] * 100
            }
        
        return None


# 便捷函数
async def create_live_pipeline(config: Dict[str, Any]) -> LiveDataManager:
    """
    创建实时数据管道
    
    Example:
        config = {
            'eastmoney': {'enabled': True},
            'xueqiu': {'enabled': True, 'token': 'xxx'}
        }
    """
    manager = LiveDataManager()
    
    if config.get('eastmoney', {}).get('enabled', False):
        manager.register_connector(EastmoneyLiveConnector(config['eastmoney']))
    
    if config.get('xueqiu', {}).get('enabled', False):
        manager.register_connector(XueqiuLiveConnector(config['xueqiu']))
    
    await manager.start_all()
    return manager
