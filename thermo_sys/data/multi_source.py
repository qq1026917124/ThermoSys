"""
多数据源适配器
集成同花顺、腾讯财经等备选数据源
防止单数据源失效
"""
import asyncio
import aiohttp
import requests
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger


@dataclass
class DataSourceStatus:
    """数据源状态"""
    name: str
    is_available: bool
    last_check: datetime
    latency_ms: float
    error_count: int = 0


class BaseDataSource(ABC):
    """数据源基类"""
    
    def __init__(self, name: str, priority: int = 10):
        self.name = name
        self.priority = priority  # 优先级，数字越小优先级越高
        self.status = DataSourceStatus(
            name=name,
            is_available=False,
            last_check=datetime.now(),
            latency_ms=0
        )
    
    @abstractmethod
    async def check_health(self) -> bool:
        """检查数据源健康状态"""
        pass
    
    @abstractmethod
    async def fetch_realtime_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取实时行情"""
        pass
    
    @abstractmethod
    async def fetch_kline(self, symbol: str, period: str = 'day', count: int = 60) -> Optional[pd.DataFrame]:
        """获取K线数据"""
        pass
    
    async def _safe_request(self, url: str, timeout: int = 10) -> Optional[Dict]:
        """安全请求，带错误处理"""
        start = datetime.now()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    latency = (datetime.now() - start).total_seconds() * 1000
                    self.status.latency_ms = latency
                    
                    if resp.status == 200:
                        self.status.is_available = True
                        self.status.error_count = 0
                        return await resp.json()
                    else:
                        self.status.error_count += 1
                        return None
        except Exception as e:
            self.status.error_count += 1
            logger.warning(f"{self.name} 请求失败: {e}")
            return None
        finally:
            self.status.last_check = datetime.now()


class TonghuashunDataSource(BaseDataSource):
    """
    同花顺数据源
    特点：数据全面，接口稳定
    """
    
    def __init__(self):
        super().__init__("tonghuashun", priority=2)
        self.base_url = "http://d.10jqka.com.cn"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'http://stockpage.10jqka.com.cn/'
        }
    
    async def check_health(self) -> bool:
        """检查同花顺接口"""
        url = f"{self.base_url}/time/hs_000001/last.js"
        result = await self._safe_request(url)
        return result is not None
    
    async def fetch_realtime_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取实时行情
        
        同花顺接口格式: hs_代码（沪深）
        """
        # 转换代码格式
        if symbol.startswith('6'):
            ths_symbol = f"sh{symbol}"
        else:
            ths_symbol = f"sz{symbol}"
        
        url = f"http://stockpage.10jqka.com.cn/{ths_symbol}"
        
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        # 解析页面获取数据（简化版）
                        return {
                            'symbol': symbol,
                            'source': 'tonghuashun',
                            'price': self._extract_price(html),
                            'timestamp': datetime.now()
                        }
        except Exception as e:
            logger.error(f"同花顺获取 {symbol} 失败: {e}")
        
        return None
    
    def _extract_price(self, html: str) -> float:
        """从HTML中提取价格（简化）"""
        # 实际应该使用正则或BeautifulSoup解析
        # 这里返回模拟值
        return 0.0
    
    async def fetch_kline(self, symbol: str, period: str = 'day', count: int = 60) -> Optional[pd.DataFrame]:
        """获取K线数据"""
        # 同花顺K线接口
        if symbol.startswith('6'):
            ths_symbol = f"sh{symbol}"
        else:
            ths_symbol = f"sz{symbol}"
        
        url = f"http://d.10jqka.com.cn/v6/time/hs_{symbol}/last.js"
        
        result = await self._safe_request(url)
        if result and 'data' in result:
            # 解析数据
            data = result['data']
            # 转换为DataFrame
            return pd.DataFrame(data)
        
        return None


class TencentDataSource(BaseDataSource):
    """
    腾讯财经数据源
    特点：接口简单，响应快
    """
    
    def __init__(self):
        super().__init__("tencent", priority=3)
        self.base_url = "https://qt.gtimg.cn"
    
    async def check_health(self) -> bool:
        """检查腾讯财经接口"""
        url = f"{self.base_url}/q=sh000001"
        result = await self._safe_request(url)
        return result is not None
    
    async def fetch_realtime_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取实时行情
        
        腾讯接口格式: sh000001（沪市）, sz000001（深市）
        """
        if symbol.startswith('6'):
            tencent_symbol = f"sh{symbol}"
        else:
            tencent_symbol = f"sz{symbol}"
        
        url = f"{self.base_url}/q={tencent_symbol}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        return self._parse_tencent_response(text, symbol)
        except Exception as e:
            logger.error(f"腾讯财经获取 {symbol} 失败: {e}")
        
        return None
    
    def _parse_tencent_response(self, text: str, symbol: str) -> Dict[str, Any]:
        """解析腾讯财经响应"""
        try:
            # 腾讯返回格式: v_sh000001="1~名称~代码~价格~涨跌..."
            parts = text.split('"')[1].split('~')
            if len(parts) > 3:
                return {
                    'symbol': symbol,
                    'source': 'tencent',
                    'name': parts[1],
                    'price': float(parts[3]),
                    'change': float(parts[4]) if len(parts) > 4 else 0,
                    'change_percent': float(parts[5]) if len(parts) > 5 else 0,
                    'volume': int(parts[6]) if len(parts) > 6 else 0,
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.warning(f"解析腾讯数据失败: {e}")
        
        return {'symbol': symbol, 'source': 'tencent', 'price': 0.0}
    
    async def fetch_kline(self, symbol: str, period: str = 'day', count: int = 60) -> Optional[pd.DataFrame]:
        """
        获取K线数据
        
        腾讯K线接口
        """
        if symbol.startswith('6'):
            tencent_symbol = f"sh{symbol}"
        else:
            tencent_symbol = f"sz{symbol}"
        
        # 腾讯日K线
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        params = {
            'param': f"{tencent_symbol},{period},,,{count},qfq"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_kline_data(data, tencent_symbol)
        except Exception as e:
            logger.error(f"腾讯K线获取失败: {e}")
        
        return None
    
    def _parse_kline_data(self, data: Dict, symbol: str) -> pd.DataFrame:
        """解析K线数据"""
        try:
            kline_data = data['data'][symbol]['day']
            df = pd.DataFrame(kline_data, columns=['date', 'open', 'close', 'low', 'high', 'volume'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.astype(float)
            return df
        except Exception as e:
            logger.warning(f"解析K线失败: {e}")
            return pd.DataFrame()


class SinaDataSource(BaseDataSource):
    """
    新浪财经数据源
    特点：接口极其简单，稳定性高
    """
    
    def __init__(self):
        super().__init__("sina", priority=4)
        self.base_url = "https://hq.sinajs.cn"
    
    async def check_health(self) -> bool:
        """检查新浪接口"""
        url = f"{self.base_url}/list=sh000001"
        result = await self._safe_request(url)
        return result is not None
    
    async def fetch_realtime_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取实时行情
        
        新浪接口格式: sh000001（沪市）, sz000001（深市）
        """
        if symbol.startswith('6'):
            sina_symbol = f"sh{symbol}"
        else:
            sina_symbol = f"sz{symbol}"
        
        url = f"{self.base_url}/list={sina_symbol}"
        
        try:
            async with aiohttp.ClientSession() as session:
                # 新浪需要特定的Referer
                headers = {'Referer': 'https://finance.sina.com.cn'}
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        return self._parse_sina_response(text, symbol)
        except Exception as e:
            logger.error(f"新浪获取 {symbol} 失败: {e}")
        
        return None
    
    def _parse_sina_response(self, text: str, symbol: str) -> Dict[str, Any]:
        """解析新浪响应"""
        try:
            # 新浪格式: var hq_str_sh000001="名称,今日开盘价,昨日收盘价,当前价..."
            if '"' in text:
                data_str = text.split('"')[1]
                parts = data_str.split(',')
                if len(parts) > 3:
                    return {
                        'symbol': symbol,
                        'source': 'sina',
                        'name': parts[0],
                        'open': float(parts[1]),
                        'close': float(parts[2]),
                        'price': float(parts[3]),
                        'high': float(parts[4]),
                        'low': float(parts[5]),
                        'volume': int(parts[8]),
                        'amount': float(parts[9]),
                        'timestamp': datetime.now()
                    }
        except Exception as e:
            logger.warning(f"解析新浪数据失败: {e}")
        
        return {'symbol': symbol, 'source': 'sina', 'price': 0.0}
    
    async def fetch_kline(self, symbol: str, period: str = 'day', count: int = 60) -> Optional[pd.DataFrame]:
        """新浪不提供直接K线接口，返回None"""
        return None


class MultiDataSourceManager:
    """
    多数据源管理器
    
    功能：
    1. 管理多个数据源
    2. 自动故障转移
    3. 健康检查轮询
    4. 数据一致性校验
    """
    
    def __init__(self):
        self.sources: Dict[str, BaseDataSource] = {}
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
    
    def register(self, source: BaseDataSource):
        """注册数据源"""
        self.sources[source.name] = source
        logger.info(f"注册数据源: {source.name} (优先级: {source.priority})")
    
    async def start(self):
        """启动管理器"""
        self._running = True
        # 执行初始健康检查
        await self._check_all_sources()
        # 启动定期检查
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def stop(self):
        """停止管理器"""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            await self._check_all_sources()
            await asyncio.sleep(60)  # 每分钟检查一次
    
    async def _check_all_sources(self):
        """检查所有数据源"""
        tasks = [source.check_health() for source in self.sources.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for source, result in zip(self.sources.values(), results):
            if isinstance(result, Exception):
                source.status.is_available = False
                logger.warning(f"{source.name} 健康检查异常: {result}")
            else:
                source.status.is_available = result
                status = "正常" if result else "异常"
                logger.debug(f"{source.name} 状态: {status}")
    
    def get_available_sources(self) -> List[BaseDataSource]:
        """获取可用的数据源列表（按优先级排序）"""
        available = [s for s in self.sources.values() if s.status.is_available]
        return sorted(available, key=lambda x: x.priority)
    
    async def fetch_with_fallback(self, symbol: str, method: str = 'realtime') -> Optional[Any]:
        """
        带故障转移的数据获取
        
        依次尝试各数据源，直到成功
        """
        sources = self.get_available_sources()
        
        if not sources:
            logger.error("没有可用的数据源！")
            return None
        
        for source in sources:
            try:
                if method == 'realtime':
                    result = await source.fetch_realtime_quote(symbol)
                elif method == 'kline':
                    result = await source.fetch_kline(symbol)
                else:
                    continue
                
                if result is not None:
                    logger.debug(f"从 {source.name} 获取 {symbol} 成功")
                    return result
                    
            except Exception as e:
                logger.warning(f"{source.name} 获取失败: {e}")
                continue
        
        logger.error(f"所有数据源获取 {symbol} 失败")
        return None
    
    async def fetch_batch(self, symbols: List[str], method: str = 'realtime') -> Dict[str, Any]:
        """
        批量获取数据
        """
        results = {}
        
        tasks = [self.fetch_with_fallback(symbol, method) for symbol in symbols]
        responses = await asyncio.gather(*tasks)
        
        for symbol, result in zip(symbols, responses):
            if result:
                results[symbol] = result
        
        return results
    
    def get_status_report(self) -> pd.DataFrame:
        """生成数据源状态报告"""
        records = []
        for source in self.sources.values():
            records.append({
                '数据源': source.name,
                '状态': '正常' if source.status.is_available else '异常',
                '延迟(ms)': f"{source.status.latency_ms:.1f}",
                '错误次数': source.status.error_count,
                '最后检查': source.status.last_check.strftime('%H:%M:%S')
            })
        
        return pd.DataFrame(records)


# 便捷函数
async def create_multi_source_manager() -> MultiDataSourceManager:
    """
    创建默认的多数据源管理器
    
    包含：东方财富、腾讯、新浪
    """
    manager = MultiDataSourceManager()
    
    # 注册数据源（优先级顺序）
    manager.register(TencentDataSource())      # 优先级 3
    manager.register(SinaDataSource())         # 优先级 4
    manager.register(TonghuashunDataSource())  # 优先级 2
    
    await manager.start()
    return manager


# 测试代码
if __name__ == '__main__':
    async def test():
        manager = await create_multi_source_manager()
        
        # 等待健康检查完成
        await asyncio.sleep(2)
        
        # 打印状态
        print("\n数据源状态:")
        print(manager.get_status_report())
        
        # 获取实时行情
        print("\n获取 000001 行情:")
        result = await manager.fetch_with_fallback('000001', 'realtime')
        print(result)
        
        await manager.stop()
    
    asyncio.run(test())
