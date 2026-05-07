"""
测试多数据源模块
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import asyncio
import numpy as np
import pandas as pd

from thermo_sys.data.multi_source import (
    TencentDataSource,
    SinaDataSource,
    TonghuashunDataSource,
    MultiDataSourceManager,
    create_multi_source_manager
)


class TestTencentDataSource:
    """测试腾讯财经数据源"""
    
    def test_creation(self):
        ds = TencentDataSource()
        assert ds.name == "tencent"
        assert ds.priority == 3
    
    def test_parse_response(self):
        ds = TencentDataSource()
        # 模拟腾讯返回格式 (使用~分隔符)
        mock_text = 'v_sh000001="1~上证指数~000001~3050.50~10.20~0.34~1234567~7890123"'
        result = ds._parse_tencent_response(mock_text, '000001')
        
        assert result['symbol'] == '000001'
        assert result['source'] == 'tencent'
        assert result['price'] == 3050.50


class TestSinaDataSource:
    """测试新浪财经数据源"""
    
    def test_creation(self):
        ds = SinaDataSource()
        assert ds.name == "sina"
        assert ds.priority == 4
    
    def test_parse_response(self):
        ds = SinaDataSource()
        # 模拟新浪返回格式
        mock_text = 'var hq_str_sh000001="上证指数,3045.20,3040.50,3050.80,3060.00,3035.00,3050.00,3051.00,12345678,9876543210"'
        result = ds._parse_sina_response(mock_text, '000001')
        
        assert result['symbol'] == '000001'
        assert result['source'] == 'sina'
        assert result['price'] == 3050.80


class TestMultiSourceManager:
    """测试多数据源管理器"""
    
    def test_creation(self):
        manager = MultiDataSourceManager()
        assert len(manager.sources) == 0
    
    def test_register(self):
        manager = MultiDataSourceManager()
        ds = TencentDataSource()
        manager.register(ds)
        assert "tencent" in manager.sources
    
    def test_available_sources(self):
        manager = MultiDataSourceManager()
        ds = TencentDataSource()
        ds.status.is_available = True
        manager.register(ds)
        
        available = manager.get_available_sources()
        assert len(available) == 1
        assert available[0].name == "tencent"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
