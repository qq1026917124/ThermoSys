"""
Dashboard模块测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from thermo_sys.dashboard.app import ThermoDashboard
from thermo_sys.dashboard.monitor import SystemHealthMonitor


class TestThermoDashboard:
    """测试Dashboard"""
    
    def test_dashboard_creation(self):
        """测试Dashboard创建"""
        monitor = SystemHealthMonitor()
        dashboard = ThermoDashboard(monitor=monitor, port=8050)
        
        assert dashboard.monitor is not None
        assert dashboard.port == 8050
        assert dashboard.app is not None
    
    def test_layout_setup(self):
        """测试布局设置"""
        monitor = SystemHealthMonitor()
        dashboard = ThermoDashboard(monitor=monitor)
        
        # 检查布局中是否包含关键组件
        layout = dashboard.app.layout
        assert layout is not None
    
    def test_static_report_generation(self):
        """测试静态报告生成"""
        monitor = SystemHealthMonitor()
        dashboard = ThermoDashboard(monitor=monitor)
        
        # 生成报告
        dashboard.generate_static_report('test_report.html')
        
        import os
        assert os.path.exists('test_report.html')
        
        # 清理
        os.remove('test_report.html')


class TestSystemHealthMonitor:
    """测试系统健康监控"""
    
    def test_monitor_creation(self):
        """测试监控器创建"""
        monitor = SystemHealthMonitor()
        assert monitor is not None
    
    def test_record_metrics(self):
        """测试记录指标"""
        monitor = SystemHealthMonitor()
        
        monitor.record('world_model_mse', 0.1)
        monitor.record('sharpe_ratio', 1.5)
        
        assert len(monitor.metrics['world_model_mse']) == 1
        assert len(monitor.metrics['sharpe_ratio']) == 1
    
    def test_compute_trend(self):
        """测试趋势计算"""
        monitor = SystemHealthMonitor()
        
        # 添加趋势数据（递减表示改善）
        for i in range(50):
            monitor.record('world_model_mse', 0.2 - i * 0.002)
        
        trend = monitor.compute_trend('world_model_mse')
        
        assert 'tau' in trend
        assert 'p_value' in trend
        assert 'is_improving' in trend
        # 应该是改善的（tau < 0对于反向指标）
        assert trend['is_improving'] == True
    
    def test_health_score(self):
        """测试健康度评分"""
        monitor = SystemHealthMonitor()
        
        # 添加一些数据
        for i in range(20):
            monitor.record('world_model_mse', 0.1 + np.random.randn() * 0.01)
            monitor.record('sharpe_ratio', 1.0 + np.random.randn() * 0.2)
        
        score = monitor.get_health_score()
        
        assert 0 <= score <= 100
    
    def test_alerts(self):
        """测试告警生成"""
        monitor = SystemHealthMonitor()
        
        # 添加异常数据（连续负夏普）
        for i in range(25):
            monitor.record('sharpe_ratio', -0.5)
        
        alerts = monitor.generate_alert()
        
        assert len(alerts) > 0
        assert alerts[0]['level'] == 'critical'
    
    def test_batch_record(self):
        """测试批量记录"""
        monitor = SystemHealthMonitor()
        
        data = {
            'world_model_mse': 0.1,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.05
        }
        
        monitor.record_batch(data)
        
        assert len(monitor.metrics['world_model_mse']) == 1
        assert len(monitor.metrics['sharpe_ratio']) == 1
        assert len(monitor.metrics['max_drawdown']) == 1
    
    def test_insufficient_data(self):
        """测试数据不足情况"""
        monitor = SystemHealthMonitor()
        
        # 只添加少量数据
        for i in range(5):
            monitor.record('world_model_mse', 0.1)
        
        trend = monitor.compute_trend('world_model_mse')
        
        assert trend['status'] == 'insufficient_data'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
