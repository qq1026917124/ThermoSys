"""
系统健康度监控面板
追踪Agent能力的长期演化趋势
"""
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from typing import Dict, List, Optional
from datetime import datetime


class SystemHealthMonitor:
    """
    系统健康度监控器
    
    追踪指标：
    1. 预测能力：world_model_mse, causal_ate_accuracy
    2. 决策能力：sharpe_ratio, max_drawdown, regime_switch_latency
    3. 自我优化能力：reflection_task_success_rate, code_change_improvement_rate
    4. 热力学一致性：thermo_violation_count
    """
    
    def __init__(self, window_sizes: Optional[Dict[str, int]] = None):
        self.window_sizes = window_sizes or {
            'world_model_mse': 252,
            'sharpe_ratio': 252,
            'regime_switch_latency': 20,
            'reflection_task_success_rate': 50,
            'thermo_violation_count': 252
        }
        
        self.metrics = defaultdict(lambda: deque(maxlen=1000))
        self.timestamps = deque(maxlen=1000)
        
    def record(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """记录指标值"""
        self.metrics[metric_name].append(value)
        if timestamp:
            self.timestamps.append(timestamp)
    
    def record_batch(self, data: Dict[str, float], timestamp: Optional[datetime] = None):
        """批量记录指标"""
        for name, value in data.items():
            self.record(name, value, timestamp)
    
    def compute_trend(self, metric_name: str, window: Optional[int] = None) -> Dict[str, float]:
        """
        计算指标的趋势（Mann-Kendall检验）
        
        Returns:
            {'tau': float, 'p_value': float, 'is_improving': bool}
        """
        from thermo_sys.utils.math_utils import mann_kendall_trend
        
        data = list(self.metrics[metric_name])
        if window:
            data = data[-window:]
        
        if len(data) < 10:
            return {'tau': 0.0, 'p_value': 1.0, 'is_improving': False, 'status': 'insufficient_data'}
        
        tau, p_value = mann_kendall_trend(np.array(data))
        
        # 判定是否改善（取决于指标类型）
        # 对于MSE/MDD等反向指标，tau<0表示改善
        # 对于Sharpe等正向指标，tau>0表示改善
        reverse_metrics = ['world_model_mse', 'max_drawdown', 'regime_switch_latency', 'thermo_violation_count']
        is_reverse = metric_name in reverse_metrics
        
        is_improving = (tau < 0) if is_reverse else (tau > 0)
        is_significant = p_value < 0.1
        
        return {
            'tau': tau,
            'p_value': p_value,
            'is_improving': is_improving,
            'is_significant': is_significant,
            'status': 'improving' if is_improving and is_significant else 
                     'degrading' if not is_improving and is_significant else 'stable'
        }
    
    def compute_all_trends(self) -> pd.DataFrame:
        """计算所有指标的趋势"""
        results = []
        for metric_name in self.metrics.keys():
            trend = self.compute_trend(metric_name)
            results.append({
                'metric': metric_name,
                **trend,
                'current_value': list(self.metrics[metric_name])[-1] if self.metrics[metric_name] else None,
                'window_size': len(self.metrics[metric_name])
            })
        
        return pd.DataFrame(results)
    
    def detect_anomaly(self, metric_name: str, threshold_std: float = 3.0) -> bool:
        """检测指标是否异常（超过3倍标准差）"""
        data = list(self.metrics[metric_name])
        if len(data) < 30:
            return False
        
        recent = data[-5:]
        historical = data[:-5]
        
        mean = np.mean(historical)
        std = np.std(historical)
        
        if std < 1e-8:
            return False
        
        z_scores = [(x - mean) / std for x in recent]
        return any(abs(z) > threshold_std for z in z_scores)
    
    def get_health_score(self) -> float:
        """
        计算综合健康度评分 [0, 100]
        """
        scores = []
        
        # 各项指标的权重
        weights = {
            'world_model_mse': 0.2,
            'sharpe_ratio': 0.3,
            'max_drawdown': 0.2,
            'reflection_task_success_rate': 0.15,
            'thermo_violation_count': 0.15
        }
        
        for metric, weight in weights.items():
            if metric not in self.metrics or len(self.metrics[metric]) < 10:
                continue
            
            trend = self.compute_trend(metric)
            if trend['is_improving']:
                scores.append(weight * 100)
            elif trend['status'] == 'stable':
                scores.append(weight * 60)
            else:
                scores.append(weight * 20)
        
        if not scores:
            return 50.0
        
        # 计算加权平均
        total_weighted_score = sum(scores)
        total_weight = sum(weights[m] for m in weights if m in self.metrics and len(self.metrics[m]) >= 10)
        
        return min(100, max(0, total_weighted_score / total_weight)) if total_weight > 0 else 50.0
    
    def generate_alert(self) -> List[Dict[str, str]]:
        """生成异常告警"""
        alerts = []
        
        # 检查预测能力退化
        if 'world_model_mse' in self.metrics:
            trend = self.compute_trend('world_model_mse')
            if trend['status'] == 'degrading' and trend.get('is_significant', False):
                alerts.append({
                    'level': 'warning',
                    'metric': 'world_model_mse',
                    'message': '世界模型预测精度持续下降，建议触发元学习适应或检查数据质量',
                    'suggested_action': 'trigger_adaptation'
                })
        
        # 检查策略绩效
        if 'sharpe_ratio' in self.metrics:
            recent_sharpe = list(self.metrics['sharpe_ratio'])[-20:]
            if len(recent_sharpe) >= 20 and np.mean(recent_sharpe) < 0:
                alerts.append({
                    'level': 'critical',
                    'metric': 'sharpe_ratio',
                    'message': '近期夏普比率为负，策略可能失效',
                    'suggested_action': 'pause_trading'
                })
        
        # 检查热力学一致性
        if 'thermo_violation_count' in self.metrics:
            recent_violations = sum(list(self.metrics['thermo_violation_count'])[-20:])
            if recent_violations > 10:
                alerts.append({
                    'level': 'warning',
                    'metric': 'thermo_violation_count',
                    'message': f'近期违反热力学约束交易次数过多({recent_violations})',
                    'suggested_action': 'tighten_constraints'
                })
        
        return alerts
    
    def get_summary(self) -> Dict[str, any]:
        """生成监控摘要"""
        return {
            'health_score': self.get_health_score(),
            'total_records': len(self.timestamps),
            'trends': self.compute_all_trends().to_dict('records'),
            'alerts': self.generate_alert(),
            'timestamp': datetime.now().isoformat()
        }


class ThermoDashboard:
    """
    热力学监控面板（基于Dash/Plotly）
    """
    
    def __init__(self, monitor: SystemHealthMonitor, port: int = 8050):
        self.monitor = monitor
        self.port = port
        self.app = None
        
    def _create_app(self):
        """创建Dash应用"""
        try:
            import dash
            from dash import dcc, html
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Dash/Plotly not installed. Run: pip install dash plotly")
            return None
        
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1("ThermoSys 市场热力学监控面板"),
            
            html.Div([
                html.H3(f"系统健康度: {self.monitor.get_health_score():.1f}/100")
            ]),
            
            dcc.Graph(id='thermo-state-chart'),
            dcc.Graph(id='performance-chart'),
            dcc.Graph(id='capability-trend-chart'),
            
            dcc.Interval(id='interval-component', interval=30*1000)  # 30秒刷新
        ])
        
        return app
    
    def run(self, debug: bool = False):
        """启动面板服务器"""
        self.app = self._create_app()
        if self.app:
            self.app.run(host='0.0.0.0', port=self.port, debug=debug)
    
    def generate_static_report(self, output_path: str = 'report.html'):
        """生成静态HTML报告"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not installed")
            return
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('热力学状态演化', '策略绩效', '能力趋势'),
            vertical_spacing=0.1
        )
        
        # 这里简化处理，实际应从monitor获取历史数据绘制
        fig.write_html(output_path)
        print(f"报告已保存至: {output_path}")
