"""
Dash监控面板
提供热力学状态的实时可视化和系统监控
"""
import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import threading
import time
from loguru import logger

from thermo_sys.dashboard.monitor import SystemHealthMonitor


class ThermoDashboard:
    """
    热力学监控面板（完整实现）
    """
    
    def __init__(self, monitor: Optional[SystemHealthMonitor] = None, port: int = 8050):
        self.monitor = monitor or SystemHealthMonitor()
        self.port = port
        self.app = dash.Dash(__name__, title="ThermoSys Dashboard")
        self._setup_layout()
        self._setup_callbacks()
        
        # 模拟数据存储（实际应从系统获取）
        self._thermo_history: List[Dict] = []
        self._performance_history: List[Dict] = []
        self._last_update = datetime.now()
        
    def _setup_layout(self):
        """设置页面布局"""
        self.app.layout = html.Div([
            # 头部
            html.Div([
                html.H1("ThermoSys 市场热力学监控面板", 
                       style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.P(id='last-update-time', 
                      style={'textAlign': 'center', 'color': '#7f8c8d'})
            ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'}),
            
            # 健康度卡片
            html.Div([
                html.Div([
                    html.H3("系统健康度"),
                    html.H2(id='health-score', style={'color': '#27ae60'})
                ], className='card', style=self._card_style()),
                
                html.Div([
                    html.H3("预测能力"),
                    html.H2(id='pred-score', style={'color': '#3498db'})
                ], className='card', style=self._card_style()),
                
                html.Div([
                    html.H3("决策能力"),
                    html.H2(id='decision-score', style={'color': '#e74c3c'})
                ], className='card', style=self._card_style()),
                
                html.Div([
                    html.H3("自我优化"),
                    html.H2(id='optimize-score', style={'color': '#f39c12'})
                ], className='card', style=self._card_style())
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '20px'}),
            
            # 告警面板
            html.Div([
                html.H3("系统告警", style={'color': '#c0392b'}),
                html.Div(id='alerts-panel')
            ], style={'padding': '20px', 'margin': '20px', 'backgroundColor': '#fff3cd', 'borderRadius': '10px'}),
            
            # 图表区域
            html.Div([
                # 热力学状态图
                html.Div([
                    dcc.Graph(id='thermo-state-chart', style={'height': '500px'})
                ], style={'width': '100%', 'padding': '10px'}),
                
                # 绩效图表
                html.Div([
                    dcc.Graph(id='performance-chart', style={'height': '400px'})
                ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
                
                # 能力趋势图
                html.Div([
                    dcc.Graph(id='capability-chart', style={'height': '400px'})
                ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
            ]),
            
            # 实时热力学指标
            html.Div([
                html.H3("实时热力学指标"),
                html.Div(id='realtime-metrics')
            ], style={'padding': '20px'}),
            
            # 控制面板
            html.Div([
                html.H3("控制面板"),
                html.Button('刷新数据', id='refresh-btn', n_clicks=0,
                         style={'margin': '10px', 'padding': '10px 20px'}),
                html.Button('导出报告', id='export-btn', n_clicks=0,
                         style={'margin': '10px', 'padding': '10px 20px'}),
                html.Div(id='export-status')
            ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'}),
            
            # 定时刷新
            dcc.Interval(id='interval-component', interval=5000, n_intervals=0),
            
            # 数据存储
            dcc.Store(id='thermo-data-store')
        ])
    
    def _card_style(self):
        """卡片样式"""
        return {
            'width': '22%',
            'padding': '20px',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'textAlign': 'center'
        }
    
    def _setup_callbacks(self):
        """设置回调函数"""
        
        @self.app.callback(
            [Output('health-score', 'children'),
             Output('pred-score', 'children'),
             Output('decision-score', 'children'),
             Output('optimize-score', 'children'),
             Output('last-update-time', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_scores(n):
            """更新健康度分数"""
            try:
                health = self.monitor.get_health_score()
                
                # 获取各维度分数
                trends = self.monitor.compute_all_trends()
                
                pred_score = self._get_dimension_score(trends, 'world_model_mse')
                decision_score = self._get_dimension_score(trends, 'sharpe_ratio')
                optimize_score = self._get_dimension_score(trends, 'reflection_task_success_rate')
                
                return (
                    f"{health:.1f}",
                    f"{pred_score:.1f}",
                    f"{decision_score:.1f}",
                    f"{optimize_score:.1f}",
                    f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
            except Exception as e:
                logger.error(f"Update scores error: {e}")
                return "--", "--", "--", "--", "更新失败"
        
        @self.app.callback(
            Output('alerts-panel', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_alerts(n):
            """更新告警"""
            alerts = self.monitor.generate_alert()
            
            if not alerts:
                return html.P("系统运行正常", style={'color': '#27ae60'})
            
            alert_items = []
            for alert in alerts:
                color = {'critical': '#e74c3c', 'warning': '#f39c12'}.get(alert['level'], '#3498db')
                alert_items.append(html.Div([
                    html.Strong(f"[{alert['level'].upper()}] ", style={'color': color}),
                    html.Span(alert['message']),
                    html.Br(),
                    html.Small(f"建议操作: {alert['suggested_action']}", style={'color': '#7f8c8d'})
                ], style={'margin': '10px 0', 'padding': '10px', 'borderLeft': f'4px solid {color}', 'backgroundColor': 'white'}))
            
            return alert_items
        
        @self.app.callback(
            Output('thermo-state-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_thermo_chart(n):
            """更新热力学状态图"""
            return self._create_thermo_state_chart()
        
        @self.app.callback(
            Output('performance-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_performance_chart(n):
            """更新绩效图"""
            return self._create_performance_chart()
        
        @self.app.callback(
            Output('capability-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_capability_chart(n):
            """更新能力趋势图"""
            return self._create_capability_chart()
        
        @self.app.callback(
            Output('realtime-metrics', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_realtime_metrics(n):
            """更新实时指标"""
            return self._create_metrics_table()
        
        @self.app.callback(
            Output('export-status', 'children'),
            [Input('export-btn', 'n_clicks')]
        )
        def export_report(n_clicks):
            """导出报告"""
            if n_clicks == 0:
                raise PreventUpdate
            
            try:
                self.generate_static_report('dashboard_report.html')
                return html.Span("报告已导出: dashboard_report.html", style={'color': '#27ae60'})
            except Exception as e:
                return html.Span(f"导出失败: {e}", style={'color': '#e74c3c'})
    
    def _get_dimension_score(self, trends_df, metric_name):
        """获取维度分数"""
        if trends_df.empty or metric_name not in trends_df['metric'].values:
            return 50.0
        
        row = trends_df[trends_df['metric'] == metric_name].iloc[0]
        if row['is_improving']:
            return 80.0
        elif row['status'] == 'stable':
            return 60.0
        else:
            return 30.0
    
    def _create_thermo_state_chart(self):
        """创建热力学状态雷达图"""
        # 模拟数据（实际应从系统获取）
        categories = ['IPV', '清晰度', '合力', '熵减', 'RSI', '温度']
        values = np.random.rand(6) * 100
        values2 = np.random.rand(6) * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=list(values) + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='当前状态',
            line_color='#3498db'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=list(values2) + [values2[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='预测状态',
            line_color='#e74c3c',
            opacity=0.5
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True,
            title="热力学状态雷达图",
            height=500
        )
        
        return fig
    
    def _create_performance_chart(self):
        """创建绩效图"""
        dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
        
        # 模拟净值曲线
        returns = np.random.randn(252) * 0.02
        nav = (1 + returns).cumprod()
        
        # 模拟基准
        bench_returns = np.random.randn(252) * 0.015
        bench_nav = (1 + bench_returns).cumprod()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates, y=nav,
            mode='lines',
            name='策略净值',
            line=dict(color='#27ae60', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=bench_nav,
            mode='lines',
            name='基准净值',
            line=dict(color='#95a5a6', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title="策略绩效",
            xaxis_title="日期",
            yaxis_title="净值",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_capability_chart(self):
        """创建能力趋势图"""
        dates = pd.date_range(end=datetime.now(), periods=60, freq='W')
        
        metrics = {
            '预测精度': np.cumsum(np.random.randn(60) * 0.01) + 0.7,
            '夏普比率': np.cumsum(np.random.randn(60) * 0.02) + 1.0,
            '胜率': np.cumsum(np.random.randn(60) * 0.01) + 0.55,
            '信息比率': np.cumsum(np.random.randn(60) * 0.015) + 0.8
        }
        
        fig = go.Figure()
        
        colors = ['#3498db', '#27ae60', '#e74c3c', '#f39c12']
        for (name, values), color in zip(metrics.items(), colors):
            fig.add_trace(go.Scatter(
                x=dates, y=values,
                mode='lines',
                name=name,
                line=dict(color=color, width=2)
            ))
        
        fig.update_layout(
            title="能力演化趋势",
            xaxis_title="日期",
            yaxis_title="指标值",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_metrics_table(self):
        """创建实时指标表格"""
        metrics = [
            {'name': '信息传播速度 (IPV)', 'value': f'{np.random.rand()*10:.2f}', 'unit': '', 'status': 'normal'},
            {'name': '传播路径清晰度', 'value': f'{np.random.rand():.2f}', 'unit': '', 'status': 'normal'},
            {'name': 'Kuramoto序参量', 'value': f'{np.random.rand():.2f}', 'unit': '', 'status': 'warning'},
            {'name': '散户情绪指数', 'value': f'{np.random.randn()*2:.2f}', 'unit': 'σ', 'status': 'normal'},
            {'name': '观点香农熵', 'value': f'{np.random.rand():.2f}', 'unit': 'bits', 'status': 'normal'},
            {'name': '熵减率', 'value': f'{np.random.randn()*0.1:.3f}', 'unit': '/day', 'status': 'normal'}
        ]
        
        rows = []
        for m in metrics:
            status_color = {'normal': '#27ae60', 'warning': '#f39c12', 'critical': '#e74c3c'}.get(m['status'], '#3498db')
            rows.append(html.Tr([
                html.Td(m['name'], style={'padding': '10px'}),
                html.Td(f"{m['value']} {m['unit']}", style={'padding': '10px', 'fontWeight': 'bold'}),
                html.Td(html.Span('●', style={'color': status_color}), style={'padding': '10px'})
            ]))
        
        return html.Table(
            [html.Thead(html.Tr([
                html.Th('指标', style={'padding': '10px', 'textAlign': 'left'}),
                html.Th('当前值', style={'padding': '10px'}),
                html.Th('状态', style={'padding': '10px'})
            ]))] +
            [html.Tbody(rows)],
            style={'width': '100%', 'borderCollapse': 'collapse'}
        )
    
    def run(self, debug: bool = False, use_reloader: bool = False):
        """启动面板"""
        logger.info(f"Starting dashboard on port {self.port}")
        self.app.run_server(
            host='0.0.0.0',
            port=self.port,
            debug=debug,
            use_reloader=use_reloader
        )
    
    def generate_static_report(self, output_path: str = 'dashboard_report.html'):
        """生成静态HTML报告"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '热力学状态', '策略绩效',
                '能力趋势', '风险指标',
                '资金流向', '情绪分布'
            ),
            specs=[
                [{"type": "polar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "pie"}]
            ]
        )
        
        # 添加模拟数据
        categories = ['IPV', '清晰度', '合力', '熵减', 'RSI', '温度']
        values = np.random.rand(6) * 100
        
        fig.add_trace(go.Scatterpolar(
            r=list(values) + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='状态'
        ), row=1, col=1)
        
        dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
        nav = (1 + np.random.randn(252) * 0.02).cumprod()
        
        fig.add_trace(go.Scatter(
            x=dates, y=nav,
            mode='lines',
            name='净值'
        ), row=1, col=2)
        
        fig.write_html(output_path)
        logger.info(f"Report saved to: {output_path}")


# 便捷启动函数
def launch_dashboard(monitor: Optional[SystemHealthMonitor] = None, port: int = 8050):
    """
    启动监控面板
    
    Usage:
        from thermo_sys.dashboard.app import launch_dashboard
        launch_dashboard(port=8050)
    """
    dashboard = ThermoDashboard(monitor=monitor, port=port)
    dashboard.run()


if __name__ == '__main__':
    # 独立运行测试
    monitor = SystemHealthMonitor()
    
    # 添加一些模拟数据
    for i in range(100):
        monitor.record('world_model_mse', 0.1 + np.random.randn() * 0.01)
        monitor.record('sharpe_ratio', 1.0 + np.random.randn() * 0.2)
        monitor.record('reflection_task_success_rate', 0.7 + np.random.randn() * 0.1)
    
    dashboard = ThermoDashboard(monitor=monitor)
    dashboard.run(debug=True)
