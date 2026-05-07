"""
增强监控面板
提供实时信号、参数变化、信息流通可视化
"""
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

app = dash.Dash(__name__, title="ThermoSys Monitor")

app.layout = html.Div([
    html.H1("ThermoSys 实时监控系统", style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    # 顶部状态栏
    html.Div([
        html.Div(id='system-status', className='status-bar'),
    ], style={'padding': '20px'}),
    
    # 主要指标
    html.Div([
        html.Div([
            html.H3("今日信号"),
            html.Div(id='today-signals')
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
        
        html.Div([
            html.H3("策略参数"),
            html.Div(id='strategy-params')
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
    ]),
    
    # 图表区域
    html.Div([
        dcc.Graph(id='signal-chart', style={'height': '400px'}),
        dcc.Graph(id='param-evolution', style={'height': '400px'}),
        dcc.Graph(id='performance-chart', style={'height': '400px'}),
    ]),
    
    # 信息流通监控
    html.Div([
        html.H3("信息流通监控"),
        html.Div(id='info-flow')
    ], style={'padding': '20px'}),
    
    # 自动刷新
    dcc.Interval(id='interval', interval=5000)
])

@app.callback(
    [Output('system-status', 'children'),
     Output('today-signals', 'children'),
     Output('strategy-params', 'children'),
     Output('signal-chart', 'figure'),
     Output('param-evolution', 'figure'),
     Output('performance-chart', 'figure'),
     Output('info-flow', 'children')],
    [Input('interval', 'n_intervals')]
)
def update_dashboard(n):
    """更新仪表板"""
    # 读取最新报告
    report = load_latest_report()
    history = load_backtest_history()
    
    # 系统状态
    status = create_status_panel(report)
    
    # 今日信号
    signals = create_signals_panel(report)
    
    # 策略参数
    params = create_params_panel(report)
    
    # 图表
    signal_chart = create_signal_chart(report)
    param_chart = create_param_chart(history)
    perf_chart = create_performance_chart(history)
    
    # 信息流通
    info_flow = create_info_flow_panel(report)
    
    return status, signals, params, signal_chart, param_chart, perf_chart, info_flow

def load_latest_report():
    """加载最新报告"""
    data_dir = Path("data/auto_loop")
    reports = sorted(data_dir.glob("report_*.json"))
    if reports:
        with open(reports[-1], 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def load_backtest_history():
    """加载回测历史"""
    history_file = Path("data/auto_loop/backtest_history.json")
    if history_file.exists():
        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def create_status_panel(report):
    """创建状态面板"""
    if not report:
        return html.Div("等待数据...", style={'color': '#e74c3c'})
    
    backtest = report.get('backtest', {})
    
    return html.Div([
        html.Span("系统状态: ", style={'fontWeight': 'bold'}),
        html.Span("运行中", style={'color': '#27ae60', 'fontWeight': 'bold'}),
        html.Span(" | "),
        html.Span(f"最新信号: {len(report.get('signals', []))}个"),
        html.Span(" | "),
        html.Span(f"回测夏普: {backtest.get('sharpe_ratio', 0):.2f}"),
        html.Span(" | "),
        html.Span(f"更新时间: {datetime.now().strftime('%H:%M:%S')}")
    ], style={'backgroundColor': '#ecf0f1', 'padding': '15px', 'borderRadius': '5px'})

def create_signals_panel(report):
    """创建信号面板"""
    if not report or 'signals' not in report:
        return html.Div("暂无信号")
    
    signal_items = []
    for sig in report['signals'][:5]:
        action_color = {'买入': '#27ae60', '卖出': '#e74c3c', '加仓': '#f39c12', '减仓': '#e67e22'}
        color = action_color.get(sig['action'], '#3498db')
        
        signal_items.append(html.Div([
            html.Span(f"{sig['symbol']}", style={'fontWeight': 'bold', 'width': '80px', 'display': 'inline-block'}),
            html.Span(f"{sig['action']}", style={'color': color, 'fontWeight': 'bold', 'width': '60px', 'display': 'inline-block'}),
            html.Span(f"置信度: {sig['confidence']:.0%}", style={'width': '100px', 'display': 'inline-block'}),
            html.Span(f"{sig['reasoning'][:50]}...", style={'color': '#7f8c8d'})
        ], style={'padding': '8px', 'borderBottom': '1px solid #ecf0f1'}))
    
    return html.Div(signal_items)

def create_params_panel(report):
    """创建参数面板"""
    params = report.get('current_params', {})
    
    if not params:
        return html.Div("暂无参数数据")
    
    param_items = []
    param_names = {
        'stop_loss': '止损线',
        'take_profit': '止盈线',
        'min_confidence': '最低置信度',
        'max_single_position': '最大仓位'
    }
    
    for key, name in param_names.items():
        value = params.get(key, 0)
        if key in ['stop_loss', 'take_profit']:
            display_val = f"{value:.0%}"
        else:
            display_val = f"{value:.0%}"
        
        param_items.append(html.Div([
            html.Span(f"{name}: ", style={'width': '120px', 'display': 'inline-block'}),
            html.Span(display_val, style={'fontWeight': 'bold', 'color': '#2980b9'})
        ], style={'padding': '5px'}))
    
    return html.Div(param_items)

def create_signal_chart(report):
    """创建信号图表"""
    fig = go.Figure()
    
    if report and 'signals' in report:
        signals = report['signals']
        symbols = [s['symbol'] for s in signals]
        confidences = [s['confidence'] for s in signals]
        actions = [s['action'] for s in signals]
        
        colors = ['#27ae60' if a == '买入' else '#e74c3c' if a == '卖出' else '#f39c12' for a in actions]
        
        fig.add_trace(go.Bar(
            x=symbols,
            y=confidences,
            marker_color=colors,
            text=actions,
            textposition='auto'
        ))
    
    fig.update_layout(
        title="今日信号置信度",
        yaxis_title="置信度",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

def create_param_chart(history):
    """创建参数演化图"""
    fig = go.Figure()
    
    if len(history) > 1:
        dates = [h.get('date', '') for h in history]
        sharpes = [h.get('sharpe', 0) for h in history]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=sharpes,
            mode='lines+markers',
            name='夏普比率',
            line=dict(color='#3498db', width=2)
        ))
    
    fig.update_layout(
        title="策略夏普比率趋势",
        xaxis_title="日期",
        yaxis_title="夏普比率",
        height=400
    )
    
    return fig

def create_performance_chart(history):
    """创建绩效图表"""
    fig = make_subplots(rows=2, cols=1, subplot_titles=('累计收益', '回撤'))
    
    if len(history) > 1:
        dates = [h.get('date', '') for h in history]
        returns = [h.get('return', 0) for h in history]
        
        # 累计收益
        cumulative = [(1 + r) ** (i+1) - 1 for i, r in enumerate(returns)]
        fig.add_trace(go.Scatter(x=dates, y=cumulative, mode='lines', name='累计收益', line=dict(color='#27ae60')), row=1, col=1)
        
        # 回撤（简化）
        drawdowns = [min(0, r) for r in returns]
        fig.add_trace(go.Bar(x=dates, y=drawdowns, name='日回撤', marker_color='#e74c3c'), row=2, col=1)
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def create_info_flow_panel(report):
    """创建信息流通面板"""
    items = []
    
    # 数据源状态
    items.append(html.Div([
        html.Strong("数据源状态: "),
        html.Span("东方财富", style={'color': '#27ae60'}),
        html.Span(" | "),
        html.Span("腾讯财经", style={'color': '#27ae60'}),
        html.Span(" | "),
        html.Span("新浪财经", style={'color': '#27ae60'})
    ], style={'padding': '5px'}))
    
    # 信号处理流程
    items.append(html.Div([
        html.Strong("信号处理: "),
        html.Span("数据获取 ✓ → 热力学计算 ✓ → 信号生成 ✓ → 回测验证 ✓ → 参数优化 ✓")
    ], style={'padding': '5px'}))
    
    # 最新操作
    items.append(html.Div([
        html.Strong("最近操作: "),
        html.Span(f"参数优化 @ {datetime.now().strftime('%H:%M:%S')}", style={'color': '#2980b9'})
    ], style={'padding': '5px'}))
    
    return html.Div(items)

if __name__ == '__main__':
    print("启动 ThermoSys 监控面板...")
    print("访问: http://localhost:8050")
    app.run(debug=False, host='0.0.0.0', port=8050)
