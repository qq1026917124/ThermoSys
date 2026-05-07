# ThermoSys 市场热力学分析系统

基于热力学框架的市场情绪分析与自主进化Agent系统。

## 核心特性

- **散户情绪指标 (RSI)**：基于行为数据为主的客观情绪量化
- **信息传播速度 (IPV)**：量化信息在异质网络中的有效扩散速率
- **热力学传播路径**：构建板块间热阻网络，计算信息传导路径清晰度
- **合力形成指标 (CF)**：使用Kuramoto序参量和熵减率识别市场共振
- **自主进化Agent**：世界模型 + 元强化学习 + 反思-进化闭环

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行回测

```bash
python -m thermo_sys.main --mode backtest --start 2019-01-01 --end 2024-12-31
```

### 运行测试

```bash
python -m pytest tests/ -v
```

## 项目结构

```
ThermoSys/
├── thermo_sys/
│   ├── core/           # 热力学状态计算模块
│   │   ├── thermo_state.py      # 统一状态接口
│   │   ├── rsi.py               # 散户情绪指数
│   │   ├── ipv.py               # 信息传播速度
│   │   ├── heat_transfer.py     # 热传导网络
│   │   └── coherence.py         # 合力指标
│   ├── agent/          # Agent核心系统
│   │   ├── world_model.py       # 热力学世界模型
│   │   ├── policy.py            # 策略网络
│   │   └── meta_controller.py   # 元学习控制器
│   ├── meta/           # 元认知层
│   │   ├── reflection.py        # 反思Agent
│   │   └── evolution.py         # 进化Agent
│   ├── data/           # 数据层
│   │   ├── collectors.py        # 数据抓取
│   │   └── processors.py        # 数据清洗处理
│   ├── backtest/       # 回测框架
│   │   ├── engine.py            # 回测引擎
│   │   └── metrics.py           # 绩效指标
│   ├── dashboard/      # 监控面板
│   │   └── monitor.py           # 系统健康监控
│   ├── utils/          # 工具函数
│   └── main.py         # 主入口
├── config/             # 配置文件
├── tests/              # 单元测试
├── docs/               # 文档
└── scripts/            # 脚本
```

## 核心概念

### 热力学观测空间

系统将市场抽象为热力学状态向量：

| 维度 | 指标 | 含义 |
|-----|------|------|
| 全局 | IPV | 信息传播速度 |
| 全局 | Clarity | 传播路径清晰度 |
| 全局 | Coherence | Kuramoto序参量 |
| 全局 | Entropy | 观点香农熵 |
| 全局 | RSI | 散户情绪指数 |
| 板块 | Temperatures | 各板块信息密度 |
| 群体 | Phases | 投资者群体相位角 |

### 择时信号

系统输出三维择时矩阵：

- **速度维度**：IPV > 90%分位 + R₀ > 2 → 信息加速扩散
- **路径维度**：Clarity > 0.7 → 热点有序传导
- **合力维度**：Coherence > 0.6 + 熵减 → 趋势形成

### 自我优化闭环

```
交易执行 → 结果反馈 → 世界模型更新 → 反思Agent分析 → 进化Agent改进 → A/B测试 → 部署/回滚
```

## 配置说明

配置文件位于 `config/system_config.yaml`，包含：

- `data_sources`: 数据源配置（东方财富、雪球、百度指数等）
- `thermo_state`: 热力学参数（权重、阈值、窗口大小）
- `agent`: Agent网络结构和训练参数
- `meta_cognition`: 反思和进化周期配置
- `backtest`: 回测参数

## 开发计划

- [x] 核心热力学模块
- [x] Agent决策系统
- [x] 元认知层框架
- [x] 回测引擎
- [x] 实盘数据接入 (WebSocket实时流)
- [ ] 分布式训练
- [x] Dash监控面板 (完整可视化)
- [x] 因果发现模块（DoWhy风格集成）

## 免责声明

本系统仅供研究和学习使用，不构成投资建议。市场有风险，投资需谨慎。
