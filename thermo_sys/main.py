"""
ThermoSys 主入口
整合所有模块，提供完整的运行流程
"""
import os
import sys
import yaml
import asyncio
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from thermo_sys.core import (
    ThermoState, ThermoStateEncoder,
    RetailSentimentIndex, InformationPropagationVelocity,
    HeatTransferNetwork, CoherenceForce
)
from thermo_sys.agent import ThermoWorldModel, ThermoAgent, MetaController
from thermo_sys.meta import ReflectionAgent, EvolutionAgent
from thermo_sys.backtest import BacktestEngine, calculate_metrics
from thermo_sys.dashboard import SystemHealthMonitor
from thermo_sys.data.collectors import UnifiedDataPipeline
from thermo_sys.utils.data_utils import load_config


class ThermoSystem:
    """
    ThermoSys 主系统
    
    整合数据层、热力学计算层、Agent层、元认知层
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.system_config = self.config['system']
        self.thermo_config = self.config['thermo_state']
        self.agent_config = self.config['agent']
        
        # 初始化各模块
        self._init_core_modules()
        self._init_agent_modules()
        self._init_meta_modules()
        self._init_monitor()
        
        # 状态追踪
        self.thermo_history = []
        self.action_history = []
        self.equity_history = []
        
    def _init_core_modules(self):
        """初始化核心热力学模块"""
        # 散户情绪指数
        self.rsi_engine = RetailSentimentIndex(
            zscore_window=self.thermo_config['zscore_window'],
            lookback_percentile=self.thermo_config.get('lookback_percentile', 500)
        )
        
        # 信息传播速度
        self.ipv_engine = InformationPropagationVelocity(
            **self.thermo_config['ipv']
        )
        
        # 热传导网络
        self.heat_network = HeatTransferNetwork(
            sectors=self.thermo_config['sectors']
        )
        
        # 合力指标
        self.coherence_engine = CoherenceForce(
            **self.thermo_config['coherence']
        )
        
    def _init_agent_modules(self):
        """初始化Agent模块"""
        state_dim = self.agent_config['state_dim']
        action_dim = self.agent_config['action_dim']
        hidden_dim = self.agent_config['hidden_dim']
        
        # 世界模型
        self.world_model = ThermoWorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        
        # 策略Agent
        self.agent = ThermoAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_lr=self.agent_config['policy']['actor_lr'],
            critic_lr=self.agent_config['policy']['critic_lr'],
            gamma=self.agent_config['policy']['gamma'],
            tau=self.agent_config['policy']['tau'],
            reward_lambda_coherence=self.agent_config['reward']['lambda_coherence'],
            reward_lambda_entropy=self.agent_config['reward']['lambda_entropy'],
            clarity_penalty_factor=self.agent_config['reward']['clarity_penalty_factor']
        )
        
        # 元控制器
        self.meta_controller = MetaController(
            state_dim=state_dim,
            n_regimes=self.agent_config['meta_controller']['n_regimes']
        )
        
    def _init_meta_modules(self):
        """初始化元认知模块"""
        meta_config = self.config['meta_cognition']
        
        self.reflection_agent = ReflectionAgent(
            llm_endpoint=meta_config['reflection']['llm_endpoint'],
            min_episodes=meta_config['reflection']['min_episodes']
        )
        
        self.evolution_agent = EvolutionAgent(
            codebase_path=str(Path(__file__).parent.parent),
            ab_test_window=meta_config['evolution']['ab_test_window'],
            improvement_threshold=meta_config['evolution']['improvement_threshold'],
            auto_execute=meta_config['evolution']['auto_execute']
        )
        
    def _init_monitor(self):
        """初始化监控模块"""
        self.monitor = SystemHealthMonitor()
        
    def compute_thermo_state(self, data: Dict[str, pd.DataFrame], date: datetime) -> ThermoState:
        """
        计算当日热力学状态
        
        Args:
            data: 各数据源的字典
            date: 当前日期
            
        Returns:
            ThermoState
        """
        # 1. 计算RSI
        margin_balance = data.get('margin_balance', pd.Series())
        small_order_flow = data.get('money_flow', pd.DataFrame()).get('small_inflow', pd.Series())
        new_accounts = data.get('new_accounts', pd.Series())
        search_index = data.get('search_index', pd.DataFrame()).sum(axis=1)  # 所有关键词搜索量之和
        option_pcr = data.get('option_pcr', pd.Series())
        
        if len(margin_balance) > 0:
            rsi_series = self.rsi_engine.compute(
                margin_balance, small_order_flow, new_accounts,
                search_index, option_pcr
            )
            rsi = rsi_series.loc[date] if date in rsi_series.index else 0.0
        else:
            rsi = 0.0
        
        # 2. 计算IPV
        mentions = data.get('posts', pd.DataFrame()).get('post_count', pd.Series())
        engagement = data.get('posts', pd.DataFrame()).get('comment_count', pd.Series())
        volume_change = data.get('money_flow', pd.DataFrame()).get('main_inflow', pd.Series()).pct_change()
        
        if len(mentions) > 0:
            rho = self.ipv_engine.compute_info_density(mentions, engagement, volume_change)
            ipv = abs(rho.pct_change().loc[date]) if date in rho.index else 0.0
        else:
            ipv = 0.0
        
        # 3. 计算合力
        sentiment_dist = data.get('sentiment', pd.DataFrame())
        if len(sentiment_dist) > 0:
            entropy_series = self.coherence_engine.compute_entropy(sentiment_dist)
            entropy = entropy_series.loc[date] if date in entropy_series.index else 1.0
            entropy_change = entropy_series.diff().loc[date] if date in entropy_series.index else 0.0
        else:
            entropy = 1.0
            entropy_change = 0.0
        
        # 模拟板块温度和群体相位
        n_sectors = len(self.thermo_config['sectors'])
        n_groups = len(self.thermo_config['investor_groups'])
        
        sector_temperatures = {
            sector: np.random.randn() for sector in self.thermo_config['sectors']
        }
        group_phases = {
            group: np.random.uniform(-np.pi, np.pi) 
            for group in self.thermo_config['investor_groups']
        }
        
        # 计算MTS
        clarity = np.random.uniform(0, 1)
        coherence = np.random.uniform(0, 1)
        mts = 0.4 * np.clip(ipv, -3, 3) / 3 + 0.3 * clarity + 0.3 * (2 * coherence - 1)
        
        return ThermoState(
            ipv=ipv,
            clarity=clarity,
            coherence=coherence,
            sector_temperatures=sector_temperatures,
            group_phases=group_phases,
            entropy=entropy,
            entropy_change=entropy_change,
            rsi=rsi,
            mts=mts,
            timestamp=date
        )
    
    def generate_signal(self, thermo_state: ThermoState) -> Dict[str, any]:
        """
        基于热力学状态生成交易信号
        
        Returns:
            {'action': float, 'confidence': float, 'reasoning': str}
        """
        state_vec = thermo_state.to_tensor().numpy()
        action = self.agent.select_action(state_vec, deterministic=False)
        
        # 生成推理说明
        reasoning_parts = []
        
        if thermo_state.is_extreme_fear():
            reasoning_parts.append(f"RSI处于极端恐惧区({thermo_state.rsi:.2f})")
        elif thermo_state.is_extreme_greed():
            reasoning_parts.append(f"RSI处于极端贪婪区({thermo_state.rsi:.2f})")
        
        if thermo_state.is_coherent():
            reasoning_parts.append(f"市场形成合力(序参量={thermo_state.coherence:.2f})")
        
        if thermo_state.is_path_clear():
            reasoning_parts.append(f"传播路径清晰({thermo_state.clarity:.2f})")
        else:
            reasoning_parts.append(f"传播路径紊乱({thermo_state.clarity:.2f})")
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "中性状态"
        
        return {
            'action': float(action[0]) if isinstance(action, np.ndarray) else float(action),
            'confidence': abs(thermo_state.mts) / 3,
            'reasoning': reasoning,
            'thermo_state': thermo_state.to_dict
        }
    
    def update(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """
        更新Agent（每日收盘后调用）
        """
        # 添加到经验回放
        self.agent.add_experience(state, action, reward, next_state, done)
        
        # 训练一步
        train_info = self.agent.train()
        
        # 更新世界模型
        if len(self.agent.buffer) > 100:
            self.world_model.online_update(state, action, next_state, n_steps=5)
        
        # 记录监控指标
        if train_info:
            self.monitor.record('critic_loss', train_info.get('critic_loss', 0))
        
    def run_backtest(
        self,
        price: pd.Series,
        data_pipeline_output: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        运行完整回测
        """
        if start_date:
            price = price[price.index >= start_date]
        if end_date:
            price = price[price.index <= end_date]
        
        dates = price.index
        
        # 预计算所有日期的热力学状态
        thermo_states = []
        for date in dates:
            ts = self.compute_thermo_state(data_pipeline_output, date)
            thermo_states.append(ts.to_dict)
        
        thermo_df = pd.DataFrame(thermo_states)
        thermo_df.index = dates
        
        # 信号生成函数
        def signal_generator(thermo_df):
            signals = pd.DataFrame(index=thermo_df.index)
            signals['signal'] = 0
            signals['strength'] = 0.0
            
            for idx, row in thermo_df.iterrows():
                rsi = row.get('rsi', 0)
                coherence = row.get('coherence', 0)
                clarity = row.get('clarity', 0.5)
                mts = row.get('mts', 0)
                
                # 简化版信号生成
                if rsi < -2.0 and coherence > 0.6:
                    sig = 1
                elif rsi > 2.0 and coherence < 0.3:
                    sig = -1
                elif mts > 2.0:
                    sig = 1
                elif mts < -2.0:
                    sig = -1
                else:
                    sig = 0
                
                signals.loc[idx, 'signal'] = sig
                signals.loc[idx, 'strength'] = min(abs(mts) / 3, 1.0)
            
            return signals
        
        # 运行回测
        engine = BacktestEngine(
            initial_capital=self.config['backtest']['initial_capital'],
            commission=self.config['backtest']['commission'],
            slippage=self.config['backtest']['slippage']
        )
        
        result = engine.run(price, signal_generator, thermo_df)
        
        return {
            'metrics': result.metrics,
            'equity_curve': result.equity_curve,
            'positions': result.positions,
            'trades': result.trades,
            'signals': result.signals
        }
    
    def run_reflection_cycle(self):
        """
        运行反思-进化周期（每周/月调用）
        """
        print("\n[系统] 启动反思周期...")
        
        # 1. 反思Agent分析
        # 构建episode记录（简化版）
        for _ in range(min(20, len(self.thermo_history))):
            # 模拟添加episode记录
            pass
        
        analysis = self.reflection_agent.analyze_episode()
        
        if analysis['status'] != 'success':
            print(f"[反思] 数据不足: {analysis}")
            return
        
        print(f"[反思] 根因分析: {len(analysis['root_causes'])}个问题")
        
        # 2. 生成改进任务
        tasks = self.reflection_agent.generate_improvement_tasks(analysis)
        
        # 3. 进化Agent执行
        for task in tasks:
            result = self.evolution_agent.execute_task(task)
            print(f"[进化] 任务 {result.task_id}: {result.status}, 改进: {result.improvement:.4f}")
        
        # 4. 更新监控
        self.monitor.record('reflection_task_success_rate', 
                          sum(1 for t in self.evolution_agent.task_history if t.status == 'success') / 
                          max(len(self.evolution_agent.task_history), 1))
    
    def get_status(self) -> Dict:
        """获取系统当前状态"""
        return {
            'system': self.system_config['name'],
            'version': self.system_config['version'],
            'mode': self.system_config['mode'],
            'health_score': self.monitor.get_health_score(),
            'records_count': len(self.thermo_history),
            'alerts': self.monitor.generate_alert()
        }


def main():
    parser = argparse.ArgumentParser(description='ThermoSys Market Thermodynamic Analysis System')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--mode', type=str, default='backtest', choices=['backtest', 'live', 'train'])
    parser.add_argument('--start', type=str, default='2019-01-01', help='回测开始日期')
    parser.add_argument('--end', type=str, default='2024-12-31', help='回测结束日期')
    parser.add_argument('--reflect', action='store_true', help='运行反思周期')
    
    args = parser.parse_args()
    
    # 初始化系统
    print("="*60)
    print("ThermoSys 市场热力学分析系统")
    print("="*60)
    
    system = ThermoSystem(config_path=args.config)
    print(f"[初始化] 系统: {system.system_config['name']} v{system.system_config['version']}")
    print(f"[初始化] 模式: {args.mode}")
    
    if args.mode == 'backtest':
        # 模拟数据（实际应从数据管道获取）
        print("[回测] 生成模拟数据...")
        dates = pd.date_range(start=args.start, end=args.end, freq='B')
        n = len(dates)
        
        # 模拟价格
        returns = np.random.randn(n) * 0.015
        price = 100 * (1 + returns).cumprod()
        price = pd.Series(price, index=dates)
        
        # 模拟数据源
        mock_data = {
            'margin_balance': pd.Series(15000 + np.cumsum(np.random.randn(n)*100), index=dates),
            'money_flow': pd.DataFrame({
                'small_inflow': np.random.randn(n) * 5000,
                'main_inflow': np.random.randn(n) * 10000,
            }, index=dates),
            'new_accounts': pd.Series(30000 + np.random.randn(n) * 5000, index=dates),
            'option_pcr': pd.Series(0.9 + np.random.randn(n) * 0.1, index=dates),
            'posts': pd.DataFrame({
                'post_count': np.random.randint(100, 1000, n),
                'comment_count': np.random.randint(500, 5000, n),
            }, index=dates),
            'sentiment': pd.DataFrame(
                np.random.dirichlet([1,2,4,2,1], n),
                index=dates,
                columns=['extreme_bear', 'bear', 'neutral', 'bull', 'extreme_bull']
            ),
            'search_index': pd.DataFrame(
                np.random.randint(1000, 5000, (n, 6)),
                index=dates,
                columns=['股票开户', '牛市', '涨停', '割肉', '跌停', '清仓']
            )
        }
        
        print(f"[回测] 数据范围: {dates[0]} 至 {dates[-1]}, 共{n}个交易日")
        
        # 运行回测
        result = system.run_backtest(price, mock_data, args.start, args.end)
        
        # 输出结果
        print("\n" + "="*60)
        print("回测结果")
        print("="*60)
        for key, value in result['metrics'].items():
            if isinstance(value, float):
                print(f"  {key:25s}: {value:>12.4f}")
            else:
                print(f"  {key:25s}: {value}")
        
        print(f"\n  最终净值: {result['equity_curve'].iloc[-1]:,.2f}")
        print(f"  总收益率: {(result['equity_curve'].iloc[-1]/result['equity_curve'].iloc[0]-1)*100:.2f}%")
    
    if args.reflect:
        system.run_reflection_cycle()
    
    # 输出系统状态
    status = system.get_status()
    print("\n" + "="*60)
    print("系统状态")
    print("="*60)
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n[完成] ThermoSys运行结束")


if __name__ == '__main__':
    main()
