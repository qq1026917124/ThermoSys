"""
端到端自动回测闭环系统

完全自动化的策略进化：
1. 每日自动生成信号
2. 自动回测验证信号质量
3. 基于回测结果优化策略参数
4. 持续自我进化，无需人工干预

新架构：
信号生成 -> 即时回测验证 -> 策略评分 -> 参数调优 -> 再次回测 -> 最优策略输出
"""
import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from thermo_sys.core import ThermoState
from thermo_sys.execution.manual_executor import ManualTradeExecutor, WeeklyStrategy
from thermo_sys.execution.auto_backtest import AutoBacktestEngine, BacktestConfig
from thermo_sys.execution.strategy_optimizer import StrategyOptimizer, StrategyParameters
from thermo_sys.dashboard.monitor import SystemHealthMonitor


class AutoEvolutionLoop:
    """
    全自动策略进化闭环
    
    工作流程：
    
    每日循环:
      1. 获取市场数据
      2. 计算热力学状态
      3. 生成交易信号
      4. 自动回测验证信号有效性（历史数据验证）
      5. 评估信号质量
      6. 如果信号质量下降，触发参数优化
      7. 输出当日最优策略和预期绩效
    
    每周循环:
      1. 汇总本周所有回测结果
      2. 对比不同参数组合的表现
      3. 识别最优参数区域
      4. 更新策略默认参数
      5. 生成下周策略配置
    
    每月循环:
      1. 深度回测（多周期、多市场状态）
      2. 策略鲁棒性测试
      3. 如果策略失效，触发重新优化
      4. 存档策略版本
    """
    
    def __init__(self, config_path: str = "config/system_config.yaml"):
        self.config = self._load_config(config_path)
        self.signal_generator = ManualTradeExecutor(config_path)
        self.backtest_engine = AutoBacktestEngine()
        self.optimizer = StrategyOptimizer()
        self.monitor = SystemHealthMonitor()
        
        # 当前最优参数
        self.current_params = StrategyParameters()
        
        # 历史回测结果
        self.backtest_history: List[Dict] = []
        self.signal_quality_history: List[Dict] = []
        
        # 数据目录
        self.data_dir = Path("data/auto_loop")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载历史
        self._load_history()
        
        logger.info("全自动策略进化系统初始化完成")
    
    def _load_config(self, path: str) -> Dict:
        """加载配置"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"加载配置失败: {e}")
            return {}
    
    def _load_history(self):
        """加载历史回测结果"""
        history_file = self.data_dir / "backtest_history.json"
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                self.backtest_history = json.load(f)
    
    def _save_history(self):
        """保存历史回测结果"""
        history_file = self.data_dir / "backtest_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.backtest_history[-100:], f, ensure_ascii=False, indent=2)
    
    def run_daily_cycle(self, 
                       market_data: Dict,
                       price_data: pd.DataFrame = None,
                       thermo_data: Dict[str, pd.DataFrame] = None) -> Dict:
        """
        每日自动循环
        
        Args:
            market_data: 当日市场数据
            price_data: 历史价格数据（用于回测验证）
            thermo_data: 历史热力学数据
            
        Returns:
            当日策略报告
        """
        logger.info("="*70)
        logger.info(f"【{datetime.now().strftime('%Y-%m-%d')}】每日策略循环")
        logger.info("="*70)
        
        # 1. 计算热力学状态
        thermo_states = self._compute_thermo_states(market_data)
        
        # 2. 生成交易信号
        current_positions = {}  # 实际应该从持仓记录读取
        signals = self.signal_generator.generate_daily_signals(thermo_states, current_positions)
        
        logger.info(f"生成 {len(signals)} 个交易信号")
        
        # 3. 回测验证（如果有历史数据）
        backtest_result = None
        if price_data is not None and thermo_data is not None:
            logger.info("运行回测验证...")
            backtest_result = self._validate_signals(signals, price_data, thermo_data)
            
            # 4. 评估信号质量
            quality_score = self._evaluate_signal_quality(signals, backtest_result)
            
            # 5. 如果质量下降，触发优化
            if quality_score < 0.6:
                logger.warning("信号质量下降，触发参数优化...")
                self._optimize_strategy(price_data, thermo_data)
            
            # 6. 记录回测结果
            self._record_backtest(backtest_result, quality_score)
        
        # 7. 生成策略报告
        report = self._generate_daily_report(signals, backtest_result)
        
        # 8. 保存报告
        report_file = self.data_dir / f"report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"策略报告已保存: {report_file}")
        
        return report
    
    def run_weekly_optimization(self,
                               price_data: pd.DataFrame,
                               thermo_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        每周策略优化
        
        基于最近一周的回测结果，优化策略参数
        """
        logger.info("="*70)
        logger.info("【每周优化】策略参数调优")
        logger.info("="*70)
        
        # 1. 检查是否需要优化
        recent_results = self.backtest_history[-5:]  # 最近5天
        if not recent_results:
            logger.info("回测数据不足，跳过优化")
            return {}
        
        avg_quality = np.mean([r.get('quality_score', 0) for r in recent_results])
        logger.info(f"近期平均信号质量: {avg_quality:.2f}")
        
        # 2. 如果质量持续下降，进行深度优化
        if avg_quality < 0.5:
            logger.warning("信号质量持续下降，执行深度优化...")
            
            # 使用最近60天数据优化
            recent_price = price_data.tail(60)
            recent_thermo = {k: v.tail(60) for k, v in thermo_data.items()}
            
            best_params = self.optimizer.optimize(
                recent_price, 
                recent_thermo,
                n_trials=15
            )
            
            # 3. 验证新参数
            config = BacktestConfig(
                stop_loss=best_params.stop_loss,
                take_profit=best_params.take_profit,
                min_confidence=best_params.min_confidence,
                max_single_position=best_params.max_single_position
            )
            
            engine = AutoBacktestEngine(config)
            validation_result = engine.run(recent_price, recent_thermo)
            
            # 4. 如果新参数表现更好，应用
            old_sharpe = np.mean([r.get('sharpe', 0) for r in recent_results])
            new_sharpe = validation_result['sharpe_ratio']
            
            if new_sharpe > old_sharpe * 1.1:  # 提升10%以上
                self.current_params = best_params
                logger.success(f"策略参数已更新！夏普提升: {old_sharpe:.2f} -> {new_sharpe:.2f}")
            else:
                logger.info("新参数提升不显著，保持当前参数")
        
        # 5. 生成优化报告
        report = {
            'date': datetime.now().isoformat(),
            'avg_quality': avg_quality,
            'current_params': {
                'stop_loss': self.current_params.stop_loss,
                'take_profit': self.current_params.take_profit,
                'min_confidence': self.current_params.min_confidence,
                'max_single_position': self.current_params.max_single_position
            },
            'optimization_applied': avg_quality < 0.5
        }
        
        return report
    
    def run_full_backtest(self,
                         price_data: pd.DataFrame,
                         thermo_data: Dict[str, pd.DataFrame],
                         start_date: str = None,
                         end_date: str = None) -> Dict:
        """
        运行完整回测（用于策略验证）
        
        Args:
            price_data: 完整价格数据
            thermo_data: 完整热力学数据
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            完整回测报告
        """
        if start_date:
            price_data = price_data[price_data.index >= start_date]
        if end_date:
            price_data = price_data[price_data.index <= end_date]
        
        logger.info("="*70)
        logger.info(f"运行完整回测: {price_data.index[0]} ~ {price_data.index[-1]}")
        logger.info("="*70)
        
        # 1. 使用当前参数回测
        config = BacktestConfig(
            stop_loss=self.current_params.stop_loss,
            take_profit=self.current_params.take_profit,
            min_confidence=self.current_params.min_confidence,
            max_single_position=self.current_params.max_single_position
        )
        
        engine = AutoBacktestEngine(config)
        result = engine.run(price_data, thermo_data)
        
        # 2. Walk-Forward验证
        wf_result = engine.run_walk_forward(price_data, thermo_data, train_size=60, test_size=20)
        
        # 3. 生成报告
        report = {
            'backtest_period': f"{price_data.index[0]} ~ {price_data.index[-1]}",
            'total_days': len(price_data),
            'performance': {
                'total_return': result['total_return'],
                'annualized_return': result['annualized_return'],
                'sharpe_ratio': result['sharpe_ratio'],
                'max_drawdown': result['max_drawdown'],
                'volatility': result['volatility']
            },
            'walk_forward': {
                'avg_sharpe': wf_result['avg_sharpe'],
                'avg_return': wf_result['avg_return'],
                'consistency': wf_result['consistency'],
                'total_windows': wf_result['total_windows']
            },
            'trades': {
                'total': result['total_trades'],
                'avg_confidence': result['avg_confidence'],
                'total_cost': result['total_cost']
            },
            'signals': result.get('signal_quality', {})
        }
        
        # 4. 保存报告
        report_file = self.data_dir / f"full_backtest_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"完整回测完成: {report_file}")
        logger.info(f"总收益率: {report['performance']['total_return']:.2%}")
        logger.info(f"夏普比率: {report['performance']['sharpe_ratio']:.2f}")
        logger.info(f"最大回撤: {report['performance']['max_drawdown']:.2%}")
        
        return report
    
    def _validate_signals(self, 
                         signals: List,
                         price_data: pd.DataFrame,
                         thermo_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        验证信号有效性
        
        使用最近30天数据快速回测验证
        """
        # 使用最近30天数据
        recent_price = price_data.tail(30)
        recent_thermo = {k: v.tail(30) for k, v in thermo_data.items()}
        
        config = BacktestConfig(
            stop_loss=self.current_params.stop_loss,
            take_profit=self.current_params.take_profit,
            min_confidence=self.current_params.min_confidence
        )
        
        engine = AutoBacktestEngine(config)
        result = engine.run(recent_price, recent_thermo)
        
        return result
    
    def _evaluate_signal_quality(self, signals: List, backtest_result: Dict) -> float:
        """
        评估信号质量
        
        综合评分：
        - 回测夏普比率 (40%)
        - 信号置信度 (30%)
        - 收益回撤比 (30%)
        """
        if not backtest_result:
            return 0.5
        
        # 夏普评分
        sharpe = backtest_result.get('sharpe_ratio', 0)
        sharpe_score = min(max(sharpe / 2.0, 0), 1.0)  # 夏普2.0为满分
        
        # 置信度评分
        if signals:
            avg_confidence = np.mean([s.confidence for s in signals])
            confidence_score = avg_confidence
        else:
            confidence_score = 0
        
        # 收益回撤比
        total_return = backtest_result.get('total_return', 0)
        max_dd = abs(backtest_result.get('max_drawdown', 0.01))
        return_dd_ratio = total_return / max_dd if max_dd > 0 else 0
        dd_score = min(max(return_dd_ratio / 2.0, 0), 1.0)
        
        # 加权综合
        quality = sharpe_score * 0.4 + confidence_score * 0.3 + dd_score * 0.3
        
        return quality
    
    def _optimize_strategy(self, 
                          price_data: pd.DataFrame,
                          thermo_data: Dict[str, pd.DataFrame]):
        """优化策略参数"""
        # 使用最近30天数据快速优化
        recent_price = price_data.tail(30)
        recent_thermo = {k: v.tail(30) for k, v in thermo_data.items()}
        
        best_params = self.optimizer.optimize(recent_price, recent_thermo, n_trials=8)
        
        # 应用最优参数
        self.current_params = best_params
        
        logger.info("策略参数已优化:")
        logger.info(f"  止损: {best_params.stop_loss:.0%}")
        logger.info(f"  止盈: {best_params.take_profit:.0%}")
        logger.info(f"  最低置信度: {best_params.min_confidence:.0%}")
    
    def _record_backtest(self, result: Dict, quality: float):
        """记录回测结果"""
        self.backtest_history.append({
            'date': datetime.now().isoformat(),
            'sharpe': result.get('sharpe_ratio', 0),
            'return': result.get('total_return', 0),
            'max_dd': result.get('max_drawdown', 0),
            'quality_score': quality,
            'trades': result.get('total_trades', 0)
        })
        
        self._save_history()
    
    def _compute_thermo_states(self, data: Dict) -> Dict[str, Dict]:
        """计算热力学状态"""
        return data.get('thermo_states', {})
    
    def _generate_daily_report(self, 
                              signals: List,
                              backtest_result: Optional[Dict]) -> Dict:
        """生成每日策略报告"""
        report = {
            'date': datetime.now().isoformat(),
            'signals': []
        }
        
        # 信号详情
        for signal in signals[:5]:  # 只取前5个
            report['signals'].append({
                'symbol': signal.symbol,
                'action': signal.action.value,
                'confidence': signal.confidence,
                'reasoning': signal.reasoning
            })
        
        # 回测结果
        if backtest_result:
            report['backtest'] = {
                'sharpe_ratio': backtest_result.get('sharpe_ratio', 0),
                'total_return': backtest_result.get('total_return', 0),
                'max_drawdown': backtest_result.get('max_drawdown', 0),
                'signal_count': backtest_result.get('signal_count', 0)
            }
        
        # 当前参数
        report['current_params'] = {
            'stop_loss': self.current_params.stop_loss,
            'take_profit': self.current_params.take_profit,
            'min_confidence': self.current_params.min_confidence
        }
        
        return report
    
    def get_performance_summary(self) -> Dict:
        """获取绩效摘要"""
        if not self.backtest_history:
            return {}
        
        recent = self.backtest_history[-20:]  # 最近20天
        
        return {
            'avg_sharpe': np.mean([r['sharpe'] for r in recent]),
            'avg_return': np.mean([r['return'] for r in recent]),
            'avg_quality': np.mean([r['quality_score'] for r in recent]),
            'total_trades': sum([r['trades'] for r in recent]),
            'optimization_count': len([r for r in recent if r.get('optimized', False)])
        }


def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ThermoSys 全自动策略进化系统')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['daily', 'weekly', 'full', 'demo'],
                       help='运行模式')
    parser.add_argument('--days', type=int, default=60,
                       help='回测天数（用于demo模式）')
    
    args = parser.parse_args()
    
    loop = AutoEvolutionLoop()
    
    if args.mode == 'demo':
        # 生成模拟数据进行演示
        print("\n生成模拟数据...")
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=args.days, freq='B')
        n = len(dates)
        
        stocks = ['000001', '000002', '600000']
        price_data = pd.DataFrame(index=dates)
        
        for stock in stocks:
            returns = np.random.randn(n) * 0.02
            price_data[stock] = 100 * (1 + returns).cumprod()
        
        thermo_data = {}
        for stock in stocks:
            thermo_data[stock] = pd.DataFrame({
                'rsi': np.random.randn(n),
                'coherence': np.random.rand(n),
                'clarity': np.random.rand(n),
                'entropy': np.random.rand(n),
                'ipv': np.random.randn(n) * 2
            }, index=dates)
        
        # 模拟3天的每日循环
        print("\n运行3天策略循环...")
        for i in range(3):
            day_data = {
                'thermo_states': {
                    stock: thermo_data[stock].iloc[-(3-i)].to_dict()
                    for stock in stocks
                }
            }
            
            report = loop.run_daily_cycle(day_data, price_data, thermo_data)
            print(f"\nDay {i+1} 报告:")
            print(f"  信号数量: {len(report['signals'])}")
            if 'backtest' in report:
                print(f"  回测夏普: {report['backtest']['sharpe_ratio']:.2f}")
        
        # 运行每周优化
        print("\n运行每周优化...")
        weekly_report = loop.run_weekly_optimization(price_data, thermo_data)
        print(f"优化应用: {weekly_report.get('optimization_applied', False)}")
        
        # 运行完整回测
        print("\n运行完整回测...")
        full_report = loop.run_full_backtest(price_data, thermo_data)
        print(f"\n回测结果:")
        print(f"  总收益: {full_report['performance']['total_return']:.2%}")
        print(f"  夏普: {full_report['performance']['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {full_report['performance']['max_drawdown']:.2%}")
        
        # 输出绩效摘要
        print("\n绩效摘要:")
        summary = loop.get_performance_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    elif args.mode == 'daily':
        print("运行每日策略循环...")
        # 实际运行时从数据源获取数据
        print("注意：需要传入实际市场数据")
    
    elif args.mode == 'weekly':
        print("运行每周优化...")
        print("注意：需要传入历史数据")
    
    elif args.mode == 'full':
        print("运行完整回测...")
        print("注意：需要传入完整历史数据")


if __name__ == '__main__':
    main()
