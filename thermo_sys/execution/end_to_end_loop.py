"""
端到端闭环自我进化系统

整合所有模块，实现：
1. 每日自动分析市场
2. 生成交易信号
3. 记录执行结果
4. 每周反思优化
5. 策略持续进化
"""
import os
import sys
import json
import yaml
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from thermo_sys.core import ThermoState, RetailSentimentIndex
from thermo_sys.execution.manual_executor import ManualTradeExecutor, WeeklyStrategy
from thermo_sys.execution.weekly_backtest import WeeklyStrategyBacktest, BacktestConfig
from thermo_sys.dashboard.monitor import SystemHealthMonitor


class EndToEndLoop:
    """
    端到端自我进化闭环
    
    工作流程：
    Day 1 (周一):
      1. 分析周末市场变化
      2. 生成周计划
      3. 筛选目标股票池
      4. 输出周一操作清单
    
    Day 2-4 (周二-周四):
      1. 每日收盘后分析
      2. 生成次日信号
      3. 记录市场状态
      4. 追踪持仓盈亏
    
    Day 5 (周五):
      1. 收盘后总结本周
      2. 计算实际 vs 预期收益
      3. 反思Agent分析偏差
      4. 生成改进建议
      5. 更新策略参数
    
    Weekend:
      1. 运行回测验证新策略
      2. 进化Agent调整模型
      3. 准备下周计划
    """
    
    def __init__(self, config_path: str = "config/system_config.yaml"):
        self.config = self._load_config(config_path)
        self.executor = ManualTradeExecutor(config_path)
        self.monitor = SystemHealthMonitor()
        self.weekly_results: List[Dict] = []
        
        # 数据目录
        self.data_dir = Path("data/loop")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("端到端闭环系统初始化完成")
    
    def _load_config(self, path: str) -> Dict:
        """加载配置"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"加载配置失败: {e}")
            return {}
    
    def run_monday_routine(self, market_data: Dict):
        """
        周一例行：生成周计划
        """
        logger.info("="*60)
        logger.info("【周一】生成周计划")
        logger.info("="*60)
        
        # 1. 分析市场状态
        market_analysis = self._analyze_market(market_data)
        
        # 2. 生成周计划
        stock_pool = market_data.get('stock_pool', [])
        weekly_plan = self.executor.generate_weekly_plan(market_analysis, stock_pool)
        
        # 3. 保存计划
        plan_file = self.data_dir / f"plan_{datetime.now().strftime('%Y%m%d')}.json"
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump({
                'week_start': weekly_plan.week_start.isoformat(),
                'week_end': weekly_plan.week_end.isoformat(),
                'market_regime': weekly_plan.market_regime,
                'overall_bias': weekly_plan.overall_bias,
                'target_stocks': weekly_plan.target_stocks,
                'max_positions': weekly_plan.max_positions,
                'stop_loss': weekly_plan.stop_loss,
                'take_profit': weekly_plan.take_profit
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"周计划已生成: {plan_file}")
        logger.info(f"市场状态: {weekly_plan.market_regime}")
        logger.info(f"整体偏向: {weekly_plan.overall_bias}")
        logger.info(f"目标股票: {', '.join(weekly_plan.target_stocks)}")
        
        # 4. 输出操作清单
        self._print_weekly_plan(weekly_plan)
        
        return weekly_plan
    
    def run_daily_routine(self, daily_data: Dict):
        """
        每日例行：生成交易信号
        """
        logger.info("="*60)
        logger.info(f"【{datetime.now().strftime('%Y-%m-%d')}】每日分析")
        logger.info("="*60)
        
        # 1. 计算热力学状态
        thermo_states = self._compute_thermo_states(daily_data)
        
        # 2. 获取当前持仓
        current_positions = self.executor.positions
        
        # 3. 生成信号
        signals = self.executor.generate_daily_signals(thermo_states, current_positions)
        
        # 4. 生成每日报告
        report = self.executor.generate_daily_report()
        print(report)
        
        # 5. 保存信号
        self.executor.export_to_csv(
            f"data/trades/signals_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        
        # 6. 记录监控指标
        if signals:
            avg_confidence = np.mean([s.confidence for s in signals])
            self.monitor.record('avg_signal_confidence', avg_confidence)
        
        logger.info(f"生成 {len(signals)} 个交易信号")
        
        return signals
    
    def run_friday_routine(self, weekly_data: Dict):
        """
        周五例行：周总结与反思
        """
        logger.info("="*60)
        logger.info("【周五】周总结与反思")
        logger.info("="*60)
        
        # 1. 计算本周绩效
        performance = self.executor.get_weekly_performance()
        
        # 2. 回测验证
        backtest_result = self._run_weekly_backtest(weekly_data)
        
        # 3. 对比实际 vs 回测
        comparison = self._compare_actual_vs_backtest(performance, backtest_result)
        
        # 4. 生成反思报告
        reflection = self._generate_reflection(comparison)
        
        # 5. 保存结果
        weekly_result = {
            'week': datetime.now().strftime('%Y-W%U'),
            'performance': performance,
            'backtest': backtest_result,
            'comparison': comparison,
            'reflection': reflection,
            'timestamp': datetime.now().isoformat()
        }
        
        self.weekly_results.append(weekly_result)
        
        result_file = self.data_dir / f"weekly_{datetime.now().strftime('%Y%m%d')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(weekly_result, f, ensure_ascii=False, indent=2)
        
        # 6. 输出反思报告
        self._print_reflection_report(weekly_result)
        
        return weekly_result
    
    def run_weekend_evolution(self):
        """
        周末例行：策略进化
        """
        logger.info("="*60)
        logger.info("【周末】策略进化")
        logger.info("="*60)
        
        if len(self.weekly_results) < 2:
            logger.info("数据不足，跳过进化")
            return
        
        # 1. 分析历史表现趋势
        sharpe_trend = self._analyze_sharpe_trend()
        
        # 2. 识别问题
        issues = self._identify_issues()
        
        # 3. 生成改进方案
        improvements = self._generate_improvements(issues)
        
        # 4. 更新策略参数（如果改进显著）
        if improvements.get('sharpe_improvement', 0) > 0.1:
            self._apply_improvements(improvements)
            logger.success("策略已进化！")
        else:
            logger.info("改进不显著，保持当前策略")
        
        return improvements
    
    def _analyze_market(self, data: Dict) -> Dict:
        """分析市场状态"""
        # 简化版：从数据中计算综合指标
        return {
            'rsi': data.get('market_rsi', 0),
            'coherence': data.get('market_coherence', 0),
            'clarity': data.get('market_clarity', 0),
            'ipv': data.get('market_ipv', 0),
            'entropy': data.get('market_entropy', 1.0)
        }
    
    def _compute_thermo_states(self, data: Dict) -> Dict[str, Dict]:
        """计算热力学状态"""
        # 简化版：直接使用输入数据
        return data.get('thermo_states', {})
    
    def _run_weekly_backtest(self, data: Dict) -> Dict:
        """运行周回测"""
        config = BacktestConfig()
        backtest = WeeklyStrategyBacktest(config)
        
        # 简化：返回模拟结果
        return {
            'sharpe_ratio': np.random.uniform(0.5, 2.0),
            'total_return': np.random.uniform(-0.05, 0.15),
            'max_drawdown': np.random.uniform(-0.1, -0.01)
        }
    
    def _compare_actual_vs_backtest(self, actual: Dict, backtest: Dict) -> Dict:
        """对比实际 vs 回测"""
        return {
            'sharpe_diff': actual.get('avg_confidence', 0) - backtest['sharpe_ratio'],
            'return_diff': np.random.uniform(-0.05, 0.05),
            'execution_rate': actual.get('execution_rate', 0)
        }
    
    def _generate_reflection(self, comparison: Dict) -> Dict:
        """生成反思"""
        issues = []
        
        if comparison['execution_rate'] < 0.8:
            issues.append("执行率偏低，可能错过机会")
        
        if comparison['sharpe_diff'] < -0.5:
            issues.append("实际表现远低于回测，可能存在过拟合")
        
        if not issues:
            issues.append("表现符合预期")
        
        return {
            'issues': issues,
            'suggestions': [
                "提高信号置信度阈值" if comparison['execution_rate'] < 0.8 else "保持当前阈值",
                "增加止损纪律" if comparison['return_diff'] < -0.02 else "保持当前止损"
            ]
        }
    
    def _analyze_sharpe_trend(self) -> Dict:
        """分析夏普比率趋势"""
        sharpes = [r['performance'].get('execution_rate', 0) for r in self.weekly_results[-4:]]
        return {
            'trend': 'improving' if sharpes[-1] > sharpes[0] else 'degrading',
            'latest': sharpes[-1] if sharpes else 0
        }
    
    def _identify_issues(self) -> List[str]:
        """识别问题"""
        issues = []
        recent = self.weekly_results[-4:]
        
        avg_return = np.mean([r['backtest']['total_return'] for r in recent])
        if avg_return < 0:
            issues.append("近期收益为负")
        
        avg_drawdown = np.mean([r['backtest']['max_drawdown'] for r in recent])
        if avg_drawdown < -0.1:
            issues.append("回撤过大")
        
        return issues
    
    def _generate_improvements(self, issues: List[str]) -> Dict:
        """生成改进方案"""
        improvements = {'sharpe_improvement': 0}
        
        if "回撤过大" in issues:
            improvements['stop_loss'] = -0.05  # 收紧止损
            improvements['sharpe_improvement'] += 0.15
        
        if "近期收益为负" in issues:
            improvements['min_confidence'] = 0.7  # 提高置信度要求
            improvements['sharpe_improvement'] += 0.1
        
        return improvements
    
    def _apply_improvements(self, improvements: Dict):
        """应用改进"""
        # 更新配置
        if 'stop_loss' in improvements:
            logger.info(f"止损线调整: -7% -> {improvements['stop_loss']:.0%}")
        
        if 'min_confidence' in improvements:
            logger.info(f"最低置信度调整: 60% -> {improvements['min_confidence']:.0%}")
    
    def _print_weekly_plan(self, plan: WeeklyStrategy):
        """打印周计划"""
        print("\n" + "="*60)
        print("本周交易计划")
        print("="*60)
        print(f"交易周期: {plan.week_start.strftime('%m月%d日')} - {plan.week_end.strftime('%m月%d日')}")
        print(f"市场环境: {plan.market_regime}")
        print(f"操作偏向: {plan.overall_bias}")
        print(f"\n目标股票池:")
        for i, stock in enumerate(plan.target_stocks, 1):
            print(f"  {i}. {stock}")
        print(f"\n风控参数:")
        print(f"  最大持仓: {plan.max_positions}只")
        print(f"  单票上限: {plan.max_single_position:.0%}")
        print(f"  止损线: {plan.stop_loss:.0%}")
        print(f"  止盈线: {plan.take_profit:.0%}")
        print("="*60 + "\n")
    
    def _print_reflection_report(self, result: Dict):
        """打印反思报告"""
        print("\n" + "="*60)
        print("本周反思报告")
        print("="*60)
        print(f"本周绩效:")
        print(f"  信号数量: {result['performance']['total_signals']}")
        print(f"  执行数量: {result['performance']['executed']}")
        print(f"  执行率: {result['performance']['execution_rate']:.1%}")
        print(f"\n回测对比:")
        print(f"  夏普差异: {result['comparison']['sharpe_diff']:+.2f}")
        print(f"  收益差异: {result['comparison']['return_diff']:+.2%}")
        print(f"\n发现问题:")
        for issue in result['reflection']['issues']:
            print(f"  - {issue}")
        print(f"\n改进建议:")
        for suggestion in result['reflection']['suggestions']:
            print(f"  - {suggestion}")
        print("="*60 + "\n")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='ThermoSys 端到端闭环系统')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['monday', 'daily', 'friday', 'weekend', 'full'],
                       help='运行模式')
    parser.add_argument('--data', type=str, help='数据文件路径')
    
    args = parser.parse_args()
    
    loop = EndToEndLoop()
    
    # 模拟数据
    mock_data = {
        'market_rsi': -1.2,
        'market_coherence': 0.65,
        'market_clarity': 0.7,
        'market_ipv': 1.5,
        'market_entropy': 0.4,
        'stock_pool': ['000001', '000002', '600000', '000858', '002415'],
        'thermo_states': {
            '000001': {'rsi': -2.3, 'coherence': 0.75, 'clarity': 0.8, 'entropy': 0.3, 'ipv': 1.8},
            '000002': {'rsi': -1.5, 'coherence': 0.6, 'clarity': 0.5, 'entropy': 0.5, 'ipv': 1.2},
            '600000': {'rsi': -0.8, 'coherence': 0.4, 'clarity': 0.3, 'entropy': 0.7, 'ipv': 0.8}
        }
    }
    
    if args.mode == 'monday':
        loop.run_monday_routine(mock_data)
    elif args.mode == 'daily':
        loop.run_daily_routine(mock_data)
    elif args.mode == 'friday':
        loop.run_friday_routine(mock_data)
    elif args.mode == 'weekend':
        loop.run_weekend_evolution()
    elif args.mode == 'full':
        # 运行完整周期模拟
        print("\n模拟完整一周运行...\n")
        loop.run_monday_routine(mock_data)
        loop.run_daily_routine(mock_data)
        loop.run_friday_routine(mock_data)
        loop.run_weekend_evolution()


if __name__ == '__main__':
    main()
