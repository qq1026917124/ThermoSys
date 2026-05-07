"""
策略参数优化器
基于回测结果自动搜索最优参数
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger
import itertools

from thermo_sys.execution.auto_backtest import AutoBacktestEngine, BacktestConfig


@dataclass
class StrategyParameters:
    """策略参数空间"""
    stop_loss: float = -0.07
    take_profit: float = 0.15
    min_confidence: float = 0.6
    max_single_position: float = 0.3
    rsi_buy_threshold: float = -2.0
    rsi_sell_threshold: float = 2.0
    coherence_buy: float = 0.6
    coherence_sell: float = 0.3
    clarity_threshold: float = 0.3


class StrategyOptimizer:
    """
    策略参数优化器
    
    使用网格搜索寻找最优参数组合：
    1. 定义参数搜索空间
    2. 对每个参数组合运行回测
    3. 根据夏普比率、回撤等指标评分
    4. 返回最优参数
    """
    
    def __init__(self):
        self.best_params: StrategyParameters = None
        self.best_score: float = -np.inf
        self.optimization_history: List[Dict] = []
    
    def optimize(self,
                price_data: pd.DataFrame,
                thermo_data: Dict[str, pd.DataFrame],
                param_grid: Dict[str, List] = None,
                metric: str = 'sharpe_ratio',
                n_trials: int = 20) -> StrategyParameters:
        """
        优化策略参数
        
        Args:
            price_data: 价格数据
            thermo_data: 热力学数据
            param_grid: 参数搜索网格，None则使用默认
            metric: 优化目标指标
            n_trials: 最大尝试次数
            
        Returns:
            最优参数
        """
        if param_grid is None:
            param_grid = self._default_param_grid()
        
        logger.info(f"开始策略参数优化，目标指标: {metric}")
        logger.info(f"参数空间: {len(list(itertools.product(*param_grid.values())))} 种组合")
        
        # 生成参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # 随机采样（避免组合爆炸）
        all_combinations = list(itertools.product(*param_values))
        if len(all_combinations) > n_trials:
            np.random.shuffle(all_combinations)
            combinations = all_combinations[:n_trials]
        else:
            combinations = all_combinations
        
        # 测试每个组合
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            logger.info(f"\n测试组合 {i+1}/{len(combinations)}: {params}")
            
            # 运行回测
            score, result = self._evaluate_params(params, price_data, thermo_data, metric)
            
            # 记录结果
            self.optimization_history.append({
                'trial': i + 1,
                'params': params,
                'score': score,
                'sharpe': result.get('sharpe_ratio', 0),
                'return': result.get('total_return', 0),
                'max_dd': result.get('max_drawdown', 0)
            })
            
            # 更新最优
            if score > self.best_score:
                self.best_score = score
                self.best_params = StrategyParameters(**params)
                logger.success(f"发现更优参数！得分: {score:.3f}")
        
        logger.info(f"\n优化完成！最优得分: {self.best_score:.3f}")
        logger.info(f"最优参数: {self.best_params}")
        
        return self.best_params
    
    def _evaluate_params(self, 
                        params: Dict, 
                        price_data: pd.DataFrame,
                        thermo_data: Dict[str, pd.DataFrame],
                        metric: str) -> Tuple[float, Dict]:
        """
        评估一组参数
        
        Returns:
            (得分, 回测结果)
        """
        # 创建配置
        config = BacktestConfig(**{k: v for k, v in params.items() 
                                  if k in ['stop_loss', 'take_profit', 'min_confidence', 'max_single_position']})
        
        # 运行回测
        engine = AutoBacktestEngine(config)
        result = engine.run(price_data, thermo_data)
        
        # 计算得分（多目标优化）
        score = self._calculate_score(result, metric)
        
        return score, result
    
    def _calculate_score(self, result: Dict, metric: str) -> float:
        """
        计算综合得分
        
        综合考量：
        - 夏普比率（主要）
        - 收益率
        - 最大回撤（惩罚）
        - 交易次数（适度惩罚过度交易）
        """
        sharpe = result.get('sharpe_ratio', 0)
        total_return = result.get('total_return', 0)
        max_dd = abs(result.get('max_drawdown', 0))
        n_trades = result.get('total_trades', 0)
        
        # 基础得分
        if metric == 'sharpe_ratio':
            base_score = sharpe
        elif metric == 'total_return':
            base_score = total_return * 10  # 放大
        elif metric == 'risk_adjusted':
            base_score = sharpe * 0.6 + total_return * 2 - max_dd * 2
        else:
            base_score = sharpe
        
        # 惩罚项
        # 1. 回撤过大
        dd_penalty = max(0, max_dd - 0.15) * 5  # 回撤超过15%重罚
        
        # 2. 交易过于频繁（假设每天超过2次算频繁）
        n_days = len(result.get('equity_curve', []))
        if n_days > 0:
            trades_per_day = n_trades / n_days
            trade_penalty = max(0, trades_per_day - 2) * 0.5
        else:
            trade_penalty = 0
        
        # 3. 收益为负
        return_penalty = abs(min(0, total_return)) * 3
        
        score = base_score - dd_penalty - trade_penalty - return_penalty
        
        return score
    
    def _default_param_grid(self) -> Dict[str, List]:
        """默认参数搜索空间"""
        return {
            'stop_loss': [-0.05, -0.07, -0.10],
            'take_profit': [0.10, 0.15, 0.20],
            'min_confidence': [0.5, 0.6, 0.7],
            'max_single_position': [0.2, 0.3, 0.4]
        }
    
    def get_optimization_report(self) -> pd.DataFrame:
        """生成优化报告"""
        if not self.optimization_history:
            return pd.DataFrame()
        
        records = []
        for item in self.optimization_history:
            record = {
                'trial': item['trial'],
                'score': item['score'],
                'sharpe': item['sharpe'],
                'return': item['return'],
                'max_dd': item['max_dd']
            }
            record.update(item['params'])
            records.append(record)
        
        df = pd.DataFrame(records)
        return df.sort_values('score', ascending=False)
    
    def analyze_param_importance(self) -> Dict[str, float]:
        """
        分析参数重要性
        
        使用简单相关性分析
        """
        if not self.optimization_history:
            return {}
        
        df = self.get_optimization_report()
        if df.empty:
            return {}
        
        importance = {}
        param_cols = [c for c in df.columns if c not in ['trial', 'score', 'sharpe', 'return', 'max_dd']]
        
        for param in param_cols:
            corr = df[param].corr(df['score'])
            importance[param] = abs(corr) if not np.isnan(corr) else 0
        
        # 归一化
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance


if __name__ == '__main__':
    # 测试优化器
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='B')
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
    
    optimizer = StrategyOptimizer()
    best_params = optimizer.optimize(price_data, thermo_data, n_trials=10)
    
    print("\n" + "="*60)
    print("优化完成")
    print("="*60)
    print(f"最优参数:")
    print(f"  止损: {best_params.stop_loss:.0%}")
    print(f"  止盈: {best_params.take_profit:.0%}")
    print(f"  最低置信度: {best_params.min_confidence:.0%}")
    print(f"  最大单票仓位: {best_params.max_single_position:.0%}")
    
    print(f"\n参数重要性:")
    importance = optimizer.analyze_param_importance()
    for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {param}: {imp:.1%}")
    
    print(f"\nTop 5 参数组合:")
    report = optimizer.get_optimization_report().head()
    print(report[['score', 'sharpe', 'return', 'max_dd', 'stop_loss', 'take_profit']])
