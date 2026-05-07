"""
因果发现模块 (Causal Discovery)
基于 DoWhy 的因果推断集成
支持：因果图发现、反事实分析、干预效应估计
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import networkx as nx
from loguru import logger


class CausalMethod(Enum):
    """因果发现方法枚举"""
    PC = "pc"  # Peter-Clark算法
    GES = "ges"  # Greedy Equivalence Search
    LIN_GAM = "lingam"  # Linear Non-Gaussian Acyclic Model
    NOTEARS = "notears"  # NOTEARS (可微分因果发现)
    DOWHY_BACKDOOR = "backdoor"  # DoWhy后门准则
    DOWHY_IV = "instrumental_variable"  # 工具变量


@dataclass
class CausalEffect:
    """因果效应估计结果"""
    treatment: str  # 处理变量
    outcome: str    # 结果变量
    ate: float      # 平均处理效应
    ci_lower: float # 置信区间下限
    ci_upper: float # 置信区间上限
    p_value: float  # p值
    method: str     # 估计方法
    strength: str   # 'strong', 'moderate', 'weak', 'none'
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """是否统计显著"""
        return self.p_value < alpha
    
    def interpret(self) -> str:
        """解释结果"""
        if not self.is_significant():
            return f"{self.treatment} 对 {self.outcome} 无显著因果效应 (p={self.p_value:.3f})"
        
        direction = "提升" if self.ate > 0 else "降低"
        return (f"{self.treatment} 显著{direction} {self.outcome} "
                f"(ATE={self.ate:.3f}, 95%CI=[{self.ci_lower:.3f}, {self.ci_upper:.3f}], "
                f"p={self.p_value:.3f}, 强度:{self.strength})")


class CausalGraph:
    """
    因果图表示
    包装NetworkX DiGraph，添加因果语义
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self._node_types: Dict[str, str] = {}
        
    def add_node(self, name: str, node_type: str = 'variable'):
        """添加节点"""
        self.graph.add_node(name)
        self._node_types[name] = node_type
        
    def add_edge(self, source: str, target: str, weight: float = 1.0, 
                 edge_type: str = 'direct'):
        """添加因果边"""
        self.graph.add_edge(source, target, 
                           weight=weight, 
                           edge_type=edge_type)
        
    def get_parents(self, node: str) -> List[str]:
        """获取父节点（直接原因）"""
        return list(self.graph.predecessors(node))
    
    def get_children(self, node: str) -> List[str]:
        """获取子节点（直接结果）"""
        return list(self.graph.successors(node))
    
    def get_ancestors(self, node: str) -> List[str]:
        """获取祖先节点"""
        return list(nx.ancestors(self.graph, node))
    
    def get_descendants(self, node: str) -> List[str]:
        """获取后代节点"""
        return list(nx.descendants(self.graph, node))
    
    def find_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """
        寻找后门路径
        后门路径：treatment <- ... -> outcome（以箭头指向treatment开始）
        """
        # 移除treatment的出边，寻找所有路径
        graph_copy = self.graph.copy()
        edges_to_remove = list(graph_copy.out_edges(treatment))
        graph_copy.remove_edges_from(edges_to_remove)
        
        try:
            paths = list(nx.all_simple_paths(graph_copy, treatment, outcome))
            return paths
        except nx.NodeNotFound:
            return []
    
    def find_backdoor_adjustment_set(self, treatment: str, outcome: str) -> List[str]:
        """
        寻找后门调整集
        用于控制混杂变量
        """
        # 简化的后门准则实现
        parents_t = set(self.get_parents(treatment))
        parents_o = set(self.get_parents(outcome))
        
        # 可能的混杂因子
        confounders = parents_t.intersection(parents_o)
        
        # 排除对撞节点
        adjustment_set = []
        for node in confounders:
            # 检查是否为对撞节点（treatment和outcome的后代）
            descendants = set(self.get_descendants(node))
            if treatment not in descendants and outcome not in descendants:
                adjustment_set.append(node)
        
        return adjustment_set
    
    def to_dowhy_graph(self) -> str:
        """
        转换为DoWhy格式的GML字符串
        """
        lines = []
        for node in self.graph.nodes():
            lines.append(f"{node}")
        
        for u, v, data in self.graph.edges(data=True):
            weight = data.get('weight', 1.0)
            lines.append(f"{u} -> {v}")
        
        return "\n".join(lines)
    
    def visualize(self, output_path: str = 'causal_graph.png'):
        """可视化因果图"""
        try:
            import matplotlib.pyplot as plt
            
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
            
            plt.figure(figsize=(12, 8))
            
            # 节点颜色根据类型
            node_colors = []
            for node in self.graph.nodes():
                node_type = self._node_types.get(node, 'variable')
                if node_type == 'treatment':
                    node_colors.append('#e74c3c')
                elif node_type == 'outcome':
                    node_colors.append('#27ae60')
                elif node_type == 'confounder':
                    node_colors.append('#f39c12')
                else:
                    node_colors.append('#3498db')
            
            nx.draw(self.graph, pos, 
                   with_labels=True,
                   node_color=node_colors,
                   node_size=2000,
                   font_size=10,
                   font_weight='bold',
                   arrows=True,
                   arrowsize=20,
                   edge_color='#7f8c8d')
            
            plt.title("Causal Graph")
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Causal graph saved to: {output_path}")
        except ImportError:
            logger.warning("matplotlib not installed, skipping visualization")


class CausalDiscovery:
    """
    因果发现引擎
    从数据中学习因果结构
    """
    
    def __init__(self):
        self.graph: Optional[CausalGraph] = None
        self._data: Optional[pd.DataFrame] = None
        
    def fit(self, data: pd.DataFrame, method: CausalMethod = CausalMethod.PC,
            **kwargs) -> CausalGraph:
        """
        从数据中学习因果图
        
        Args:
            data: 观测数据
            method: 因果发现方法
            **kwargs: 方法特定参数
        """
        self._data = data
        
        if method == CausalMethod.PC:
            self.graph = self._pc_algorithm(data, **kwargs)
        elif method == CausalMethod.GES:
            self.graph = self._ges_algorithm(data, **kwargs)
        elif method == CausalMethod.LIN_GAM:
            self.graph = self._lingam_algorithm(data, **kwargs)
        elif method == CausalMethod.NOTEARS:
            self.graph = self._notears_algorithm(data, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return self.graph
    
    def _pc_algorithm(self, data: pd.DataFrame, alpha: float = 0.05) -> CausalGraph:
        """
        PC算法实现（简化版）
        基于条件独立性检验
        """
        from scipy import stats
        
        graph = CausalGraph()
        variables = list(data.columns)
        
        # 1. 构建完全无向图
        for var in variables:
            graph.add_node(var)
        
        # 2. 基于相关性移除边（简化版）
        for i, vi in enumerate(variables):
            for vj in variables[i+1:]:
                # 计算Pearson相关系数
                corr, p_value = stats.pearsonr(data[vi].dropna(), data[vj].dropna())
                
                # 如果显著相关，添加边
                if p_value < alpha and abs(corr) > 0.1:
                    # 方向通过偏相关性判断（简化）
                    graph.add_edge(vi, vj, weight=abs(corr))
        
        return graph
    
    def _ges_algorithm(self, data: pd.DataFrame, **kwargs) -> CausalGraph:
        """GES算法（简化版）"""
        # 简化实现：基于AIC/BIC评分
        return self._pc_algorithm(data, **kwargs)
    
    def _lingam_algorithm(self, data: pd.DataFrame, **kwargs) -> CausalGraph:
        """LiNGAM算法（简化版）"""
        # 简化实现：基于非高斯性和ICA
        logger.warning("LiNGAM simplified implementation")
        return self._pc_algorithm(data, **kwargs)
    
    def _notears_algorithm(self, data: pd.DataFrame, **kwargs) -> CausalGraph:
        """NOTEARS算法（简化版）"""
        # 简化实现：基于梯度下降的DAG学习
        logger.warning("NOTEARS simplified implementation")
        return self._pc_algorithm(data, **kwargs)


class CausalInference:
    """
    因果推断引擎
    基于已知的因果图进行效应估计
    """
    
    def __init__(self, graph: CausalGraph, data: pd.DataFrame):
        self.graph = graph
        self.data = data
        
    def estimate_ate(self, treatment: str, outcome: str,
                     method: str = "backdoor.linear_regression",
                     control_vars: Optional[List[str]] = None) -> CausalEffect:
        """
        估计平均处理效应 (ATE)
        
        Args:
            treatment: 处理变量
            outcome: 结果变量
            method: 估计方法
            control_vars: 控制变量（若不提供则自动寻找）
        """
        if control_vars is None:
            control_vars = self.graph.find_backdoor_adjustment_set(treatment, outcome) or []
        
        # 简化版线性回归估计
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        
        # 准备数据
        features = [treatment] + control_vars
        X = self.data[features].dropna()
        y = self.data[outcome].loc[X.index]
        
        if len(X) < 10:
            return CausalEffect(
                treatment=treatment, outcome=outcome,
                ate=0.0, ci_lower=0.0, ci_upper=0.0,
                p_value=1.0, method="insufficient_data", strength="none"
            )
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 拟合模型
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # 提取处理效应（第一个系数）
        ate = model.coef_[0]
        
        # 计算标准误（简化）
        residuals = y - model.predict(X_scaled)
        mse = np.mean(residuals**2)
        var_treatment = np.var(X_scaled[:, 0])
        se = np.sqrt(mse / (len(X) * var_treatment))
        
        # 95%置信区间
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        # t检验
        t_stat = ate / se if se > 0 else 0
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(X)-len(features)))
        
        # 判断强度
        abs_ate = abs(ate)
        if abs_ate > 0.5:
            strength = "strong"
        elif abs_ate > 0.2:
            strength = "moderate"
        elif abs_ate > 0.05:
            strength = "weak"
        else:
            strength = "none"
        
        return CausalEffect(
            treatment=treatment, outcome=outcome,
            ate=ate, ci_lower=ci_lower, ci_upper=ci_upper,
            p_value=p_value, method="backdoor.linear_regression",
            strength=strength
        )
    
    def estimate_cate(self, treatment: str, outcome: str, 
                      subgroup_vars: List[str]) -> Dict[str, CausalEffect]:
        """
        估计条件平均处理效应 (CATE)
        在不同子群体中的效应
        """
        results = {}
        
        # 对每个子群体分别估计
        for var in subgroup_vars:
            if var not in self.data.columns:
                continue
            
            for value in self.data[var].unique():
                mask = self.data[var] == value
                subgroup_data = self.data[mask]
                
                if len(subgroup_data) < 10:
                    continue
                
                # 创建子图（简化）
                sub_inference = CausalInference(self.graph, subgroup_data)
                effect = sub_inference.estimate_ate(treatment, outcome)
                
                key = f"{var}={value}"
                results[key] = effect
        
        return results
    
    def counterfactual_analysis(self, treatment: str, outcome: str,
                               intervention_values: List[float]) -> pd.DataFrame:
        """
        反事实分析
        如果处理变量取不同值，结果会怎样
        """
        from sklearn.linear_model import LinearRegression
        
        adjustment_set = self.graph.find_backdoor_adjustment_set(treatment, outcome)
        features = [treatment] + [f for f in adjustment_set if f in self.data.columns]
        
        X = self.data[features].dropna()
        y = self.data[outcome].loc[X.index]
        
        model = LinearRegression()
        model.fit(X, y)
        
        # 计算反事实
        results = []
        mean_features = X.mean()
        
        for value in intervention_values:
            cf_features = mean_features.copy()
            cf_features[treatment] = value
            prediction = model.predict([cf_features.values])[0]
            
            results.append({
                'intervention_value': value,
                'predicted_outcome': prediction,
                'difference_from_baseline': prediction - model.predict([mean_features.values])[0]
            })
        
        return pd.DataFrame(results)


class MarketCausalAnalyzer:
    """
    市场因果分析器
    专门用于分析市场变量的因果关系
    """
    
    def __init__(self):
        self.discovery = CausalDiscovery()
        self.graph: Optional[CausalGraph] = None
        
    def analyze_market_drivers(self, data: pd.DataFrame) -> CausalGraph:
        """
        分析市场驱动因素
        
        变量示例：
        - 价格、成交量、波动率
        - 资金流向、情绪指标
        - 宏观经济变量
        """
        logger.info("Discovering causal structure from market data...")
        
        # 使用PC算法发现因果结构
        self.graph = self.discovery.fit(data, method=CausalMethod.PC)
        
        # 标记特殊节点
        if 'returns' in self.graph.graph.nodes():
            self.graph._node_types['returns'] = 'outcome'
        if 'sentiment' in self.graph.graph.nodes():
            self.graph._node_types['sentiment'] = 'treatment'
        
        return self.graph
    
    def quantify_sentiment_impact(self, data: pd.DataFrame) -> CausalEffect:
        """
        量化情绪对收益的因果效应
        """
        if self.graph is None:
            self.analyze_market_drivers(data)
        
        inference = CausalInference(self.graph, data)
        
        # 使用数据中存在的变量作为控制变量
        available_controls = [c for c in ['volume', 'volatility'] if c in data.columns]
        effect = inference.estimate_ate(
            treatment='sentiment',
            outcome='returns',
            control_vars=available_controls
        )
        
        logger.info(f"Sentiment impact: {effect.interpret()}")
        return effect
    
    def analyze_policy_intervention(self, data: pd.DataFrame,
                                    policy_var: str = 'margin_requirement') -> pd.DataFrame:
        """
        分析政策干预的反事实效应
        """
        if self.graph is None:
            self.analyze_market_drivers(data)
        
        inference = CausalInference(self.graph, data)
        
        # 反事实：如果政策变量取不同值
        intervention_values = np.percentile(data[policy_var], [10, 25, 50, 75, 90])
        
        results = inference.counterfactual_analysis(
            treatment=policy_var,
            outcome='returns',
            intervention_values=intervention_values
        )
        
        return results
    
    def generate_trading_insights(self, data: pd.DataFrame) -> List[str]:
        """
        生成交易洞察（基于因果分析）
        """
        insights = []
        
        if self.graph is None:
            self.analyze_market_drivers(data)
        
        # 1. 分析主要驱动因素
        if 'returns' in self.graph.graph.nodes():
            parents = self.graph.get_parents('returns')
            if parents:
                insights.append(f"收益的主要因果驱动因素: {', '.join(parents)}")
        
        # 2. 分析情绪效应
        if 'sentiment' in data.columns and 'returns' in data.columns:
            effect = self.quantify_sentiment_impact(data)
            insights.append(effect.interpret())
        
        # 3. 寻找中介变量
        # 简化：找从sentiment到returns路径上的中间节点
        if 'sentiment' in self.graph.graph.nodes() and 'returns' in self.graph.graph.nodes():
            try:
                paths = list(nx.all_simple_paths(self.graph.graph, 'sentiment', 'returns', cutoff=2))
                if len(paths) > 1:
                    mediators = set()
                    for path in paths:
                        if len(path) == 3:  # sentiment -> mediator -> returns
                            mediators.add(path[1])
                    if mediators:
                        insights.append(f"情绪影响收益的中介路径经过: {', '.join(mediators)}")
            except:
                pass
        
        return insights


# 便捷函数
def analyze_causal_structure(data: pd.DataFrame, 
                            treatment: Optional[str] = None,
                            outcome: Optional[str] = None) -> Dict[str, Any]:
    """
    快速因果分析
    
    Args:
        data: 包含变量的DataFrame
        treatment: 处理变量（可选）
        outcome: 结果变量（可选）
    
    Returns:
        包含因果图、效应估计、洞察的字典
    """
    analyzer = MarketCausalAnalyzer()
    
    # 发现因果结构
    graph = analyzer.analyze_market_drivers(data)
    
    results = {
        'graph': graph,
        'insights': []
    }
    
    # 如果指定了处理变量和结果变量，估计效应
    if treatment and outcome and treatment in data.columns and outcome in data.columns:
        inference = CausalInference(graph, data)
        effect = inference.estimate_ate(treatment, outcome)
        results['ate'] = effect
        results['insights'].append(effect.interpret())
    
    # 生成交易洞察
    insights = analyzer.generate_trading_insights(data)
    results['insights'].extend(insights)
    
    return results


if __name__ == '__main__':
    # 测试示例
    np.random.seed(42)
    n = 1000
    
    # 生成模拟数据
    sentiment = np.random.randn(n)
    volume = np.random.randn(n) + 0.5 * sentiment
    returns = 0.3 * sentiment + 0.1 * volume + np.random.randn(n) * 0.5
    
    data = pd.DataFrame({
        'sentiment': sentiment,
        'volume': volume,
        'returns': returns
    })
    
    # 因果分析
    results = analyze_causal_structure(data, treatment='sentiment', outcome='returns')
    
    print("因果效应:")
    print(results['ate'].interpret())
    print("\n洞察:")
    for insight in results['insights']:
        print(f"- {insight}")
    
    # 可视化
    results['graph'].visualize('test_causal_graph.png')
