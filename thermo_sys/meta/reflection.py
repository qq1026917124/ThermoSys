"""
反思Agent (Reflection Agent)
交易后分析：对比世界模型预测 vs 实际结果，生成"经验总结"和改进任务
"""
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime


@dataclass
class EpisodeRecord:
    """单条交易记录"""
    timestamp: datetime
    state: Dict[str, float]
    action: float
    predicted_next_state: Dict[str, float]
    true_next_state: Dict[str, float]
    reward: float
    financial_return: float
    world_model_uncertainty: float


class ReflectionAgent:
    """
    反思Agent
    
    功能：
    1. 识别世界模型的系统性偏差
    2. 分析策略亏损根因
    3. 生成改进任务队列
    """
    
    def __init__(
        self,
        llm_endpoint: Optional[str] = None,
        min_episodes: int = 20,
        error_threshold: float = 0.5
    ):
        self.llm_endpoint = llm_endpoint
        self.min_episodes = min_episodes
        self.error_threshold = error_threshold
        self.episode_buffer: List[EpisodeRecord] = []
        
    def add_episode(self, record: EpisodeRecord):
        """添加交易记录到缓冲区"""
        self.episode_buffer.append(record)
        
    def analyze_episode(self, window: int = 20) -> Dict[str, Any]:
        """
        分析最近window个交易日的表现
        
        Returns:
            分析报告字典
        """
        if len(self.episode_buffer) < self.min_episodes:
            return {'status': 'insufficient_data', 'min_required': self.min_episodes}
        
        recent = self.episode_buffer[-window:]
        
        # 1. 世界模型预测误差分析
        prediction_errors = self._analyze_prediction_errors(recent)
        
        # 2. 策略亏损分析
        loss_analysis = self._analyze_losses(recent)
        
        # 3. 热力学状态偏差模式
        thermo_patterns = self._analyze_thermo_patterns(recent)
        
        # 4. 生成根因分析
        root_causes = self._identify_root_causes(
            prediction_errors, loss_analysis, thermo_patterns
        )
        
        return {
            'status': 'success',
            'window': window,
            'prediction_errors': prediction_errors,
            'loss_analysis': loss_analysis,
            'thermo_patterns': thermo_patterns,
            'root_causes': root_causes,
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_prediction_errors(self, episodes: List[EpisodeRecord]) -> Dict[str, Any]:
        """分析预测误差的统计特征"""
        errors = []
        for ep in episodes:
            error = {
                k: ep.predicted_next_state.get(k, 0) - ep.true_next_state.get(k, 0)
                for k in ['ipv', 'clarity', 'coherence', 'entropy', 'rsi']
            }
            errors.append(error)
        
        error_df = pd.DataFrame(errors)
        
        # 计算各维度的统计量
        stats = {}
        for col in error_df.columns:
            stats[col] = {
                'mean_error': float(error_df[col].mean()),
                'std_error': float(error_df[col].std()),
                'max_abs_error': float(error_df[col].abs().max()),
                'bias_direction': 'overestimate' if error_df[col].mean() > 0 else 'underestimate'
            }
        
        # 识别最大偏差维度
        max_bias_dim = max(stats.keys(), key=lambda k: abs(stats[k]['mean_error']))
        
        return {
            'per_dimension': stats,
            'max_bias_dimension': max_bias_dim,
            'overall_rmse': float(np.sqrt((error_df**2).mean().mean()))
        }
    
    def _analyze_losses(self, episodes: List[EpisodeRecord]) -> Dict[str, Any]:
        """分析亏损交易的共同特征"""
        loss_episodes = [ep for ep in episodes if ep.financial_return < 0]
        
        if not loss_episodes:
            return {'has_losses': False}
        
        # 提取亏损时的热力学状态特征
        loss_states = pd.DataFrame([ep.state for ep in loss_episodes])
        profit_states = pd.DataFrame([ep.state for ep in episodes if ep.financial_return > 0])
        
        analysis = {
            'has_losses': True,
            'loss_ratio': len(loss_episodes) / len(episodes),
            'avg_loss': float(np.mean([ep.financial_return for ep in loss_episodes])),
            'loss_states_mean': loss_states.mean().to_dict() if not loss_states.empty else {},
            'profit_states_mean': profit_states.mean().to_dict() if not profit_states.empty else {},
        }
        
        # 识别亏损时的共性状态特征
        if not loss_states.empty and not profit_states.empty:
            diff = loss_states.mean() - profit_states.mean()
            most_different = diff.abs().idxmax()
            analysis['most_different_feature'] = {
                'feature': most_different,
                'loss_mean': float(loss_states[most_different].mean()),
                'profit_mean': float(profit_states[most_different].mean()),
                'difference': float(diff[most_different])
            }
        
        return analysis
    
    def _analyze_thermo_patterns(self, episodes: List[EpisodeRecord]) -> Dict[str, Any]:
        """分析热力学状态的演化模式"""
        states = [ep.state for ep in episodes]
        
        if len(states) < 5:
            return {'status': 'insufficient_data'}
        
        df = pd.DataFrame(states)
        
        patterns = {
            'entropy_trend': 'decreasing' if df['entropy'].is_monotonic_decreasing else 
                           'increasing' if df['entropy'].is_monotonic_increasing else 'mixed',
            'coherence_regime': 'high' if df['coherence'].mean() > 0.6 else 
                              'low' if df['coherence'].mean() < 0.3 else 'medium',
            'clarity_regime': 'high' if df['clarity'].mean() > 0.7 else 
                            'low' if df['clarity'].mean() < 0.3 else 'medium',
            'avg_rsi': float(df['rsi'].mean()),
            'rsi_volatility': float(df['rsi'].std())
        }
        
        return patterns
    
    def _identify_root_causes(
        self,
        pred_errors: Dict,
        loss_analysis: Dict,
        thermo_patterns: Dict
    ) -> List[Dict[str, Any]]:
        """识别系统性失效的根因"""
        causes = []
        
        # 根因1：世界模型偏差
        if pred_errors['overall_rmse'] > self.error_threshold:
            causes.append({
                'type': 'model_failure',
                'severity': 'high',
                'description': f"世界模型RMSE({pred_errors['overall_rmse']:.3f})超过阈值",
                'affected_dimension': pred_errors.get('max_bias_dimension', 'unknown'),
                'suggested_fix': '增加模型容量或引入外部冲击特征'
            })
        
        # 根因2：策略在特定热力学状态下失效
        if loss_analysis.get('has_losses', False):
            if loss_analysis['loss_ratio'] > 0.5:
                causes.append({
                    'type': 'strategy_failure',
                    'severity': 'critical',
                    'description': f"亏损比例({loss_analysis['loss_ratio']:.1%})过高",
                    'suggested_fix': '调整热力学约束奖励权重或降低仓位上限'
                })
            
            # 检查是否在高不确定性下操作
            mdf = loss_analysis.get('most_different_feature', {})
            if mdf.get('feature') == 'clarity' and mdf.get('loss_mean', 1) < 0.3:
                causes.append({
                    'type': 'risk_management',
                    'severity': 'medium',
                    'description': "策略在路径清晰度低时仍然重仓",
                    'suggested_fix': '收紧clarity_penalty因子'
                })
        
        # 根因3：市场体制变化
        if thermo_patterns.get('coherence_regime') == 'low' and \
           thermo_patterns.get('entropy_trend') == 'increasing':
            causes.append({
                'type': 'regime_mismatch',
                'severity': 'medium',
                'description': "市场处于高熵混沌态，策略未适应",
                'suggested_fix': '触发元学习控制器快速适应'
            })
        
        return causes
    
    def generate_improvement_tasks(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        将分析转化为可执行的任务队列
        """
        tasks = []
        root_causes = analysis.get('root_causes', [])
        
        for cause in root_causes:
            task = self._cause_to_task(cause)
            if task:
                tasks.append(task)
        
        # 如果没有严重问题，添加常规优化任务
        if not any(c['severity'] == 'critical' for c in root_causes):
            tasks.append({
                'type': 'optimization',
                'priority': 'low',
                'target': 'hyperparameters',
                'action': 'run_hyperparam_search',
                'description': '执行常规超参搜索以小幅提升性能'
            })
        
        return tasks
    
    def _cause_to_task(self, cause: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """将根因转换为具体任务"""
        task_map = {
            'model_failure': {
                'type': 'architecture',
                'target': 'ThermoWorldModel',
                'action': 'increase_capacity'
            },
            'strategy_failure': {
                'type': 'reward_shaping',
                'target': 'ThermoAgent',
                'action': 'adjust_reward_weights'
            },
            'risk_management': {
                'type': 'constraint',
                'target': 'ThermoAgent',
                'action': 'tighten_clarity_constraint'
            },
            'regime_mismatch': {
                'type': 'meta_learning',
                'target': 'MetaController',
                'action': 'trigger_adaptation'
            }
        }
        
        template = task_map.get(cause['type'])
        if not template:
            return None
        
        return {
            **template,
            'priority': cause.get('severity', 'medium'),
            'description': cause.get('description', ''),
            'suggested_fix': cause.get('suggested_fix', ''),
            'source_cause': cause
        }
    
    def generate_llm_prompt(self, analysis: Dict[str, Any]) -> str:
        """
        生成给大模型的分析提示词
        """
        prompt = f"""你是一位量化交易系统分析师。以下是过去{analysis.get('window', 20)}个交易日的系统表现分析：

## 世界模型预测误差
- 整体RMSE: {analysis.get('prediction_errors', {}).get('overall_rmse', 'N/A'):.4f}
- 最大偏差维度: {analysis.get('prediction_errors', {}).get('max_bias_dimension', 'N/A')}

## 策略亏损分析
- 亏损比例: {analysis.get('loss_analysis', {}).get('loss_ratio', 0):.1%}
- 平均亏损: {analysis.get('loss_analysis', {}).get('avg_loss', 0):.4f}

## 热力学模式
- 熵趋势: {analysis.get('thermo_patterns', {}).get('entropy_trend', 'N/A')}
- 合力状态: {analysis.get('thermo_patterns', {}).get('coherence_regime', 'N/A')}
- 路径清晰度: {analysis.get('thermo_patterns', {}).get('clarity_regime', 'N/A')}

## 已识别的根因
"""
        for i, cause in enumerate(analysis.get('root_causes', []), 1):
            prompt += f"{i}. [{cause.get('severity', 'unknown')}] {cause.get('description', '')}\n"
        
        prompt += """
请分析：
1. 世界模型的哪个假设可能失效了？
2. 热力学观测空间中是否缺失了关键特征？
3. 提出具体的系统改进建议。
"""
        return prompt
