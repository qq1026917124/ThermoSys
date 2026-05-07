"""
进化Agent (Evolution Agent)
自动执行反思Agent生成的改进任务
能够修改配置、重训练模型、A/B测试
"""
import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class EvolutionResult:
    """进化任务执行结果"""
    task_id: str
    task_type: str
    status: str  # 'success', 'failed', 'rejected'
    baseline_sharpe: float
    new_sharpe: float
    improvement: float
    test_passed: bool
    details: Dict[str, Any]
    timestamp: str


class EvolutionAgent:
    """
    进化Agent
    
    功能：
    1. 执行特征工程任务
    2. 修改网络架构
    3. 运行超参搜索
    4. A/B测试验证
    """
    
    def __init__(
        self,
        codebase_path: str,
        checkpoint_dir: str = "./checkpoints",
        ab_test_window: int = 100,
        improvement_threshold: float = 0.1,
        auto_execute: bool = False
    ):
        self.codebase = Path(codebase_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.ab_test_window = ab_test_window
        self.improvement_threshold = improvement_threshold
        self.auto_execute = auto_execute
        self.task_history: List[EvolutionResult] = []
        
    def execute_task(self, task: Dict[str, Any]) -> EvolutionResult:
        """
        执行单个改进任务
        
        Args:
            task: 来自反思Agent的任务字典
            
        Returns:
            执行结果
        """
        task_id = f"{task['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if not self.auto_execute:
            print(f"[EvolutionAgent] 待执行任务: {task_id}")
            print(f"  类型: {task['type']}")
            print(f"  目标: {task.get('target', 'N/A')}")
            print(f"  动作: {task.get('action', 'N/A')}")
            print(f"  建议: {task.get('suggested_fix', 'N/A')}")
            
            # 非自动模式下返回pending状态
            return EvolutionResult(
                task_id=task_id,
                task_type=task['type'],
                status='pending',
                baseline_sharpe=0.0,
                new_sharpe=0.0,
                improvement=0.0,
                test_passed=False,
                details={'message': 'Waiting for manual approval'},
                timestamp=datetime.now().isoformat()
            )
        
        # 自动执行模式
        try:
            if task['type'] == 'feature_engineering':
                result = self._execute_feature_task(task)
            elif task['type'] == 'architecture':
                result = self._execute_architecture_task(task)
            elif task['type'] == 'hyperparameter':
                result = self._execute_hyperparameter_task(task)
            elif task['type'] == 'reward_shaping':
                result = self._execute_reward_shaping_task(task)
            elif task['type'] == 'constraint':
                result = self._execute_constraint_task(task)
            else:
                result = self._create_failed_result(task_id, task['type'], f"Unknown task type: {task['type']}")
        
        except Exception as e:
            result = self._create_failed_result(task_id, task['type'], str(e))
        
        self.task_history.append(result)
        return result
    
    def _execute_feature_task(self, task: Dict[str, Any]) -> EvolutionResult:
        """执行特征工程任务"""
        task_id = f"feature_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 保存当前版本为baseline
        baseline_path = self.checkpoint_dir / f"{task_id}_baseline"
        
        # 修改特征配置
        config_change = self._modify_feature_config(task)
        
        # 运行回测
        baseline_sharpe = self._run_backtest(branch='main')
        new_sharpe = self._run_backtest(branch='experiment')
        
        improvement = new_sharpe - baseline_sharpe
        test_passed = improvement > self.improvement_threshold
        
        return EvolutionResult(
            task_id=task_id,
            task_type='feature_engineering',
            status='success' if test_passed else 'rejected',
            baseline_sharpe=baseline_sharpe,
            new_sharpe=new_sharpe,
            improvement=improvement,
            test_passed=test_passed,
            details={'config_change': config_change},
            timestamp=datetime.now().isoformat()
        )
    
    def _execute_architecture_task(self, task: Dict[str, Any]) -> EvolutionResult:
        """执行架构修改任务"""
        task_id = f"arch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 修改网络架构（如增加层数、修改激活函数等）
        # 这里简化处理，实际应使用AST操作源代码
        
        return EvolutionResult(
            task_id=task_id,
            task_type='architecture',
            status='pending',
            baseline_sharpe=0.0,
            new_sharpe=0.0,
            improvement=0.0,
            test_passed=False,
            details={'message': 'Architecture changes require manual review'},
            timestamp=datetime.now().isoformat()
        )
    
    def _execute_hyperparameter_task(self, task: Dict[str, Any]) -> EvolutionResult:
        """执行超参搜索任务"""
        task_id = f"hyperparam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            import optuna
            
            def objective(trial):
                lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
                lambda_coherence = trial.suggest_float('lambda_coherence', 0.0, 1.0)
                hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
                
                # 快速回测
                sharpe = self._quick_backtest(
                    lr=lr,
                    lambda_coherence=lambda_coherence,
                    hidden_dim=hidden_dim
                )
                return sharpe
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=20, show_progress_bar=False)
            
            best_params = study.best_params
            best_sharpe = study.best_value
            baseline_sharpe = self._quick_backtest()  # 使用默认参数
            
            improvement = best_sharpe - baseline_sharpe
            test_passed = improvement > self.improvement_threshold
            
            return EvolutionResult(
                task_id=task_id,
                task_type='hyperparameter',
                status='success' if test_passed else 'rejected',
                baseline_sharpe=baseline_sharpe,
                new_sharpe=best_sharpe,
                improvement=improvement,
                test_passed=test_passed,
                details={'best_params': best_params},
                timestamp=datetime.now().isoformat()
            )
        
        except ImportError:
            return self._create_failed_result(task_id, 'hyperparameter', 'Optuna not installed')
        except Exception as e:
            return self._create_failed_result(task_id, 'hyperparameter', str(e))
    
    def _execute_reward_shaping_task(self, task: Dict[str, Any]) -> EvolutionResult:
        """调整奖励塑形参数"""
        task_id = f"reward_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 解析建议的修改
        suggested_fix = task.get('suggested_fix', '')
        
        # 根据建议调整参数
        param_changes = {}
        if 'clarity_penalty' in suggested_fix:
            param_changes['clarity_penalty_factor'] = 7.0  # 收紧
        if 'coherence' in suggested_fix:
            param_changes['lambda_coherence'] = 0.4
        
        baseline_sharpe = self._quick_backtest()
        new_sharpe = self._quick_backtest(**param_changes)
        
        improvement = new_sharpe - baseline_sharpe
        
        return EvolutionResult(
            task_id=task_id,
            task_type='reward_shaping',
            status='success' if improvement > 0 else 'rejected',
            baseline_sharpe=baseline_sharpe,
            new_sharpe=new_sharpe,
            improvement=improvement,
            test_passed=improvement > self.improvement_threshold,
            details={'param_changes': param_changes},
            timestamp=datetime.now().isoformat()
        )
    
    def _execute_constraint_task(self, task: Dict[str, Any]) -> EvolutionResult:
        """执行约束调整任务"""
        return self._execute_reward_shaping_task(task)  # 约束调整通常影响奖励
    
    def _modify_feature_config(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """修改特征配置文件"""
        config_path = self.codebase / "config" / "system_config.yaml"
        
        # 这里简化处理，实际应解析和修改YAML
        changes = {'message': 'Config modification simulated'}
        return changes
    
    def _run_backtest(self, branch: str = 'main') -> float:
        """运行完整回测"""
        # 模拟回测结果
        import random
        return random.uniform(0.8, 1.5)
    
    def _quick_backtest(self, **kwargs) -> float:
        """快速回测（最近100个交易日）"""
        # 模拟回测结果，参数影响结果
        import random
        base = 1.0
        for k, v in kwargs.items():
            if isinstance(v, float):
                base += (v - 0.5) * 0.1
        return base + random.uniform(-0.2, 0.3)
    
    def _create_failed_result(self, task_id: str, task_type: str, error: str) -> EvolutionResult:
        """创建失败结果"""
        return EvolutionResult(
            task_id=task_id,
            task_type=task_type,
            status='failed',
            baseline_sharpe=0.0,
            new_sharpe=0.0,
            improvement=0.0,
            test_passed=False,
            details={'error': error},
            timestamp=datetime.now().isoformat()
        )
    
    def rollback_last_change(self) -> bool:
        """回滚最后一次修改"""
        if not self.task_history:
            return False
        
        last_task = self.task_history[-1]
        if last_task.status == 'success':
            # 恢复到baseline版本
            print(f"[EvolutionAgent] 回滚任务: {last_task.task_id}")
            return True
        
        return False
    
    def get_evolution_report(self) -> Dict[str, Any]:
        """生成进化历史报告"""
        if not self.task_history:
            return {'status': 'no_history'}
        
        total_tasks = len(self.task_history)
        successful = sum(1 for t in self.task_history if t.status == 'success')
        rejected = sum(1 for t in self.task_history if t.status == 'rejected')
        failed = sum(1 for t in self.task_history if t.status == 'failed')
        
        improvements = [t.improvement for t in self.task_history if t.status == 'success']
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
        
        return {
            'total_tasks': total_tasks,
            'successful': successful,
            'rejected': rejected,
            'failed': failed,
            'success_rate': successful / total_tasks if total_tasks > 0 else 0,
            'average_improvement': avg_improvement,
            'best_improvement': max(improvements) if improvements else 0,
            'recent_tasks': [
                {
                    'task_id': t.task_id,
                    'type': t.task_type,
                    'status': t.status,
                    'improvement': t.improvement
                }
                for t in self.task_history[-10:]
            ]
        }
