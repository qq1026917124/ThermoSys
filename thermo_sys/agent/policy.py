"""
热力学感知策略网络
Actor-Critic架构，Critic评估热力学价值而非仅财务收益
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
from collections import deque, namedtuple


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ThermoPolicyNetwork(nn.Module):
    """
    热力学策略网络
    
    特点：
    1. 双Critic：评估财务收益和热力学稳定性
    2. 热力学约束奖励塑形
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Actor网络（输出高斯策略参数）
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Linear(hidden_dim, action_dim)
        
        # 双Critic网络
        self.critic_reward = self._make_critic(state_dim, action_dim, hidden_dim)
        self.critic_thermo = self._make_critic(state_dim, action_dim, hidden_dim)
        
    def _make_critic(self, s_dim: int, a_dim: int, hidden_dim: int) -> nn.Module:
        """构建Critic网络"""
        return nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Actor前向传播
        
        Returns:
            mean, log_std: 高斯策略参数
        """
        hidden = self.actor(state)
        mean = torch.tanh(self.actor_mean(hidden))  # 输出范围[-1, 1]
        log_std = torch.clamp(
            self.actor_log_std(hidden),
            self.log_std_min,
            self.log_std_max
        )
        return mean, log_std
    
    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作并返回log概率"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # 重参数化采样
        noise = torch.randn_like(mean)
        action = mean + std * noise
        action = torch.clamp(action, -1, 1)
        
        # 计算log概率
        log_prob = -0.5 * (((action - mean) / (std + 1e-8))**2 + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估状态-动作对
        
        Returns:
            q_reward: 财务收益Q值
            q_thermo: 热力学稳定性Q值
            log_prob: 动作log概率
        """
        sa = torch.cat([state, action], dim=-1)
        q_reward = self.critic_reward(sa)
        q_thermo = self.critic_thermo(sa)
        
        # 重新计算log_prob
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        log_prob = -0.5 * (((action - mean) / (std + 1e-8))**2 + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return q_reward, q_thermo, log_prob


class ThermoAgent:
    """
    热力学Agent
    整合策略网络、经验回放、训练流程
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = 3e-5,
        critic_lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 256,
        reward_lambda_coherence: float = 0.3,
        reward_lambda_entropy: float = 0.2,
        clarity_penalty_factor: float = 5.0,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        
        # 策略网络
        self.policy = ThermoPolicyNetwork(state_dim, action_dim).to(device)
        self.policy_target = ThermoPolicyNetwork(state_dim, action_dim).to(device)
        self.policy_target.load_state_dict(self.policy.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.policy.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            list(self.policy.critic_reward.parameters()) + 
            list(self.policy.critic_thermo.parameters()),
            lr=critic_lr
        )
        
        # 经验回放缓冲区
        self.buffer = deque(maxlen=buffer_size)
        
        # 奖励塑形参数
        self.lambda_coherence = reward_lambda_coherence
        self.lambda_entropy = reward_lambda_entropy
        self.clarity_penalty = clarity_penalty_factor
        
        self.training_step = 0
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        选择动作
        
        Args:
            state: 热力学状态向量
            deterministic: 是否确定性选择（测试时用）
            
        Returns:
            action: 动作向量（仓位比例）
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            mean, log_std = self.policy(state_t)
            
            if deterministic:
                action = mean
            else:
                std = torch.exp(log_std)
                noise = torch.randn_like(mean)
                action = mean + std * noise
                action = torch.clamp(action, -1, 1)
        
        return action.cpu().numpy()[0]
    
    def shape_reward(
        self,
        financial_return: float,
        next_thermo_state: Dict[str, float],
        action: np.ndarray
    ) -> float:
        """
        热力学约束奖励塑形
        
        不仅看赚钱，还要看是否以"健康的热力学方式"赚钱
        """
        r_fin = financial_return
        
        # 1. 路径清晰度惩罚（避免在无序市场重仓）
        clarity = next_thermo_state.get('clarity', 0.5)
        position_size = np.abs(action).mean()
        clarity_penalty = -position_size * max(0, 0.3 - clarity) * self.clarity_penalty
        
        # 2. 熵减奖励（顺势而为）
        entropy_change = next_thermo_state.get('entropy_change', 0)
        entropy_bonus = self.lambda_entropy * (-entropy_change) * action.mean()
        
        # 3. 合力奖励
        coherence = next_thermo_state.get('coherence', 0)
        coherence_bonus = self.lambda_coherence * coherence * action.mean()
        
        return r_fin + clarity_penalty + entropy_bonus + coherence_bonus
    
    def add_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """添加经验到缓冲区"""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def train(self) -> Optional[Dict[str, float]]:
        """
        执行一步训练（SAC风格）
        """
        if len(self.buffer) < self.batch_size:
            return None
        
        # 采样批次
        batch = self._sample_batch()
        
        states = torch.tensor(batch['states'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.tensor(batch['next_states'], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch['dones'], dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        # --- Critic更新 ---
        with torch.no_grad():
            next_actions, next_log_probs = self.policy_target.sample_action(next_states)
            q_reward_next, q_thermo_next, _ = self.policy_target.evaluate(next_states, next_actions)
            
            # 熵正则化
            alpha = 0.2  # 温度参数
            q_next = 0.7 * q_reward_next + 0.3 * q_thermo_next - alpha * next_log_probs
            q_target = rewards + self.gamma * (1 - dones) * q_next
        
        q_reward_pred, q_thermo_pred, _ = self.policy.evaluate(states, actions)
        
        critic_loss = F.mse_loss(q_reward_pred, q_target) + \
                      0.3 * F.mse_loss(q_thermo_pred, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.critic_reward.parameters()) + 
            list(self.policy.critic_thermo.parameters()), 1.0
        )
        self.critic_optimizer.step()
        
        # --- Actor更新（延迟）---
        actor_loss = torch.tensor(0.0)
        if self.training_step % 2 == 0:
            new_actions, log_probs = self.policy.sample_action(states)
            q_reward_new, q_thermo_new, _ = self.policy.evaluate(states, new_actions)
            q_new = 0.7 * q_reward_new + 0.3 * q_thermo_new
            
            actor_loss = -(q_new - alpha * log_probs).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # 软更新目标网络
            self._soft_update(self.policy_target, self.policy)
        
        self.training_step += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if isinstance(actor_loss, torch.Tensor) else 0.0,
            'q_reward_mean': q_reward_pred.mean().item(),
            'q_thermo_mean': q_thermo_pred.mean().item(),
        }
    
    def _sample_batch(self) -> Dict[str, np.ndarray]:
        """采样训练批次"""
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        return {
            'states': np.array([e.state for e in batch]),
            'actions': np.array([e.action for e in batch]),
            'rewards': np.array([e.reward for e in batch]),
            'next_states': np.array([e.next_state for e in batch]),
            'dones': np.array([e.done for e in batch])
        }
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy': self.policy.state_dict(),
            'policy_target': self.policy_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'training_step': self.training_step
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.policy_target.load_state_dict(checkpoint['policy_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.training_step = checkpoint['training_step']
