"""
Agent模块单元测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pytest

from thermo_sys.agent import ThermoWorldModel, ThermoPolicyNetwork, ThermoAgent
from thermo_sys.agent.meta_controller import MetaController


class TestWorldModel:
    def test_forward(self):
        model = ThermoWorldModel(state_dim=19, action_dim=1, hidden_dim=64)
        state = torch.randn(4, 19)
        action = torch.randn(4, 1)
        
        next_state, uncertainty = model(state, action)
        assert next_state.shape == (4, 19)
        assert uncertainty.shape == (4, 19)
        assert (uncertainty > 0).all()
        
    def test_loss(self):
        model = ThermoWorldModel(state_dim=19, action_dim=1, hidden_dim=64)
        state = torch.randn(4, 19)
        action = torch.randn(4, 1)
        next_state = torch.randn(4, 19)
        
        losses = model.compute_loss(state, action, next_state)
        assert 'total' in losses
        assert 'nll' in losses
        assert 'mse' in losses
        
    def test_predict_with_uncertainty(self):
        model = ThermoWorldModel(state_dim=10, action_dim=1, hidden_dim=32)
        state = np.random.randn(10)
        action = np.random.randn(1)
        
        mean, std, epistemic = model.predict_with_uncertainty(state, action, num_samples=20)
        assert mean.shape == (1, 10)
        assert std.shape == (1, 10)


class TestPolicyNetwork:
    def test_actor(self):
        policy = ThermoPolicyNetwork(state_dim=19, action_dim=1, hidden_dim=64)
        state = torch.randn(4, 19)
        
        mean, log_std = policy(state)
        assert mean.shape == (4, 1)
        assert log_std.shape == (4, 1)
        assert (mean >= -1).all() and (mean <= 1).all()
        
    def test_sample_action(self):
        policy = ThermoPolicyNetwork(state_dim=19, action_dim=1, hidden_dim=64)
        state = torch.randn(4, 19)
        
        action, log_prob = policy.sample_action(state)
        assert action.shape == (4, 1)
        assert (action >= -1).all() and (action <= 1).all()
        
    def test_evaluate(self):
        policy = ThermoPolicyNetwork(state_dim=19, action_dim=1, hidden_dim=64)
        state = torch.randn(4, 19)
        action = torch.randn(4, 1)
        
        q_reward, q_thermo, log_prob = policy.evaluate(state, action)
        assert q_reward.shape == (4, 1)
        assert q_thermo.shape == (4, 1)


class TestThermoAgent:
    def test_select_action(self):
        agent = ThermoAgent(state_dim=10, action_dim=1, device='cpu')
        state = np.random.randn(10)
        
        action = agent.select_action(state, deterministic=True)
        assert isinstance(action, np.ndarray)
        assert -1 <= action <= 1
        
    def test_add_experience(self):
        agent = ThermoAgent(state_dim=10, action_dim=1, device='cpu')
        
        for i in range(10):
            agent.add_experience(
                state=np.random.randn(10),
                action=np.random.randn(1),
                reward=np.random.randn(),
                next_state=np.random.randn(10),
                done=False
            )
        
        assert len(agent.buffer) == 10
        
    def test_reward_shaping(self):
        agent = ThermoAgent(state_dim=10, action_dim=1, device='cpu')
        
        thermo_state = {'clarity': 0.2, 'entropy_change': -0.1, 'coherence': 0.7}
        action = np.array([0.8])
        
        shaped = agent.shape_reward(0.01, thermo_state, action)
        # 清晰度低时应惩罚
        assert shaped < 0.01


class TestMetaController:
    def test_identify_regime(self):
        controller = MetaController(state_dim=10, n_regimes=4, hidden_dim=32)
        
        # 模拟状态历史
        state_history = np.random.randn(20, 10)
        
        result = controller.identify_regime(state_history)
        assert 'regime_id' in result
        assert 'regime_name' in result
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 1
        
    def test_forward(self):
        controller = MetaController(state_dim=10, n_regimes=4, hidden_dim=32)
        state_history = torch.randn(2, 20, 10)
        
        regime_id, logits, fast_weight = controller(state_history)
        assert isinstance(regime_id, int)
        assert 0 <= regime_id < 4
        assert logits.shape == (2, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
