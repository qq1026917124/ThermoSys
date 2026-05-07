"""
核心模块单元测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from thermo_sys.core import (
    ThermoState, ThermoStateEncoder,
    RetailSentimentIndex,
    InformationPropagationVelocity,
    HeatTransferNetwork,
    CoherenceForce
)


class TestThermoState:
    def test_creation(self):
        state = ThermoState(
            ipv=1.5,
            clarity=0.8,
            coherence=0.7,
            entropy=0.5,
            rsi=-1.2
        )
        assert state.ipv == 1.5
        assert state.clarity == 0.8
        
    def test_extreme_detection(self):
        fear_state = ThermoState(rsi=-2.5)
        assert fear_state.is_extreme_fear()
        assert not fear_state.is_extreme_greed()
        
        greed_state = ThermoState(rsi=2.5)
        assert greed_state.is_extreme_greed()
        
    def test_tensor_conversion(self):
        state = ThermoState(
            sector_temperatures={'A': 1.0, 'B': 2.0},
            group_phases={'G1': 0.5}
        )
        tensor = state.to_tensor()
        assert tensor.shape[0] == 7 + 2 + 1  # 7 global + 2 sectors + 1 group


class TestRetailSentimentIndex:
    def test_compute(self):
        n = 100
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B')
        
        margin = pd.Series(15000 + np.cumsum(np.random.randn(n)*100), index=dates)
        small_flow = pd.Series(np.random.randn(n)*5000, index=dates)
        accounts = pd.Series(30000 + np.random.randn(n)*5000, index=dates)
        search = pd.Series(1000 + np.random.randn(n)*200, index=dates)
        pcr = pd.Series(0.9 + np.random.randn(n)*0.1, index=dates)
        
        rsi = RetailSentimentIndex()
        result = rsi.compute(margin, small_flow, accounts, search, pcr)
        
        assert len(result) == n
        assert result.abs().max() <= 3.0  # Z-score clipping
        
    def test_signals(self):
        rsi = RetailSentimentIndex()
        dates = pd.date_range(end=datetime.now(), periods=60, freq='B')
        rsi_series = pd.Series(np.random.randn(60), index=dates)
        
        signals = rsi.get_signal(rsi_series, method='fixed_threshold')
        assert 'signal' in signals.columns
        assert 'strength' in signals.columns
        assert signals['signal'].isin([-1, 0, 1]).all()


class TestInformationPropagationVelocity:
    def test_info_density(self):
        n = 60
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B')
        mentions = pd.Series(np.random.randint(100, 1000, n), index=dates)
        engagement = pd.Series(np.random.randint(500, 5000, n), index=dates)
        volume = pd.Series(np.random.randn(n)*10000, index=dates)
        
        ipv = InformationPropagationVelocity()
        rho = ipv.compute_info_density(mentions, engagement, volume)
        
        assert len(rho) == n
        
    def test_velocity(self):
        ipv = InformationPropagationVelocity()
        dates = pd.date_range(end=datetime.now(), periods=60, freq='B')
        rho = pd.Series(np.random.randn(60), index=dates)
        gradient = pd.Series(np.abs(np.random.randn(60)), index=dates)
        
        velocity = ipv.compute_velocity(rho, gradient)
        assert len(velocity) == 60


class TestHeatTransferNetwork:
    def test_resistance_matrix(self):
        sectors = ['A', 'B', 'C']
        network = HeatTransferNetwork(sectors)
        
        supply = pd.DataFrame(
            [[0, 0.5, 0.3], [0.5, 0, 0.4], [0.3, 0.4, 0]],
            index=sectors, columns=sectors
        )
        
        R = network.build_resistance_matrix(supply_chain_weights=supply)
        assert R.shape == (3, 3)
        assert np.all(np.diag(R) == np.inf)
        
    def test_heat_transfer(self):
        sectors = ['A', 'B', 'C']
        network = HeatTransferNetwork(sectors)
        
        supply = pd.DataFrame(
            [[0, 0.5, 0.3], [0.5, 0, 0.4], [0.3, 0.4, 0]],
            index=sectors, columns=sectors
        )
        network.build_resistance_matrix(supply_chain_weights=supply)
        
        temps = np.array([2.0, 0.5, 0.0])
        dT = network.compute_heat_transfer(temps)
        
        assert len(dT) == 3
        # 热点A应该向B和C传导
        assert dT[0] < 0  # A失去热量
        assert dT[1] > 0  # B获得热量


class TestCoherenceForce:
    def test_entropy(self):
        cf = CoherenceForce()
        dist = pd.DataFrame(
            np.random.dirichlet([1, 2, 4, 2, 1], 30),
            columns=['e1', 'e2', 'n', 'b1', 'b2']
        )
        
        entropy = cf.compute_entropy(dist)
        assert len(entropy) == 30
        assert (entropy >= 0).all() and (entropy <= 1).all()
        
    def test_kuramoto(self):
        cf = CoherenceForce()
        dates = pd.date_range(end=datetime.now(), periods=60, freq='B')
        
        flows = {
            'retail': pd.Series(np.sin(np.linspace(0, 4*np.pi, 60)), index=dates),
            'institution': pd.Series(np.sin(np.linspace(0, 4*np.pi, 60) + 0.1), index=dates),
        }
        
        order_param = cf.compute_kuramoto_order(flows)
        assert len(order_param) == 60
        assert (order_param >= 0).all() and (order_param <= 1).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
