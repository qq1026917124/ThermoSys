"""
ThermoSys: Thermodynamic Market Sentiment Analysis System
基于热力学框架的市场情绪分析与自主进化Agent系统
"""

__version__ = "1.0.0"
__author__ = "ThermoSys Team"

from thermo_sys.core.thermo_state import ThermoState, ThermoStateEncoder
from thermo_sys.core.rsi import RetailSentimentIndex
from thermo_sys.core.ipv import InformationPropagationVelocity
from thermo_sys.core.heat_transfer import HeatTransferNetwork
from thermo_sys.core.coherence import CoherenceForce

__all__ = [
    "ThermoState",
    "ThermoStateEncoder", 
    "RetailSentimentIndex",
    "InformationPropagationVelocity",
    "HeatTransferNetwork",
    "CoherenceForce",
]
