"""
热力学传播路径分析
构建板块间热阻网络，计算信息热传导路径与清晰度
"""
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from scipy.sparse.csgraph import dijkstra
from thermo_sys.utils.math_utils import granger_causality


class HeatTransferNetwork:
    """
    热传导网络分析引擎
    
    核心功能：
    1. 构建板块间热阻网络
    2. 计算最小热阻传播路径
    3. 评估路径清晰度
    """
    
    def __init__(self, sectors: List[str]):
        self.sectors = sectors
        self.n_sectors = len(sectors)
        self.R_matrix: Optional[np.ndarray] = None  # 热阻矩阵
        self.G: Optional[nx.DiGraph] = None  # 网络图
        
    def build_resistance_matrix(
        self,
        supply_chain_weights: Optional[pd.DataFrame] = None,
        fund_cooccurrence: Optional[pd.DataFrame] = None,
        retail_cognition: Optional[pd.DataFrame] = None,
        granger_causality_matrix: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        构建热阻矩阵 R_ij
        
        R_ij = 1 / (w_supply + w_fund + w_retail)
        
        Args:
            supply_chain_weights: 供应链权重矩阵
            fund_cooccurrence: 资金共现权重矩阵
            retail_cognition: 散户认知权重矩阵
            granger_causality_matrix: Granger因果强度矩阵（作为传导效率）
        """
        n = self.n_sectors
        W = np.zeros((n, n))
        
        # 累加各层权重
        if supply_chain_weights is not None:
            W += self._align_matrix(supply_chain_weights).values
        
        if fund_cooccurrence is not None:
            W += self._align_matrix(fund_cooccurrence).values
        
        if retail_cognition is not None:
            W += self._align_matrix(retail_cognition).values
        
        if granger_causality_matrix is not None:
            # Granger因果强度越高，热阻越低
            W += self._align_matrix(granger_causality_matrix).values
        
        # 计算热阻（避免除零）
        W_safe = np.where(W < 1e-6, 1e-6, W)
        self.R_matrix = 1.0 / W_safe
        
        # 对角线设为无穷大（自身到自身无热阻概念）
        np.fill_diagonal(self.R_matrix, np.inf)
        
        return self.R_matrix
    
    def _align_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """对齐矩阵到标准板块顺序"""
        aligned = pd.DataFrame(0.0, index=self.sectors, columns=self.sectors, dtype=float)
        
        for s_i in self.sectors:
            for s_j in self.sectors:
                # 尝试多种匹配方式
                val = 0
                if s_i in df.index and s_j in df.columns:
                    val = df.loc[s_i, s_j]
                elif s_j in df.index and s_i in df.columns:
                    val = df.loc[s_j, s_i]
                aligned.loc[s_i, s_j] = val
        
        return aligned
    
    def compute_heat_transfer(
        self,
        temperatures_prev: np.ndarray,
        external_shock: Optional[np.ndarray] = None,
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        离散化热传导方程
        
        ΔT_i = α * Σ_j (1/R_ij) * (T_j - T_i) + S_i
        
        Args:
            temperatures_prev: 上一期各板块温度 [n_sectors]
            external_shock: 外生冲击 [n_sectors]
            alpha: 传导系数
            
        Returns:
            温度变化量 [n_sectors]
        """
        if self.R_matrix is None:
            raise ValueError("Must build resistance matrix first")
        
        n = self.n_sectors
        dT = np.zeros(n)
        
        for i in range(n):
            for j in range(n):
                if i != j and self.R_matrix[i, j] < np.inf:
                    conductance = 1.0 / self.R_matrix[i, j]
                    dT[i] += conductance * (temperatures_prev[j] - temperatures_prev[i])
        
        dT *= alpha
        
        if external_shock is not None:
            dT += external_shock
        
        return dT
    
    def compute_transfer_tree(
        self,
        source_sector: str,
        temperatures: np.ndarray
    ) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
        """
        计算从热点板块到全市场的热传导树
        
        Args:
            source_sector: 热点板块名称
            temperatures: 各板块当前温度
            
        Returns:
            (paths, distances): 路径字典和距离字典
        """
        if self.R_matrix is None:
            raise ValueError("Must build resistance matrix first")
        
        source_idx = self.sectors.index(source_sector)
        
        # 使用Dijkstra算法计算最短路径（最小热阻）
        distances, predecessors = dijkstra(
            self.R_matrix,
            indices=source_idx,
            return_predecessors=True
        )
        
        # 构建路径字典
        paths = {}
        dist_dict = {}
        
        for i, sector in enumerate(self.sectors):
            if i == source_idx:
                continue
            
            # 回溯路径
            path = []
            curr = i
            while curr != source_idx and curr >= 0:
                path.append(self.sectors[curr])
                curr = int(predecessors[curr])
            
            if curr == source_idx:
                path.append(source_sector)
                path.reverse()
                paths[sector] = path
                dist_dict[sector] = float(distances[i])
        
        return paths, dist_dict
    
    def compute_path_clarity(
        self,
        source_sector: str,
        temperatures: np.ndarray,
        actual_flows: Optional[np.ndarray] = None
    ) -> float:
        """
        计算路径清晰度
        
        Clarity = 实际热流 / 理论最大热流
        
        Args:
            source_sector: 热点板块
            temperatures: 各板块温度
            actual_flows: 实际观测到的热流（可用资金流向近似）
            
        Returns:
            清晰度 [0, 1]
        """
        if self.R_matrix is None:
            raise ValueError("Must build resistance matrix first")
        
        source_idx = self.sectors.index(source_sector)
        t_source = temperatures[source_idx]
        
        # 理论最大热流（沿最小热阻路径）
        paths, distances = self.compute_transfer_tree(source_sector, temperatures)
        
        theoretical_max_flow = 0
        actual_total_flow = 0
        
        for sector, path in paths.items():
            target_idx = self.sectors.index(sector)
            t_target = temperatures[target_idx]
            temp_diff = abs(t_source - t_target)
            
            # 最小热阻
            r_min = distances[sector]
            
            if r_min > 0 and r_min < np.inf:
                theoretical_max_flow += temp_diff / r_min
            
            if actual_flows is not None:
                actual_total_flow += abs(actual_flows[target_idx])
            else:
                # 无实际数据时，用温度变化近似
                actual_total_flow += abs(temp_diff / (r_min + 1))
        
        if theoretical_max_flow < 1e-8:
            return 0.0
        
        clarity = min(actual_total_flow / theoretical_max_flow, 1.0)
        return clarity
    
    def identify_hotspot_sectors(
        self,
        temperatures: np.ndarray,
        temperature_threshold: float = 1.5
    ) -> List[str]:
        """
        识别热点板块
        
        Args:
            temperatures: 各板块温度
            temperature_threshold: 温度阈值（Z-score）
            
        Returns:
            热点板块列表
        """
        hotspots = []
        for i, sector in enumerate(self.sectors):
            if temperatures[i] > temperature_threshold:
                hotspots.append(sector)
        return hotspots
    
    def get_sector_ranking(
        self,
        temperatures: np.ndarray,
        sort_by: str = 'temperature'
    ) -> pd.DataFrame:
        """
        获取板块排名
        """
        df = pd.DataFrame({
            'sector': self.sectors,
            'temperature': temperatures,
        })
        
        # 计算传导强度（出度加权平均热导）
        if self.R_matrix is not None:
            conductances = []
            for i in range(self.n_sectors):
                cond = 0
                for j in range(self.n_sectors):
                    if i != j and self.R_matrix[i, j] < np.inf:
                        cond += 1.0 / self.R_matrix[i, j]
                conductances.append(cond)
            df['conductance'] = conductances
        
        df = df.sort_values(sort_by, ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        return df
