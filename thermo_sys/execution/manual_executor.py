"""
手动交易执行器
针对 A股 T+1 交易规则设计的手动执行方案

功能：
1. 生成交易信号和操作建议
2. 生成可执行的交易清单
3. 记录手动执行结果
4. 追踪持仓和盈亏
"""
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from loguru import logger


class ActionType(Enum):
    """操作类型"""
    BUY = "买入"
    SELL = "卖出"
    HOLD = "持有"
    WATCH = "观望"
    REDUCE = "减仓"
    ADD = "加仓"


@dataclass
class TradeSignal:
    """交易信号"""
    symbol: str
    action: ActionType
    confidence: float  # 置信度 0-1
    target_position: float  # 目标仓位 -1.0 ~ 1.0
    current_position: float  # 当前仓位
    reasoning: str  # 推理说明
    thermo_state: Dict[str, float]  # 热力学状态
    timestamp: datetime = field(default_factory=datetime.now)
    executed: bool = False  # 是否已执行
    execution_price: Optional[float] = None
    execution_time: Optional[datetime] = None


@dataclass
class WeeklyStrategy:
    """周内策略计划"""
    week_start: datetime
    week_end: datetime
    market_regime: str  # 牛市/熊市/震荡/过渡
    overall_bias: str  # 偏多/偏空/中性
    target_stocks: List[str]  # 目标股票池
    max_positions: int = 5  # 最大持仓数
    max_single_position: float = 0.3  # 单票最大仓位
    stop_loss: float = -0.07  # 止损线 -7%
    take_profit: float = 0.15  # 止盈线 +15%
    

class ManualTradeExecutor:
    """
    手动交易执行器
    
    因为 A股 T+1，无法全自动交易，所以：
    1. 系统生成信号和建议
    2. 用户手动在券商APP执行
    3. 记录执行结果用于回测和优化
    """
    
    def __init__(self, config_path: str = "config/system_config.yaml"):
        self.config = self._load_config(config_path)
        self.trade_history: List[TradeSignal] = []
        self.positions: Dict[str, Dict] = {}  # 当前持仓
        self.weekly_plan: Optional[WeeklyStrategy] = None
        
        # 数据文件
        self.data_dir = Path("data/trades")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载历史
        self._load_history()
    
    def _load_config(self, path: str) -> Dict:
        """加载配置"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except:
            return {}
    
    def _load_history(self):
        """加载交易历史"""
        history_file = self.data_dir / "trade_history.json"
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 恢复对象
                for item in data:
                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                    if item.get('execution_time'):
                        item['execution_time'] = datetime.fromisoformat(item['execution_time'])
                    item['action'] = ActionType(item['action'])
                    self.trade_history.append(TradeSignal(**item))
    
    def _save_history(self):
        """保存交易历史"""
        history_file = self.data_dir / "trade_history.json"
        data = []
        for signal in self.trade_history:
            item = {
                'symbol': signal.symbol,
                'action': signal.action.value,
                'confidence': signal.confidence,
                'target_position': signal.target_position,
                'current_position': signal.current_position,
                'reasoning': signal.reasoning,
                'thermo_state': signal.thermo_state,
                'timestamp': signal.timestamp.isoformat(),
                'executed': signal.executed,
                'execution_price': signal.execution_price,
                'execution_time': signal.execution_time.isoformat() if signal.execution_time else None
            }
            data.append(item)
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def generate_weekly_plan(self, 
                            market_analysis: Dict[str, Any],
                            stock_pool: List[str]) -> WeeklyStrategy:
        """
        生成周内交易计划
        
        基于热力学分析生成整周的交易策略
        """
        # 判断市场状态
        rsi = market_analysis.get('rsi', 0)
        coherence = market_analysis.get('coherence', 0)
        clarity = market_analysis.get('clarity', 0)
        ipv = market_analysis.get('ipv', 0)
        
        # 判断市场 regime
        if rsi < -1.5 and coherence > 0.6:
            regime = "牛市初期"
            bias = "偏多"
        elif rsi > 1.5 and coherence < 0.3:
            regime = "熊市"
            bias = "偏空"
        elif clarity < 0.3:
            regime = "混沌震荡"
            bias = "中性"
        else:
            regime = "震荡整理"
            bias = "中性偏多"
        
        # 过滤股票池（基于热力学评分）
        scored_stocks = []
        for symbol in stock_pool:
            score = self._score_stock(symbol, market_analysis)
            scored_stocks.append((symbol, score))
        
        # 排序取前N
        scored_stocks.sort(key=lambda x: x[1], reverse=True)
        target_stocks = [s[0] for s in scored_stocks[:self.weekly_plan.max_positions if self.weekly_plan else 5]]
        
        now = datetime.now()
        # 计算本周结束时间（周五）
        days_until_friday = 4 - now.weekday()
        if days_until_friday < 0:
            days_until_friday += 7
        week_end = now + timedelta(days=days_until_friday)
        
        plan = WeeklyStrategy(
            week_start=now,
            week_end=week_end,
            market_regime=regime,
            overall_bias=bias,
            target_stocks=target_stocks,
            max_positions=5,
            max_single_position=0.3,
            stop_loss=-0.07,
            take_profit=0.15
        )
        
        self.weekly_plan = plan
        return plan
    
    def _score_stock(self, symbol: str, market_analysis: Dict) -> float:
        """
        基于热力学的股票评分
        
        综合考虑：
        1. 板块温度（是否热点板块）
        2. 个股与市场的共振度
        3. 信息传播速度（是否有资金关注）
        """
        # 简化版评分
        base_score = np.random.uniform(0.3, 0.8)
        
        # 如果市场形成合力，给趋势明确的股票加分
        if market_analysis.get('coherence', 0) > 0.6:
            base_score += 0.1
        
        # 如果信息传播速度快，说明有资金关注
        if market_analysis.get('ipv', 0) > 1.5:
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def generate_daily_signals(self, 
                              thermo_states: Dict[str, Dict[str, float]],
                              current_positions: Dict[str, float]) -> List[TradeSignal]:
        """
        生成每日交易信号
        
        针对每只股票生成具体操作建议
        """
        signals = []
        
        for symbol, state in thermo_states.items():
            current_pos = current_positions.get(symbol, 0)
            
            # 基于热力学状态生成信号
            signal = self._generate_signal_for_stock(symbol, state, current_pos)
            signals.append(signal)
        
        # 按置信度排序
        signals.sort(key=lambda x: x.confidence, reverse=True)
        return signals
    
    def _generate_signal_for_stock(self, symbol: str, state: Dict, current_pos: float) -> TradeSignal:
        """为单只股票生成信号"""
        rsi = state.get('rsi', 0)
        coherence = state.get('coherence', 0)
        clarity = state.get('clarity', 0.5)
        entropy = state.get('entropy', 1.0)
        ipv = state.get('ipv', 0)
        
        reasoning_parts = []
        
        # 判断操作
        if rsi < -2.0 and coherence > 0.6 and clarity > 0.5:
            # 极端恐惧 + 趋势形成 + 路径清晰 = 强烈买入
            action = ActionType.BUY
            target_pos = min(current_pos + 0.3, 1.0)
            confidence = 0.9
            reasoning_parts.append(f"RSI极度恐惧({rsi:.2f})，市场情绪冰点")
            reasoning_parts.append(f"市场形成合力({coherence:.2f})，趋势确立")
            
        elif rsi > 2.0 and coherence < 0.3:
            # 极端贪婪 + 趋势瓦解 = 卖出
            action = ActionType.SELL
            target_pos = 0.0
            confidence = 0.85
            reasoning_parts.append(f"RSI极度贪婪({rsi:.2f})，市场过热")
            reasoning_parts.append(f"趋势瓦解({coherence:.2f})，合力消散")
            
        elif clarity < 0.2 and current_pos > 0.3:
            # 路径极度混乱 + 有持仓 = 减仓观望
            action = ActionType.REDUCE
            target_pos = current_pos * 0.5
            confidence = 0.7
            reasoning_parts.append(f"传播路径极度紊乱({clarity:.2f})")
            reasoning_parts.append("建议降低仓位，等待清晰信号")
            
        elif entropy > 0.8 and current_pos > 0:
            # 高熵（混乱）+ 有持仓 = 减仓
            action = ActionType.REDUCE
            target_pos = current_pos * 0.6
            confidence = 0.6
            reasoning_parts.append(f"市场高熵混乱({entropy:.2f})")
            
        elif ipv > 2.0 and coherence > 0.5 and current_pos < 0.5:
            # 信息加速传播 + 趋势形成 = 加仓
            action = ActionType.ADD
            target_pos = min(current_pos + 0.2, 1.0)
            confidence = 0.75
            reasoning_parts.append(f"信息加速传播({ipv:.2f})，资金进场")
            reasoning_parts.append(f"趋势确认({coherence:.2f})")
            
        else:
            # 默认持有
            action = ActionType.HOLD
            target_pos = current_pos
            confidence = 0.5
            reasoning_parts.append("暂无明确信号，建议持有观望")
        
        reasoning = "; ".join(reasoning_parts)
        
        return TradeSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            target_position=target_pos,
            current_position=current_pos,
            reasoning=reasoning,
            thermo_state=state
        )
    
    def execute_signal(self, signal: TradeSignal, price: float):
        """
        记录信号执行
        
        注意：实际交易在券商APP手动完成，这里只记录
        """
        signal.executed = True
        signal.execution_price = price
        signal.execution_time = datetime.now()
        
        # 更新持仓
        if signal.action in [ActionType.BUY, ActionType.ADD]:
            self.positions[signal.symbol] = {
                'position': signal.target_position,
                'entry_price': price,
                'entry_time': datetime.now()
            }
        elif signal.action in [ActionType.SELL, ActionType.REDUCE]:
            if signal.symbol in self.positions:
                if signal.target_position <= 0:
                    del self.positions[signal.symbol]
                else:
                    self.positions[signal.symbol]['position'] = signal.target_position
        
        self.trade_history.append(signal)
        self._save_history()
        
        logger.info(f"执行信号: {signal.symbol} {signal.action.value} @ {price}")
    
    def generate_daily_report(self) -> str:
        """
        生成每日交易报告（用于手动执行参考）
        
        返回可读的文本报告
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"ThermoSys 每日交易报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 70)
        
        # 1. 市场状态
        lines.append("\n【市场状态】")
        if self.weekly_plan:
            lines.append(f"  市场阶段: {self.weekly_plan.market_regime}")
            lines.append(f"  整体偏向: {self.weekly_plan.overall_bias}")
            lines.append(f"  本周计划: {self.weekly_plan.week_start.strftime('%m-%d')} ~ {self.weekly_plan.week_end.strftime('%m-%d')}")
        
        # 2. 今日信号
        today_signals = [s for s in self.trade_history 
                        if s.timestamp.date() == datetime.now().date() 
                        and not s.executed]
        
        if today_signals:
            lines.append("\n【今日待执行信号】")
            for signal in today_signals:
                lines.append(f"\n  股票: {signal.symbol}")
                lines.append(f"  操作: {signal.action.value}")
                lines.append(f"  当前仓位: {signal.current_position*100:.0f}%")
                lines.append(f"  目标仓位: {signal.target_position*100:.0f}%")
                lines.append(f"  置信度: {signal.confidence*100:.0f}%")
                lines.append(f"  理由: {signal.reasoning}")
                lines.append(f"  热力学状态:")
                for key, value in signal.thermo_state.items():
                    if isinstance(value, float):
                        lines.append(f"    {key}: {value:.3f}")
                lines.append("-" * 50)
        else:
            lines.append("\n【今日信号】暂无新信号，建议持仓观望")
        
        # 3. 当前持仓
        if self.positions:
            lines.append("\n【当前持仓】")
            for symbol, pos in self.positions.items():
                lines.append(f"  {symbol}: {pos['position']*100:.0f}% (成本: {pos['entry_price']:.2f})")
        
        # 4. 操作建议
        lines.append("\n【操作建议】")
        lines.append("  1. 请在券商APP中手动执行上述操作")
        lines.append("  2. 执行后请运行: python scripts/record_trade.py")
        lines.append("  3. 严格遵守止损纪律（-7%）")
        lines.append("  4. T+1规则：今日买入的股票明日才能卖出")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def export_to_csv(self, filepath: str = "data/trades/signals.csv"):
        """导出信号到CSV（方便导入券商软件）"""
        if not self.trade_history:
            logger.warning("没有交易信号可导出")
            return
        
        records = []
        for signal in self.trade_history:
            records.append({
                '日期': signal.timestamp.strftime('%Y-%m-%d'),
                '股票代码': signal.symbol,
                '操作': signal.action.value,
                '目标仓位(%)': f"{signal.target_position*100:.0f}",
                '置信度(%)': f"{signal.confidence*100:.0f}",
                '理由': signal.reasoning,
                'RSI': signal.thermo_state.get('rsi', ''),
                '合力': signal.thermo_state.get('coherence', ''),
                '清晰度': signal.thermo_state.get('clarity', ''),
                '已执行': '是' if signal.executed else '否'
            })
        
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"信号已导出: {filepath}")
    
    def get_weekly_performance(self) -> Dict[str, float]:
        """获取本周绩效"""
        week_start = datetime.now() - timedelta(days=datetime.now().weekday())
        week_signals = [s for s in self.trade_history if s.timestamp >= week_start]
        
        executed = [s for s in week_signals if s.executed]
        
        return {
            'total_signals': len(week_signals),
            'executed': len(executed),
            'execution_rate': len(executed) / len(week_signals) if week_signals else 0,
            'avg_confidence': np.mean([s.confidence for s in week_signals]) if week_signals else 0
        }


if __name__ == '__main__':
    # 测试
    executor = ManualTradeExecutor()
    
    # 模拟生成信号
    mock_states = {
        '000001': {'rsi': -2.3, 'coherence': 0.75, 'clarity': 0.8, 'entropy': 0.3, 'ipv': 1.8},
        '000002': {'rsi': 1.8, 'coherence': 0.2, 'clarity': 0.3, 'entropy': 0.9, 'ipv': 0.5},
        '600000': {'rsi': -0.5, 'coherence': 0.5, 'clarity': 0.5, 'entropy': 0.6, 'ipv': 1.0}
    }
    
    current_pos = {'000001': 0.2, '000002': 0.5}
    
    signals = executor.generate_daily_signals(mock_states, current_pos)
    
    print("\n生成信号:")
    for signal in signals:
        print(f"\n{signal.symbol}: {signal.action.value}")
        print(f"  置信度: {signal.confidence:.0%}")
        print(f"  理由: {signal.reasoning}")
    
    # 生成报告
    print("\n" + executor.generate_daily_report())
