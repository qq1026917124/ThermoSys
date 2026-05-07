"""
ThermoSys 每日自动运行脚本
用于 Windows 任务计划程序调用
"""
import sys
import os
import io

# ========== 编码修复（必须在最前面） ==========
# 修复Windows控制台的UTF-8编码问题
try:
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleCP(65001)
    kernel32.SetConsoleOutputCP(65001)
except Exception:
    pass

os.environ['PYTHONIOENCODING'] = 'utf-8'

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass
# ===========================================

from pathlib import Path
from datetime import datetime
from loguru import logger

# 配置日志
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logger.add(
    log_dir / "daily_run_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    encoding="utf-8"
)

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from thermo_sys.execution import AutoEvolutionLoop


def run_daily_task():
    """
    每日自动任务
    
    此脚本应配置在 Windows 任务计划程序中，
    每日收盘后运行（建议 15:30-16:00）
    """
    logger.info("=" * 70)
    logger.info(f"ThermoSys Daily Task Started - {datetime.now()}")
    logger.info("=" * 70)
    
    try:
        # 初始化系统
        loop = AutoEvolutionLoop()
        
        # TODO: 从实际数据源获取数据
        # 这里使用模拟数据作为示例
        # 实际部署时，应接入 thermo_sys.data 模块获取实时数据
        
        # 模拟市场数据
        import numpy as np
        import pandas as pd
        
        dates = pd.date_range(end=datetime.now(), periods=30, freq='B')
        n = len(dates)
        
        price_data = pd.DataFrame(index=dates)
        thermo_data = {}
        
        stocks = ['000001', '000002', '600000']
        for stock in stocks:
            returns = np.random.randn(n) * 0.02
            price_data[stock] = 100 * (1 + returns).cumprod()
            
            thermo_data[stock] = pd.DataFrame({
                'rsi': np.random.randn(n),
                'coherence': np.random.rand(n),
                'clarity': np.random.rand(n),
                'entropy': np.random.rand(n),
                'ipv': np.random.randn(n) * 2
            }, index=dates)
        
        market_data = {
            'thermo_states': {
                stock: thermo_data[stock].iloc[-1].to_dict()
                for stock in stocks
            }
        }
        
        # 运行每日循环
        logger.info("Running daily strategy cycle...")
        report = loop.run_daily_cycle(market_data, price_data, thermo_data)
        
        logger.info(f"Generated {len(report['signals'])} signals")
        if 'backtest' in report:
            logger.info(f"Backtest Sharpe: {report['backtest']['sharpe_ratio']:.2f}")
        
        # 检查是否需要每周优化（周五）
        if datetime.now().weekday() == 4:  # 周五
            logger.info("Friday detected, running weekly optimization...")
            weekly_report = loop.run_weekly_optimization(price_data, thermo_data)
            logger.info(f"Optimization applied: {weekly_report.get('optimization_applied', False)}")
        
        # 检查是否需要月度深度回测（每月第一个交易日）
        if datetime.now().day <= 5:
            logger.info("Running monthly deep backtest...")
            full_report = loop.run_full_backtest(price_data, thermo_data)
            logger.info(f"Total return: {full_report['performance']['total_return']:.2%}")
            logger.info(f"Sharpe: {full_report['performance']['sharpe_ratio']:.2f}")
        
        # 输出绩效摘要
        summary = loop.get_performance_summary()
        logger.info(f"Recent avg Sharpe: {summary.get('avg_sharpe', 0):.2f}")
        logger.info(f"Recent avg quality: {summary.get('avg_quality', 0):.2f}")
        
        logger.info("=" * 70)
        logger.info("Daily task completed successfully")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"Task execution failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = run_daily_task()
    sys.exit(0 if success else 1)
