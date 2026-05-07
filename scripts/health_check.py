"""
系统健康检查脚本
用于监控 ThermoSys 每日任务是否正确运行
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List


def check_system_health() -> Dict:
    """
    检查系统健康状态
    
    检查项：
    1. 日志文件是否存在且正常更新
    2. 报告文件是否生成
    3. 回测历史是否正常记录
    4. 系统指标是否在合理范围
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'healthy',
        'checks': {}
    }
    
    # 1. 检查日志文件
    log_dir = Path("logs")
    if log_dir.exists():
        log_files = list(log_dir.glob("daily_run_*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            age_hours = (datetime.now() - datetime.fromtimestamp(latest_log.stat().st_mtime)).total_seconds() / 3600
            
            results['checks']['log_file'] = {
                'status': 'ok' if age_hours < 24 else 'warning',
                'latest_file': str(latest_log),
                'age_hours': round(age_hours, 1),
                'size_kb': round(latest_log.stat().st_size / 1024, 1)
            }
        else:
            results['checks']['log_file'] = {'status': 'error', 'message': 'No log files found'}
            results['overall_status'] = 'error'
    else:
        results['checks']['log_file'] = {'status': 'warning', 'message': 'Log directory not found'}
    
    # 2. 检查报告文件
    report_dir = Path("data/auto_loop")
    if report_dir.exists():
        reports = list(report_dir.glob("report_*.json"))
        if reports:
            latest_report = max(reports, key=lambda x: x.stat().st_mtime)
            age_hours = (datetime.now() - datetime.fromtimestamp(latest_report.stat().st_mtime)).total_seconds() / 3600
            
            # 读取报告内容
            try:
                with open(latest_report, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                
                signal_count = len(report_data.get('signals', []))
                
                results['checks']['report'] = {
                    'status': 'ok' if age_hours < 24 else 'warning',
                    'latest_file': str(latest_report),
                    'age_hours': round(age_hours, 1),
                    'signal_count': signal_count
                }
            except Exception as e:
                results['checks']['report'] = {'status': 'error', 'message': str(e)}
                results['overall_status'] = 'error'
        else:
            results['checks']['report'] = {'status': 'error', 'message': 'No reports found'}
            results['overall_status'] = 'error'
    else:
        results['checks']['report'] = {'status': 'warning', 'message': 'Report directory not found'}
    
    # 3. 检查回测历史
    history_file = Path("data/auto_loop/backtest_history.json")
    if history_file.exists():
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            results['checks']['backtest_history'] = {
                'status': 'ok',
                'records_count': len(history),
                'latest_sharpe': history[-1].get('sharpe', 0) if history else 0
            }
            
            # 检查最近记录质量
            if history:
                recent_quality = [h.get('quality_score', 0) for h in history[-5:]]
                avg_quality = sum(recent_quality) / len(recent_quality) if recent_quality else 0
                
                if avg_quality < 0.3:
                    results['checks']['backtest_history']['quality_warning'] = 'Low signal quality detected'
                    results['overall_status'] = 'warning'
                    
        except Exception as e:
            results['checks']['backtest_history'] = {'status': 'error', 'message': str(e)}
    else:
        results['checks']['backtest_history'] = {'status': 'warning', 'message': 'History file not found'}
    
    # 4. 检查配置文件
    config_file = Path("config/system_config.yaml")
    if config_file.exists():
        results['checks']['config'] = {'status': 'ok', 'file': str(config_file)}
    else:
        results['checks']['config'] = {'status': 'error', 'message': 'Config file not found'}
        results['overall_status'] = 'error'
    
    # 确定总体状态
    error_count = sum(1 for c in results['checks'].values() if c.get('status') == 'error')
    warning_count = sum(1 for c in results['checks'].values() if c.get('status') == 'warning')
    
    if error_count > 0:
        results['overall_status'] = 'critical'
    elif warning_count > 0:
        results['overall_status'] = 'warning'
    else:
        results['overall_status'] = 'healthy'
    
    return results


def print_health_report(results: Dict):
    """打印健康报告"""
    print("=" * 70)
    print("ThermoSys 系统健康检查报告")
    print("=" * 70)
    print(f"检查时间: {results['timestamp']}")
    print(f"总体状态: {results['overall_status'].upper()}")
    print()
    
    for check_name, check_data in results['checks'].items():
        status = check_data.get('status', 'unknown')
        status_symbol = {'ok': '[OK]', 'warning': '[WARN]', 'error': '[ERR]'}.get(status, '[?]')
        
        print(f"{status_symbol} {check_name}:")
        
        if status == 'ok':
            for key, value in check_data.items():
                if key != 'status':
                    print(f"   {key}: {value}")
        else:
            print(f"   状态: {status}")
            if 'message' in check_data:
                print(f"   信息: {check_data['message']}")
        print()
    
    print("=" * 70)
    
    # 提供建议
    if results['overall_status'] == 'critical':
        print("[!] 系统存在严重问题，请立即检查！")
        print("建议:")
        print("  1. 检查日志文件错误信息")
        print("  2. 确认数据目录权限")
        print("  3. 重新运行初始化脚本")
    elif results['overall_status'] == 'warning':
        print("[!] 系统存在警告，建议关注")
        print("建议:")
        print("  1. 检查最近24小时是否正常运行")
        print("  2. 关注信号质量下降趋势")
        print("  3. 检查磁盘空间")
    else:
        print("[OK] 系统运行正常")
    
    print("=" * 70)


def main():
    """主函数"""
    results = check_system_health()
    print_health_report(results)
    
    # 保存检查结果
    check_file = Path("data/health_check.json")
    check_file.parent.mkdir(parents=True, exist_ok=True)
    with open(check_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 返回退出码
    if results['overall_status'] == 'critical':
        return 2
    elif results['overall_status'] == 'warning':
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
