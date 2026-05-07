"""
编码修复工具
修复Windows控制台的UTF-8编码问题
"""
import sys
import io

# 方法1: 使用TextIOWrapper重新包装stdout/stderr
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

# 方法2: 设置Windows控制台代码页（如果可用）
try:
    import ctypes
    kernel32 = ctypes.windll.kernel32
    # 设置代码页为UTF-8
    kernel32.SetConsoleCP(65001)
    kernel32.SetConsoleOutputCP(65001)
except Exception:
    pass

# 方法3: 设置环境变量
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

print("编码修复完成")
print("Encoding fix applied")
print("Chinese test: 中文测试")
