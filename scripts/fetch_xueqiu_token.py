"""
Selenium 自动登录雪球获取 Token
安全地自动获取并更新到配置文件
"""
import time
import json
import yaml
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from loguru import logger
import getpass


class XueqiuTokenFetcher:
    """
    雪球 Token 自动获取器
    
    功能：
    1. 使用 Selenium 模拟浏览器登录
    2. 自动提取 xq_a_token
    3. 安全保存到配置文件
    """
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.driver = None
        self.config_path = Path("config/system_config.yaml")
        
    def _create_driver(self):
        """创建 Chrome 浏览器实例"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        # 禁用自动化特征
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
            })
            logger.info("Chrome 浏览器启动成功")
        except Exception as e:
            logger.error(f"浏览器启动失败: {e}")
            raise
    
    def fetch_token(self, username: str = None, password: str = None) -> str:
        """
        获取 Token
        
        Args:
            username: 雪球用户名/手机号/邮箱（不提供则交互式输入）
            password: 密码（不提供则交互式输入）
            
        Returns:
            xq_a_token 字符串
        """
        if not username:
            username = input("请输入雪球用户名/手机号/邮箱: ")
        if not password:
            password = getpass.getpass("请输入密码: ")
        
        try:
            self._create_driver()
            
            # 访问雪球首页
            logger.info("访问雪球网站...")
            self.driver.get("https://xueqiu.com")
            time.sleep(2)
            
            # 点击登录按钮
            login_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, "nav__login__btn"))
            )
            login_btn.click()
            time.sleep(1)
            
            # 切换到密码登录
            password_tab = self.driver.find_element(By.XPATH, "//span[contains(text(), '密码登录')]")
            password_tab.click()
            time.sleep(1)
            
            # 输入用户名
            username_input = self.driver.find_element(By.NAME, "username")
            username_input.clear()
            username_input.send_keys(username)
            
            # 输入密码
            password_input = self.driver.find_element(By.NAME, "password")
            password_input.clear()
            password_input.send_keys(password)
            
            # 点击登录
            submit_btn = self.driver.find_element(By.CLASS_NAME, "modal__login__btn")
            submit_btn.click()
            
            # 等待登录完成
            logger.info("等待登录完成...")
            time.sleep(5)
            
            # 获取 cookies
            cookies = self.driver.get_cookies()
            
            # 查找 xq_a_token
            token = None
            for cookie in cookies:
                if cookie['name'] == 'xq_a_token':
                    token = cookie['value']
                    break
            
            if not token:
                # 尝试从 localStorage 获取
                token = self.driver.execute_script("return localStorage.getItem('xq_a_token')")
            
            if token:
                logger.success(f"Token 获取成功: {token[:20]}...")
                return token
            else:
                raise Exception("未能获取到 Token，请检查登录是否成功")
                
        except Exception as e:
            logger.error(f"获取 Token 失败: {e}")
            # 保存截图以便调试
            if self.driver:
                self.driver.save_screenshot("xueqiu_login_error.png")
                logger.info("错误截图已保存: xueqiu_login_error.png")
            raise
        finally:
            if self.driver:
                self.driver.quit()
    
    def update_config(self, token: str):
        """
        安全地更新配置文件
        
        将 token 保存到 system_config.yaml，但:
        1. 不直接保存明文 token（使用环境变量引用）
        2. 生成 .env 文件存储敏感信息
        """
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        # 保存 token 到 .env 文件（已加入 .gitignore）
        env_path = Path(".env")
        env_content = f"""# ThermoSys 环境变量
# 此文件包含敏感信息，不要提交到版本控制！
XUEQIU_TOKEN={token}
"""
        
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        logger.info(f"Token 已保存到 {env_path.absolute()}")
        
        # 更新 system_config.yaml 引用环境变量
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # 确保 data_sources 部分存在
        if 'data_sources' not in config:
            config['data_sources'] = {}
        
        if 'xueqiu' not in config['data_sources']:
            config['data_sources']['xueqiu'] = {}
        
        # 不直接存储 token，而是标记使用环境变量
        config['data_sources']['xueqiu']['enabled'] = True
        config['data_sources']['xueqiu']['token_source'] = 'env:XUEQIU_TOKEN'
        config['data_sources']['xueqiu']['base_url'] = 'https://stock.xueqiu.com/v5/stock/'
        config['data_sources']['xueqiu']['rate_limit'] = 5
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)
        
        logger.success(f"配置已更新: {self.config_path.absolute()}")
        logger.info("请确保 .env 文件已添加到 .gitignore！")
    
    def run(self):
        """完整流程：获取并保存 Token"""
        print("="*60)
        print("雪球 Token 自动获取工具")
        print("="*60)
        print("提示：Token 将安全保存，不会提交到 Git")
        print()
        
        token = self.fetch_token()
        self.update_config(token)
        
        print("\n" + "="*60)
        print("完成！")
        print("Token 已保存到 .env 文件")
        print("配置文件已更新")
        print("="*60)


if __name__ == '__main__':
    fetcher = XueqiuTokenFetcher(headless=False)  # 首次运行建议 headless=False 以便观察
    fetcher.run()
