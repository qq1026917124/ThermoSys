"""
数据处理器
负责数据清洗、特征工程、反垃圾过滤等
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import defaultdict
import re


class DataProcessor:
    """数据处理器基类"""
    
    @staticmethod
    def remove_outliers(series: pd.Series, method: str = 'iqr', k: float = 1.5) -> pd.Series:
        """剔除异常值"""
        if method == 'iqr':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            return series.clip(lower, upper)
        elif method == 'zscore':
            z = (series - series.mean()) / (series.std() + 1e-8)
            return series.where(z.abs() < 3, series.median())
        else:
            return series
    
    @staticmethod
    def handle_missing(data: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        """处理缺失值"""
        if method == 'ffill':
            return data.fillna(method='ffill').fillna(method='bfill')
        elif method == 'interpolate':
            return data.interpolate().fillna(method='bfill').fillna(method='ffill')
        elif method == 'median':
            return data.fillna(data.median())
        else:
            return data.dropna()


class TextProcessor:
    """
    文本数据处理器
    包括反垃圾过滤、情感分析、共现提取等
    """
    
    def __init__(self):
        # 散户专用情感词典（简化版）
        self.bull_words = ['涨停', '满仓', '梭哈', '抄底', '反弹', '突破', '牛市', '大涨']
        self.bear_words = ['割肉', '跌停', '清仓', '暴跌', '熊市', '崩盘', '套牢', '跳水']
        self.contra_words = ['国家队救市', '政策底', '老乡别走']  # 反指词典
        
    def filter_spam(self, posts: pd.DataFrame, min_history_posts: int = 10) -> pd.DataFrame:
        """
        反垃圾过滤
        
        策略：
        1. 过滤机器人账号（发帖频率异常）
        2. 过滤历史准确率低的用户
        """
        # 简化版：基于发帖频率过滤
        if 'user_id' in posts.columns and 'post_count' in posts.columns:
            # 过滤单日发帖超过阈值的用户
            spam_mask = posts.groupby('user_id')['post_count'].transform('sum') > 50
            return posts[~spam_mask]
        
        return posts
    
    def extract_sentiment(self, texts: List[str]) -> Dict[str, float]:
        """
        基于词典的情感分析
        
        Returns:
            {'bull_score', 'bear_score', 'contra_score', 'sentiment_ratio'}
        """
        bull_count = sum(1 for text in texts for word in self.bull_words if word in text)
        bear_count = sum(1 for text in texts for word in self.bear_words if word in text)
        contra_count = sum(1 for text in texts for word in self.contra_words if word in text)
        
        total = bull_count + bear_count + 1  # 避免除零
        
        return {
            'bull_score': bull_count / len(texts) if texts else 0,
            'bear_score': bear_count / len(texts) if texts else 0,
            'contra_score': contra_count / len(texts) if texts else 0,
            'sentiment_ratio': bull_count / total
        }
    
    def extract_cooccurrence(self, texts: List[str], keywords: List[str]) -> pd.DataFrame:
        """
        提取关键词共现矩阵
        
        Returns:
            共现频率矩阵
        """
        n = len(keywords)
        cooccur = np.zeros((n, n))
        
        for text in texts:
            present = [1 if kw in text else 0 for kw in keywords]
            for i in range(n):
                for j in range(i+1, n):
                    if present[i] and present[j]:
                        cooccur[i, j] += 1
                        cooccur[j, i] += 1
        
        return pd.DataFrame(cooccur, index=keywords, columns=keywords)
    
    def compute_homogeneity(self, sentiment_distribution: pd.DataFrame) -> pd.Series:
        """
        计算观点同质化程度（熵减指标）
        
        使用信息熵，熵越低观点越一致
        """
        from thermo_sys.utils.math_utils import normalized_entropy
        
        homogeneity = []
        for _, row in sentiment_distribution.iterrows():
            probs = row.values / (row.values.sum() + 1e-8)
            ent = normalized_entropy(probs)
            homogeneity.append(1 - ent)  # 转化为同质化程度，越高越一致
        
        return pd.Series(homogeneity, index=sentiment_distribution.index)


class FlowProcessor:
    """
    资金流向数据处理器
    """
    
    @staticmethod
    def decompose_flow(
        money_flow: pd.DataFrame,
        price: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        分解资金流向为各投资者群体
        
        Args:
            money_flow: 包含各档位资金流向的DataFrame
            price: 价格序列
            
        Returns:
            各群体资金序列字典
        """
        # 假设money_flow包含：small, medium, large, main
        groups = {}
        
        if 'small_inflow' in money_flow.columns:
            groups['retail'] = money_flow['small_inflow']
        
        if 'large_inflow' in money_flow.columns:
            groups['institution'] = money_flow['large_inflow']
        
        if 'main_inflow' in money_flow.columns:
            groups['hot_money'] = money_flow['main_inflow']
        
        # 计算净流向
        for key in groups:
            groups[key] = groups[key].resample('D').sum().fillna(0)
        
        return groups
    
    @staticmethod
    def compute_flow_momentum(
        flow: pd.Series,
        short_window: int = 5,
        long_window: int = 20
    ) -> pd.DataFrame:
        """
        计算资金流向动量
        
        Returns:
            DataFrame with ['short_ma', 'long_ma', 'momentum']
        """
        result = pd.DataFrame(index=flow.index)
        result['short_ma'] = flow.rolling(short_window).mean()
        result['long_ma'] = flow.rolling(long_window).mean()
        result['momentum'] = result['short_ma'] - result['long_ma']
        result['momentum_signal'] = np.where(result['momentum'] > 0, 1, -1)
        
        return result


class AntiSpamEngine:
    """
    高级反垃圾引擎
    """
    
    def __init__(self):
        self.user_history = defaultdict(lambda: {'posts': 0, 'correct_predictions': 0})
        
    def score_user(self, user_id: str, post_content: str, subsequent_return: Optional[float] = None) -> float:
        """
        计算用户信誉分 [0, 1]
        
        Args:
            user_id: 用户ID
            post_content: 帖子内容
            subsequent_return: 发帖后N日实际收益率（用于校准）
            
        Returns:
            信誉分
        """
        history = self.user_history[user_id]
        
        # 基础分
        if history['posts'] < 10:
            base_score = 0.5  # 新用户中性分
        else:
            accuracy = history['correct_predictions'] / history['posts']
            base_score = accuracy
        
        # 内容质量检测
        # 1. 检测重复内容
        repetition_penalty = 0.1 if len(set(post_content)) / len(post_content) < 0.5 else 0
        
        # 2. 检测机器人模式（固定时间间隔发帖）
        # 这里简化处理
        
        # 3. 情感极端度惩罚
        extreme_words = len([w for w in ['一定', '肯定', '绝对', '必须'] if w in post_content])
        extreme_penalty = min(extreme_words * 0.05, 0.2)
        
        score = max(0, min(1, base_score - repetition_penalty - extreme_penalty))
        
        # 更新历史
        history['posts'] += 1
        if subsequent_return is not None:
            # 判断预测是否正确（简化：看多后涨则正确）
            is_bullish = any(w in post_content for w in ['涨', '买', '抄底'])
            if (is_bullish and subsequent_return > 0) or (not is_bullish and subsequent_return < 0):
                history['correct_predictions'] += 1
        
        return score
    
    def filter_posts(self, posts_df: pd.DataFrame, min_score: float = 0.3) -> pd.DataFrame:
        """
        过滤低信誉用户的帖子
        """
        if 'user_id' not in posts_df.columns:
            return posts_df
        
        scores = posts_df['user_id'].map(
            lambda uid: self.user_history[uid]['correct_predictions'] / max(self.user_history[uid]['posts'], 1)
        )
        
        return posts_df[scores >= min_score]
