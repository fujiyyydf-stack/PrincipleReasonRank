"""
BRIGHT Benchmark 自定义检索器
使用 rank_bm25 (纯Python实现，无需Java)
"""

import re
import numpy as np
from typing import Dict, List
from tqdm import tqdm

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("[警告] rank_bm25 未安装，请运行: pip install rank_bm25")


def simple_tokenize(text: str) -> List[str]:
    """简单的英文分词"""
    # 转小写，只保留字母数字，按空格分割
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    # 过滤太短的词
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


class BM25Retriever:
    """
    BM25 检索器 (使用 rank_bm25)
    """
    
    def __init__(self):
        self.bm25 = None
        self.corpus = None
        print("[BM25Retriever] 初始化完成 (rank_bm25)")
    
    def build_index(self, documents: List[str]):
        """构建 BM25 索引"""
        print(f"[BM25Retriever] 构建索引，文档数: {len(documents)}")
        
        # 分词
        self.corpus = []
        for doc in tqdm(documents, desc="Tokenizing"):
            self.corpus.append(simple_tokenize(doc))
        
        # 构建 BM25 索引
        self.bm25 = BM25Okapi(self.corpus)
        print(f"[BM25Retriever] 索引构建完成!")
    
    def search(self, query: str) -> np.ndarray:
        """搜索并返回所有文档的分数"""
        query_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        return scores


# 全局实例
_bm25_retriever = None
_index_built_for = None


def custom_retrieval(
    queries: List[str],
    query_ids: List[str],
    documents: List[str],
    doc_ids: List[str],
    excluded_ids: Dict[str, List[str]],
    task: str = None,
    cache_dir: str = "cache",
    long_context: bool = False,
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """
    BRIGHT benchmark 标准接口 - 使用 BM25
    """
    global _bm25_retriever, _index_built_for
    
    if not HAS_BM25:
        raise ImportError("请安装 rank_bm25: pip install rank_bm25")
    
    # 初始化检索器
    if _bm25_retriever is None:
        _bm25_retriever = BM25Retriever()
    
    print(f"\n{'='*50}")
    print(f"[BM25] Task: {task}")
    print(f"[BM25] Queries: {len(queries)}, Documents: {len(documents)}")
    print(f"{'='*50}")
    
    # 构建索引（每个task只构建一次）
    cache_key = f"{task}_{long_context}"
    if _index_built_for != cache_key:
        _bm25_retriever.build_index(documents)
        _index_built_for = cache_key
    
    # 检索
    results = {}
    
    for qid, query in zip(tqdm(query_ids, desc="BM25 retrieval"), queries):
        # 获取所有文档分数
        scores = _bm25_retriever.search(query)
        
        # 过滤排除的文档并构建结果
        excluded = set(excluded_ids.get(qid, []))
        query_results = {}
        
        for did, score in zip(doc_ids, scores):
            if did not in excluded:
                query_results[did] = float(score)
        
        # 只保留 top 1000
        sorted_results = sorted(query_results.items(), key=lambda x: x[1], reverse=True)[:1000]
        results[qid] = dict(sorted_results)
    
    print(f"[BM25] 检索完成!")
    return results


# BRIGHT 标准接口
def retrieval_custom(queries, query_ids, documents, doc_ids, excluded_ids, **kwargs):
    """BRIGHT 标准接口"""
    return custom_retrieval(
        queries=queries,
        query_ids=query_ids,
        documents=documents,
        doc_ids=doc_ids,
        excluded_ids=excluded_ids,
        **kwargs
    )
