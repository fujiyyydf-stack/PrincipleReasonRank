#!/usr/bin/env python3
"""
BRIGHT Benchmark 评估器
=======================

这个模块提供了向 BRIGHT benchmark 提交和评估的完整功能。
支持 BM25 基线、自定义检索器、以及与 PrincipleRank reranker 的集成。

使用方法:
    # 1. BM25 基线评估
    python bright_evaluator.py --method bm25 --tasks biology economics
    
    # 2. 使用自定义 reranker
    python bright_evaluator.py --method custom --reranker_path /path/to/model
    
    # 3. 评估所有任务
    python bright_evaluator.py --method bm25 --all
    
    # 4. 生成提交报告
    python bright_evaluator.py --generate_report
"""

import os
import re
import json
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from tqdm import tqdm

# 数据加载
from datasets import load_dataset
import pytrec_eval

# ============================================================================
# 配置
# ============================================================================

# BRIGHT 数据集列表
BRIGHT_TASKS = {
    "short_doc": [
        "biology", "earth_science", "economics", "psychology", "robotics",
        "stackoverflow", "sustainable_living", "leetcode", "pony", "aops",
        "theoremqa_questions", "theoremqa_theorems"
    ],
    "long_doc": [
        "biology", "earth_science", "economics", "psychology", "robotics",
        "stackoverflow", "sustainable_living", "leetcode"
    ]
}

# 默认缓存目录
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache", "bright")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class EvalResult:
    """评估结果"""
    task: str
    ndcg_at_10: float
    recall_at_1: float
    recall_at_10: float
    mrr: float
    full_metrics: Dict


# ============================================================================
# BM25 检索器 (纯 Python 实现)
# ============================================================================

class BM25Retriever:
    """BM25 检索器，使用 rank_bm25"""
    
    def __init__(self):
        try:
            from rank_bm25 import BM25Okapi
            self.BM25Okapi = BM25Okapi
            self.available = True
        except ImportError:
            self.available = False
            print("[警告] rank_bm25 未安装，请运行: pip install rank_bm25")
        
        self.bm25 = None
        self.corpus = None
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = [t for t in text.split() if len(t) > 1]
        return tokens
    
    def build_index(self, documents: List[str]):
        """构建索引"""
        if not self.available:
            raise ImportError("rank_bm25 未安装")
        
        print(f"[BM25] 构建索引: {len(documents)} 文档...")
        self.corpus = [self._tokenize(doc) for doc in tqdm(documents, desc="分词")]
        self.bm25 = self.BM25Okapi(self.corpus)
        print(f"[BM25] 索引构建完成")
    
    def search(self, query: str) -> np.ndarray:
        """检索，返回所有文档的分数"""
        query_tokens = self._tokenize(query)
        return self.bm25.get_scores(query_tokens)
    
    def batch_search(
        self,
        queries: List[str],
        query_ids: List[str],
        documents: List[str],
        doc_ids: List[str],
        excluded_ids: Dict[str, List[str]],
        top_k: int = 1000
    ) -> Dict[str, Dict[str, float]]:
        """批量检索"""
        
        # 构建索引
        if self.bm25 is None:
            self.build_index(documents)
        
        results = {}
        for qid, query in zip(tqdm(query_ids, desc="BM25 检索"), queries):
            scores = self.search(query)
            excluded = set(excluded_ids.get(qid, []))
            
            # 构建结果，排除指定文档
            query_results = {}
            for did, score in zip(doc_ids, scores):
                if did not in excluded:
                    query_results[did] = float(score)
            
            # 只保留 top_k
            sorted_results = sorted(query_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
            results[qid] = dict(sorted_results)
        
        return results


# ============================================================================
# 自定义 Reranker 接口
# ============================================================================

class BaseReranker:
    """Reranker 基类，你的模型需要继承这个类"""
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        doc_ids: List[str],
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        对候选文档进行重排序
        
        Args:
            query: 查询文本
            documents: 候选文档列表
            doc_ids: 候选文档ID列表
            **kwargs: 其他参数（如 rubric）
        
        Returns:
            排序后的 (doc_id, score) 列表，按分数降序
        """
        raise NotImplementedError("请实现 rerank 方法")


class PrincipleRankReranker(BaseReranker):
    """
    PrincipleRank Reranker 示例实现
    
    TODO: 替换为你的实际模型
    """
    
    def __init__(self, model_path: str = None, use_rubric: bool = True):
        self.model_path = model_path
        self.use_rubric = use_rubric
        self.model = None
        
        # TODO: 加载你的模型
        # from your_module import load_model
        # self.model = load_model(model_path)
        
        print(f"[PrincipleRank] 初始化 (model_path={model_path})")
    
    def _generate_rubric(self, query: str) -> str:
        """
        生成排序准则
        
        TODO: 实现你的 Agentic Rubric 生成逻辑
        """
        # 示例：返回通用准则
        return """
        评估准则:
        1. 内容相关性: 文档是否直接回答查询问题
        2. 信息完整性: 文档是否提供全面的信息
        3. 信息准确性: 文档内容是否准确可靠
        """
    
    def _parse_ranking(self, output: str, doc_ids: List[str]) -> List[Tuple[str, float]]:
        """
        解析模型输出的排序结果
        
        模型输出格式: [3] > [1] > [5] > [2] > [4]
        """
        import re
        
        # 提取 [数字] 模式
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, output)
        
        if not matches:
            # 解析失败，返回原顺序
            return [(did, 1.0 / (i + 1)) for i, did in enumerate(doc_ids)]
        
        # 转换为排序结果
        results = []
        for rank, idx_str in enumerate(matches):
            idx = int(idx_str) - 1  # 1-indexed 转 0-indexed
            if 0 <= idx < len(doc_ids):
                score = 1.0 / (rank + 1)  # 倒数排名分数
                results.append((doc_ids[idx], score))
        
        # 添加未被排序的文档
        ranked_ids = set(r[0] for r in results)
        for did in doc_ids:
            if did not in ranked_ids:
                results.append((did, 0.0))
        
        return results
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        doc_ids: List[str],
        **kwargs
    ) -> List[Tuple[str, float]]:
        """重排序"""
        
        # 生成准则（可选）
        rubric = self._generate_rubric(query) if self.use_rubric else ""
        
        # TODO: 调用你的模型
        # prompt = self._build_prompt(query, documents, rubric)
        # output = self.model.generate(prompt)
        
        # 示例：使用随机排序（请替换为实际模型）
        import random
        indices = list(range(len(doc_ids)))
        random.shuffle(indices)
        output = " > ".join([f"[{i+1}]" for i in indices])
        
        # 解析排序结果
        return self._parse_ranking(output, doc_ids)


# ============================================================================
# 评估函数
# ============================================================================

def calculate_metrics(
    scores: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int] = [1, 5, 10, 25, 50, 100]
) -> Dict:
    """
    计算检索评估指标
    
    Args:
        scores: {query_id: {doc_id: score}}
        qrels: {query_id: {doc_id: relevance}}
        k_values: 评估的 K 值列表
    
    Returns:
        包含各项指标的字典
    """
    ndcg = {f"NDCG@{k}": 0.0 for k in k_values}
    recall = {f"Recall@{k}": 0.0 for k in k_values}
    precision = {f"P@{k}": 0.0 for k in k_values}
    mrr = {"MRR": 0.0}
    
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {map_string, ndcg_string, recall_string, precision_string, "recip_rank"}
    )
    
    eval_scores = evaluator.evaluate(scores)
    
    for query_id in eval_scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += eval_scores[query_id][f"ndcg_cut_{k}"]
            recall[f"Recall@{k}"] += eval_scores[query_id][f"recall_{k}"]
            precision[f"P@{k}"] += eval_scores[query_id][f"P_{k}"]
        mrr["MRR"] += eval_scores[query_id]["recip_rank"]
    
    n_queries = len(eval_scores)
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / n_queries, 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / n_queries, 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / n_queries, 5)
    mrr["MRR"] = round(mrr["MRR"] / n_queries, 5)
    
    return {**ndcg, **recall, **precision, **mrr}


def load_bright_data(
    task: str,
    long_context: bool = False,
    cache_dir: str = DEFAULT_CACHE_DIR
) -> Tuple[Dict, Dict, List, List, Dict]:
    """
    加载 BRIGHT 数据集
    
    Returns:
        queries: {qid: query_text}
        qrels: {qid: {doc_id: relevance}}
        documents: [doc_text, ...]
        doc_ids: [doc_id, ...]
        excluded_ids: {qid: [excluded_doc_ids]}
    """
    print(f"[Data] 加载 BRIGHT 数据: {task} (long_context={long_context})")
    
    # 加载 examples
    examples = load_dataset('xlangai/bright', 'examples', cache_dir=cache_dir)[task]
    
    # 加载 documents
    doc_config = 'long_documents' if long_context else 'documents'
    doc_pairs = load_dataset('xlangai/bright', doc_config, cache_dir=cache_dir)[task]
    
    # 构建数据结构
    queries = {e['id']: e['query'] for e in examples}
    
    gold_key = 'gold_ids_long' if long_context else 'gold_ids'
    qrels = {e['id']: {gid: 1 for gid in e[gold_key]} for e in examples}
    
    excluded_ids = {e['id']: e['excluded_ids'] for e in examples}
    
    documents = [dp['content'] for dp in doc_pairs]
    doc_ids = [dp['id'] for dp in doc_pairs]
    
    print(f"[Data] 加载完成: {len(queries)} 查询, {len(documents)} 文档")
    
    return queries, qrels, documents, doc_ids, excluded_ids


# ============================================================================
# 主评估流程
# ============================================================================

class BRIGHTEvaluator:
    """BRIGHT Benchmark 评估器"""
    
    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        retriever: Optional[BM25Retriever] = None,
        reranker: Optional[BaseReranker] = None
    ):
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.retriever = retriever or BM25Retriever()
        self.reranker = reranker
        
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_task(
        self,
        task: str,
        long_context: bool = False,
        use_reranker: bool = False,
        top_k_rerank: int = 100
    ) -> EvalResult:
        """评估单个任务"""
        
        print(f"\n{'='*60}")
        print(f"评估任务: {task}")
        print(f"长文档: {long_context}, 使用Reranker: {use_reranker}")
        print(f"{'='*60}")
        
        # 加载数据
        queries, qrels, documents, doc_ids, excluded_ids = load_bright_data(
            task, long_context, self.cache_dir
        )
        
        # 第一阶段：检索
        print("\n[Stage 1] 检索...")
        query_list = list(queries.values())
        query_ids = list(queries.keys())
        
        scores = self.retriever.batch_search(
            queries=query_list,
            query_ids=query_ids,
            documents=documents,
            doc_ids=doc_ids,
            excluded_ids=excluded_ids,
            top_k=1000
        )
        
        # 第二阶段：重排（可选）
        if use_reranker and self.reranker is not None:
            print("\n[Stage 2] 重排...")
            scores = self._rerank_results(
                queries, scores, documents, doc_ids, excluded_ids, top_k_rerank
            )
        
        # 计算指标
        print("\n[Evaluation] 计算指标...")
        metrics = calculate_metrics(scores, qrels)
        
        # 保存结果
        result_dir = os.path.join(self.output_dir, f"{task}_long_{long_context}")
        os.makedirs(result_dir, exist_ok=True)
        
        with open(os.path.join(result_dir, "scores.json"), "w") as f:
            json.dump(scores, f, indent=2)
        
        with open(os.path.join(result_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n结果: nDCG@10={metrics['NDCG@10']:.4f}, MRR={metrics['MRR']:.4f}")
        
        return EvalResult(
            task=task,
            ndcg_at_10=metrics['NDCG@10'],
            recall_at_1=metrics['Recall@1'],
            recall_at_10=metrics['Recall@10'],
            mrr=metrics['MRR'],
            full_metrics=metrics
        )
    
    def _rerank_results(
        self,
        queries: Dict[str, str],
        retrieval_scores: Dict[str, Dict[str, float]],
        documents: List[str],
        doc_ids: List[str],
        excluded_ids: Dict[str, List[str]],
        top_k: int
    ) -> Dict[str, Dict[str, float]]:
        """使用 reranker 重排检索结果"""
        
        doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
        reranked_scores = {}
        
        for qid, query in tqdm(queries.items(), desc="重排"):
            # 获取 top_k 候选
            query_scores = retrieval_scores.get(qid, {})
            sorted_candidates = sorted(query_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            candidate_doc_ids = [c[0] for c in sorted_candidates]
            candidate_docs = [documents[doc_id_to_idx[did]] for did in candidate_doc_ids]
            
            # 重排
            reranked = self.reranker.rerank(query, candidate_docs, candidate_doc_ids)
            
            # 更新分数
            reranked_scores[qid] = {did: score for did, score in reranked}
        
        return reranked_scores
    
    def evaluate_all(
        self,
        tasks: List[str] = None,
        long_context: bool = False,
        use_reranker: bool = False
    ) -> List[EvalResult]:
        """评估多个任务"""
        
        if tasks is None:
            tasks = BRIGHT_TASKS["long_doc" if long_context else "short_doc"]
        
        results = []
        for task in tasks:
            try:
                result = self.evaluate_task(task, long_context, use_reranker)
                results.append(result)
            except Exception as e:
                print(f"[Error] 任务 {task} 失败: {e}")
        
        return results
    
    def generate_submission_report(
        self,
        results: List[EvalResult],
        model_name: str = "PrincipleRank",
        model_description: str = ""
    ) -> str:
        """生成提交报告"""
        
        # 计算平均分
        avg_ndcg = np.mean([r.ndcg_at_10 for r in results])
        avg_recall1 = np.mean([r.recall_at_1 for r in results])
        avg_mrr = np.mean([r.mrr for r in results])
        
        report = f"""
# BRIGHT Benchmark 提交报告

## 基本信息

- **模型名称**: {model_name}
- **评估日期**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **评估任务数**: {len(results)}

## 平均分数

| 指标 | 分数 |
|------|------|
| **nDCG@10** | **{avg_ndcg * 100:.2f}** |
| Recall@1 | {avg_recall1 * 100:.2f} |
| MRR | {avg_mrr * 100:.2f} |

## 各任务详细分数

| 任务 | nDCG@10 | Recall@10 | MRR |
|------|---------|-----------|-----|
"""
        for r in results:
            report += f"| {r.task} | {r.ndcg_at_10 * 100:.2f} | {r.recall_at_10 * 100:.2f} | {r.mrr * 100:.2f} |\n"
        
        report += f"""
## 模型描述

{model_description if model_description else "请填写模型描述"}

## 提交说明

将此报告和结果发送至: **suhongjin96@gmail.com**

邮件需包含:
1. ✅ 12个数据集的 nDCG@10 分数
2. ✅ 模型描述（名称、大小、是否使用LLM推理等）
3. ⬜ 开源代码链接（推荐）

---

*报告生成时间: {datetime.now().isoformat()}*
"""
        
        # 保存报告
        report_path = os.path.join(self.output_dir, "SUBMISSION_REPORT.md")
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"\n提交报告已保存: {report_path}")
        
        return report


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="BRIGHT Benchmark 评估器")
    
    parser.add_argument("--method", type=str, default="bm25",
                        choices=["bm25", "custom"],
                        help="评估方法: bm25 (基线) 或 custom (自定义reranker)")
    
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="要评估的任务列表")
    
    parser.add_argument("--all", action="store_true",
                        help="评估所有任务")
    
    parser.add_argument("--quick", action="store_true",
                        help="快速测试 (仅3个任务)")
    
    parser.add_argument("--long_context", action="store_true",
                        help="使用长文档设置")
    
    parser.add_argument("--reranker_path", type=str, default=None,
                        help="Reranker 模型路径")
    
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR,
                        help="数据缓存目录")
    
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="结果输出目录")
    
    parser.add_argument("--generate_report", action="store_true",
                        help="生成提交报告")
    
    args = parser.parse_args()
    
    # 确定评估任务
    if args.quick:
        tasks = ["biology", "economics", "leetcode"]
    elif args.all:
        tasks = BRIGHT_TASKS["long_doc" if args.long_context else "short_doc"]
    else:
        tasks = args.tasks or ["biology"]
    
    # 初始化 reranker
    reranker = None
    use_reranker = False
    if args.method == "custom":
        reranker = PrincipleRankReranker(model_path=args.reranker_path)
        use_reranker = True
    
    # 创建评估器
    evaluator = BRIGHTEvaluator(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        reranker=reranker
    )
    
    # 运行评估
    results = evaluator.evaluate_all(
        tasks=tasks,
        long_context=args.long_context,
        use_reranker=use_reranker
    )
    
    # 生成报告
    if args.generate_report or args.all:
        report = evaluator.generate_submission_report(
            results,
            model_name="PrincipleRank" if args.method == "custom" else "BM25 Baseline"
        )
        print(report)


if __name__ == "__main__":
    main()
