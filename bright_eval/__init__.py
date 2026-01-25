"""
BRIGHT Benchmark 评估模块

提供向 BRIGHT benchmark 提交的完整功能。

使用示例:
    from bright_eval import BRIGHTEvaluator, PrincipleRankReranker
    
    # BM25 基线评估
    evaluator = BRIGHTEvaluator()
    results = evaluator.evaluate_all(tasks=["biology"])
    
    # 使用自定义 reranker
    reranker = PrincipleRankReranker(model_path="your_model")
    evaluator = BRIGHTEvaluator(reranker=reranker)
    results = evaluator.evaluate_all(use_reranker=True)
"""

from .bright_evaluator import (
    BRIGHTEvaluator,
    BM25Retriever,
    BaseReranker,
    PrincipleRankReranker,
    calculate_metrics,
    load_bright_data,
    EvalResult,
    BRIGHT_TASKS,
)

__version__ = "1.0.0"
__all__ = [
    "BRIGHTEvaluator",
    "BM25Retriever", 
    "BaseReranker",
    "PrincipleRankReranker",
    "calculate_metrics",
    "load_bright_data",
    "EvalResult",
    "BRIGHT_TASKS",
]
