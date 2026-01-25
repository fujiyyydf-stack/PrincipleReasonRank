# PrincipleRank 工作流程详解

本文档详细解释 BRIGHT benchmark 的打分逻辑、数据流程、以及训练/推理时候选集的处理方式。

---

## 一、BRIGHT Benchmark 打分逻辑详解

### 1.1 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BRIGHT 评估流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Query        First-Stage Retriever        Reranker           Evaluation    │
│                                                                             │
│  "How to..."  ──► BM25/Dense Retriever ──► 你的模型 ──►  pytrec_eval        │
│                   返回 Top-100             重排 Top-K      计算 nDCG@10      │
│                                                                             │
│                   {qid: {did: score}}     {qid: {did: score}}               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 分数格式说明

**输入给评估器的格式**：
```python
# scores 字典：每个 query 对应其候选文档的分数
scores = {
    "query_1": {
        "doc_1": 0.95,   # 分数越高，排名越靠前
        "doc_2": 0.82,
        "doc_3": 0.45,
        ...
    },
    "query_2": {...},
}

# qrels 字典：ground truth，1 表示相关，0 表示不相关
qrels = {
    "query_1": {
        "doc_1": 1,   # 这个文档是相关的
        "doc_5": 1,
        ...
    }
}

# 评估
from pytrec_eval import RelevanceEvaluator
evaluator = RelevanceEvaluator(qrels, {"ndcg_cut.10"})
results = evaluator.evaluate(scores)
```

### 1.3 生成式 Reranker 的分数转换问题

**核心问题**：生成式模型输出的是**排序顺序**（如 `[3] > [1] > [5] > [2] > [4]`），而不是数值分数。

**解决方案**：将排序位置转换为分数

```python
def convert_ranking_to_scores(ranking_order: List[str], doc_ids: List[str]) -> Dict[str, float]:
    """
    将排序顺序转换为分数
    
    Args:
        ranking_order: 模型输出的排序，如 ["doc_3", "doc_1", "doc_5", ...]
        doc_ids: 所有候选文档ID
    
    Returns:
        scores: {doc_id: score}
    """
    scores = {}
    n = len(ranking_order)
    
    for rank, doc_id in enumerate(ranking_order):
        # 方法1: 倒数排名分数 (常用)
        scores[doc_id] = 1.0 / (rank + 1)
        
        # 方法2: 线性分数
        # scores[doc_id] = (n - rank) / n
        
        # 方法3: 指数衰减
        # scores[doc_id] = math.exp(-0.1 * rank)
    
    # 未被排序的文档给一个很低的分数
    for doc_id in doc_ids:
        if doc_id not in scores:
            scores[doc_id] = 0.0
    
    return scores
```

**在你的代码中的位置**：
`rerank/rank_listwise_os_llm.py` 中的 `sliding_windows` 方法会处理这个转换，候选文档的 `.score` 属性会被更新。

---

## 二、关于 Meta-ranking 校准机制的建议

### 2.1 当前设计的复杂性

你论文中的 Meta-ranking 设计：
```
生成式模型 → 排序 + 推理 → Meta 模型打分 → 加权融合
                 ↓
          Qwen3-Reranker
```

**推理时的问题**：
1. 需要**两次模型调用**（生成式 + 判别式）
2. 需要从生成式输出**解析排序结果**
3. 需要设计**融合权重** α
4. 增加**推理延迟**

### 2.2 建议：简化方案

**方案 A：删除 Meta-ranking（推荐初期采用）**

```markdown
修改论文第 4.4 节，将 Meta-ranking 改为"可选扩展"或"未来工作"：

原文：
**（3）提出了生成-判别混合的 Meta-ranking 校准机制**

修改为：
**（3）设计了自适应推理深度机制**（原 4.5 节内容提升）

或直接删除这一贡献点，聚焦于前两点。
```

**方案 B：简化 Meta-ranking（如果想保留）**

```python
# 简化版：仅在格式解析失败时使用判别式模型
def rerank_with_fallback(query, docs, gen_model, disc_model):
    # 1. 尝试生成式排序
    gen_output = gen_model.generate(query, docs)
    ranking = parse_ranking(gen_output)
    
    # 2. 如果解析成功，直接返回
    if ranking is not None:
        return convert_to_scores(ranking)
    
    # 3. 解析失败，回退到判别式模型
    scores = disc_model.score(query, docs)
    return scores
```

---

## 三、候选集的获取与处理

### 3.1 核心原则：检索只做一次

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         候选集处理流程                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   [一次性检索]                    [缓存文件]           [训练/推理复用]        │
│                                                                             │
│   Query Corpus  ──► Retriever ──► runs/xxx.txt  ──► SFT训练               │
│                      (BM25等)      (TREC格式)    ──► DPO训练               │
│                                                    ──► 推理评估              │
│                                                                             │
│   只执行一次!                                        多次复用                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 你项目中的检索结果位置

```bash
PrincipleReasonRank/
├── runs/                          # 检索结果缓存
│   ├── biology/
│   │   └── bm25_gpt4cot_top100.txt   # BM25 检索结果
│   ├── economics/
│   │   └── bm25_gpt4cot_top100.txt
│   ├── leetcode/
│   │   └── bm25_top100.txt
│   └── ...                        # 其他数据集
```

**文件格式（TREC 格式）**：
```
query_id Q0 doc_id rank score run_name
q1 Q0 doc_123 1 25.4321 bm25
q1 Q0 doc_456 2 24.1234 bm25
q1 Q0 doc_789 3 23.5678 bm25
...
```

### 3.3 如何使用更好的检索器

**当前代码支持的检索方式**（见 `run_rank_llm.py`）：

```python
class RetrievalMethod(Enum):
    BM25 = "bm25"                              # 稀疏检索
    SPLADE_P_P_ENSEMBLE_DISTIL = "SPLADE++..."  # 学习型稀疏
    D_BERT_KD_TASB = "distilbert_tas_b"         # 稠密检索
    E5_MISTRAL = "e5-mistral-7b-instruct"       # 大模型 Embedding
    REASONIR = "reasonir"                       # 推理增强检索
    RaDeR = "RaDeR-..."                         # SOTA 方法
```

**推荐的检索器选择**：

| 场景 | 推荐检索器 | 原因 |
|------|-----------|------|
| 快速基线 | BM25 | 速度快，效果稳定 |
| BRIGHT 数据集 | BM25 + GPT4 CoT | 官方推荐，已有缓存 |
| 追求 SOTA | ReasonIR / RaDeR | 推理增强，效果最好 |

### 3.4 一次检索，多次复用的代码流程

```python
# run_rank_llm.py 中的逻辑

# 1. 检查是否有缓存的检索结果
first_stage_run_path = f'runs/{dataset}/bm25_gpt4cot_top100.txt'

if os.path.exists(first_stage_run_path):
    # 2a. 有缓存，直接加载
    print(f'Loading first stage run from {first_stage_run_path}.')
    results = load_from_trec_file(first_stage_run_path)
else:
    # 2b. 无缓存，执行检索并保存
    results = searcher.batch_search(queries, ...)
    save_to_trec_file(results, first_stage_run_path)

# 3. 使用检索结果进行重排
reranked = reranker.rerank_batch(results)
```

---

## 四、训练数据构造流程

### 4.1 SFT 数据构造

```python
def construct_sft_data(dataset_name: str, retriever: str = "bm25"):
    """
    构造 SFT 训练数据
    
    输入：
    - queries: 查询列表
    - 检索结果: runs/{dataset}/bm25_top100.txt
    
    输出：
    - SFT 训练样本: (query, docs, rubric) → (reasoning, ranking)
    """
    
    # 1. 加载检索结果（只做一次检索）
    retrieval_results = load_retrieval_results(f"runs/{dataset_name}/{retriever}_top100.txt")
    
    # 2. 对每个 query 构造训练样本
    sft_samples = []
    for qid, candidates in retrieval_results.items():
        query = queries[qid]
        
        # 3. 采样候选子集（通常 20-30 个）
        sampled_docs = sample_candidates(candidates, k=20)
        
        # 4. 生成准则（Rubric）
        rubric = generate_rubric(query)  # 用你的 Agentic 管线
        
        # 5. 用强模型生成推理和排序
        reasoning, ranking = strong_model.generate(query, sampled_docs, rubric)
        
        # 6. 构造训练样本
        sft_samples.append({
            "input": format_input(query, sampled_docs, rubric),
            "output": f"<think>{reasoning}</think><answer>{ranking}</answer>"
        })
    
    return sft_samples
```

### 4.2 DPO 数据构造

```python
def construct_dpo_data(sft_model, dataset_name: str):
    """
    构造 DPO 偏好对数据
    """
    
    # 1. 加载相同的检索结果
    retrieval_results = load_retrieval_results(f"runs/{dataset_name}/bm25_top100.txt")
    
    dpo_pairs = []
    for qid, candidates in retrieval_results.items():
        query = queries[qid]
        sampled_docs = sample_candidates(candidates, k=20)
        rubric = get_rubric(qid)  # 使用预生成的 rubric
        
        # 2. 用 SFT 模型采样多个输出
        outputs = []
        for _ in range(8):
            output = sft_model.generate(query, sampled_docs, rubric, temperature=0.8)
            outputs.append(output)
        
        # 3. 用 Verifier 评分
        scores = verifier.score_batch(query, sampled_docs, rubric, outputs)
        
        # 4. 选择最好和较差的作为偏好对
        best_idx = np.argmax(scores)
        worst_idx = np.argmin(scores)
        
        if scores[best_idx] - scores[worst_idx] > threshold:
            dpo_pairs.append({
                "input": format_input(query, sampled_docs, rubric),
                "chosen": outputs[best_idx],
                "rejected": outputs[worst_idx]
            })
    
    return dpo_pairs
```

### 4.3 推理时的流程

```python
def inference(model, dataset_name: str):
    """
    推理评估流程
    """
    
    # 1. 加载相同的检索结果（与训练时一致）
    retrieval_results = load_retrieval_results(f"runs/{dataset_name}/bm25_top100.txt")
    
    all_scores = {}
    for qid, candidates in retrieval_results.items():
        query = queries[qid]
        
        # 2. 可选：加载或生成 rubric
        rubric = get_rubric(qid)  # 或设为 None 如果不使用准则
        
        # 3. 模型重排
        output = model.generate(query, candidates, rubric)
        
        # 4. 解析排序结果，转换为分数
        ranking = parse_ranking(output)
        scores = convert_to_scores(ranking, [c.docid for c in candidates])
        
        all_scores[qid] = scores
    
    # 5. 评估
    metrics = evaluate(all_scores, qrels)
    return metrics
```

---

## 五、推荐的简化实现路径

### 5.1 Phase 1: 基础版本（建议先完成）

```
目标：实现一个能跑通 BRIGHT 评估的基础 reranker

1. 数据准备
   - 使用现有的 BM25 检索结果（runs/ 目录下已有）
   - 暂不生成 Rubric，使用通用指令

2. 模型训练
   - SFT: 用 DeepSeek-R1 生成训练数据
   - 跳过 DPO（初期简化）

3. 评估
   - 在 BRIGHT 3 个任务上测试
   - 对比 BM25 baseline
```

### 5.2 Phase 2: 增强版本

```
目标：加入准则引导和偏好优化

1. Agentic Rubric 生成
   - 实现多智能体管线
   - 生成 Query-Rubric 数据集

2. 训练增强
   - Rubric-aware SFT
   - DPO with Verifier

3. 全量评估
   - BRIGHT 全部 12 个任务
   - 与 ReasonRank 对比
```

### 5.3 Phase 3: 完整版本（可选）

```
目标：实现完整论文方案

1. Meta-ranking（如果需要）
2. 自适应推理深度
3. 消融实验
```

---

## 六、关键代码修改建议

### 6.1 删除 Meta-ranking 后的论文修改

**第 4.4 节整体删除或改为"未来工作"**

**第六章 6.1 节修改**：
```markdown
原文：
**（3）提出了生成-判别混合的 Meta-ranking 校准机制**

修改为：
**（3）设计了高效的推理深度自适应策略**（保持简洁，聚焦核心贡献）

或：
删除这一点，保留两个核心贡献即可。
```

### 6.2 你需要实现的核心代码

```python
# 核心文件：rerank/rank_listwise_os_llm.py

# 需要添加的功能：
# 1. Rubric 注入到 prompt
# 2. 推理输出解析（<think>...<answer>格式）
# 3. 分数转换逻辑
```

---

## 七、总结

| 问题 | 答案 |
|------|------|
| **打分逻辑** | 生成式模型输出排序顺序 → 转换为位置分数 → pytrec_eval 计算 nDCG |
| **Meta-ranking** | 建议初期删除或简化为 fallback 机制，聚焦核心贡献 |
| **检索器** | 只需检索一次，结果缓存在 `runs/` 目录，训练/推理复用 |
| **推荐检索器** | BM25（快速）或 ReasonIR（SOTA），BRIGHT 已有 GPT4-CoT 增强结果 |
| **数据流** | 检索(一次) → 缓存 → SFT数据构造 → DPO数据构造 → 推理评估 |

---

---

## 八、BRIGHT 评估模块

详细的 BRIGHT benchmark 评估和提交指南请参考:

**📁 `bright_eval/` 目录**

```
bright_eval/
├── README.md              # 完整的评估和提交指南
├── bright_evaluator.py    # 主评估脚本
├── quick_test.py          # 环境检查脚本
├── requirements.txt       # 依赖列表
└── __init__.py            # Python 模块
```

**快速使用:**

```bash
# 1. 检查环境
cd bright_eval && python quick_test.py

# 2. 运行 BM25 基线
python bright_evaluator.py --method bm25 --quick

# 3. 使用你的模型评估
python bright_evaluator.py --method custom --reranker_path /path/to/model --all

# 4. 生成提交报告
python bright_evaluator.py --generate_report
```

---

*文档版本：v1.1*  
*更新日期：2026-01-25*
