# BRIGHT Benchmark 评估与提交指南

本目录包含向 [BRIGHT Benchmark](https://brightbenchmark.github.io/) 提交的完整代码和文档。

---

## 📋 目录

- [快速开始](#-快速开始)
- [完整流程图](#-完整流程图)
- [数据格式说明](#-数据格式说明)
- [如何替换为你的模型](#-如何替换为你的模型)
- [评估指标详解](#-评估指标详解)
- [生成提交报告](#-生成提交报告)
- [提交方法](#-提交方法)
- [常见问题](#-常见问题)

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /Users/changhao/xhs_paper/PrincipleReasonRank

# 创建环境 (推荐)
conda create -n principlerank python=3.10 -y
conda activate principlerank

# 安装依赖
pip install datasets pytrec_eval rank_bm25 tqdm numpy transformers torch vllm
```

### 2. 运行 BM25 基线

```bash
cd bright_eval

# 快速测试 (3个任务)
python bright_evaluator.py --method bm25 --quick

# 全量评估 (12个任务)
python bright_evaluator.py --method bm25 --all --generate_report
```

### 3. 使用你的 Reranker

```bash
# 替换模型后运行
python bright_evaluator.py --method custom --reranker_path /path/to/your/model --all
```

---

## 🔄 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BRIGHT 评估与提交流程                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │ 1. 数据加载   │ ─► │ 2. 第一阶段   │ ─► │ 3. 第二阶段   │ ─► │ 4. 评估    │ │
│  │              │    │    检索       │    │    重排       │    │            │ │
│  │ load_dataset │    │    BM25      │    │  你的模型     │    │ pytrec_eval│ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│         │                  │                    │                  │        │
│         ▼                  ▼                    ▼                  ▼        │
│    12个数据集          Top-1000候选        重排Top-100        nDCG@10 等    │
│                                                                             │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │ 5. 生成报告   │ ─► │ 6. 发送邮件   │ ─► │ 7. 上榜!      │                  │
│  │              │    │              │    │              │                  │
│  │ SUBMISSION_  │    │ suhongjin96  │    │ BRIGHT       │                  │
│  │ REPORT.md    │    │ @gmail.com   │    │ Leaderboard  │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 数据格式说明

### BRIGHT 数据集结构

```python
# 加载方式
from datasets import load_dataset

# Examples: 包含查询和标注
examples = load_dataset('xlangai/bright', 'examples')['biology']
# 字段:
# - id: 查询ID (str)
# - query: 查询文本 (str)
# - gold_ids: 相关文档ID列表 (List[str])
# - gold_ids_long: 长文档设置的相关ID (List[str])
# - excluded_ids: 需排除的文档ID (List[str])

# Documents: 文档库
documents = load_dataset('xlangai/bright', 'documents')['biology']
# 字段:
# - id: 文档ID (str)
# - content: 文档内容 (str)
```

### 12个评估任务

| 任务名 | 领域 | 查询数 | 文档数 | 难度 |
|--------|------|--------|--------|------|
| biology | 生物学 | ~100 | ~57K | 中 |
| earth_science | 地球科学 | ~80 | ~35K | 中 |
| economics | 经济学 | ~100 | ~40K | 中 |
| psychology | 心理学 | ~90 | ~45K | 中 |
| robotics | 机器人 | ~70 | ~30K | 中 |
| stackoverflow | 编程问答 | ~150 | ~60K | 高 |
| sustainable_living | 可持续生活 | ~60 | ~25K | 低 |
| leetcode | 算法题 | ~140 | ~50K | 高 |
| pony | 编程语言 | ~50 | ~20K | 中 |
| aops | 数学竞赛 | ~100 | ~40K | 高 |
| theoremqa_questions | 定理问答 | ~80 | ~30K | 高 |
| theoremqa_theorems | 定理 | ~70 | ~25K | 高 |

---

## 🔧 如何替换为你的模型

### 方法1：修改 `PrincipleRankReranker` 类

编辑 `bright_evaluator.py` 中的 `PrincipleRankReranker` 类：

```python
class PrincipleRankReranker(BaseReranker):
    """你的 PrincipleRank Reranker 实现"""
    
    def __init__(self, model_path: str = None, use_rubric: bool = True):
        self.model_path = model_path
        self.use_rubric = use_rubric
        
        # ============================================
        # TODO: 在这里加载你的模型
        # ============================================
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            max_model_len=32768,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
        )
        
        print(f"[PrincipleRank] 模型加载完成: {model_path}")
    
    def _build_prompt(self, query: str, documents: List[str], rubric: str) -> str:
        """
        构建输入 Prompt
        
        根据你的模型训练格式调整
        """
        # ============================================
        # TODO: 根据你的训练格式调整 prompt
        # ============================================
        doc_str = "\n".join([f"[{i+1}] {doc[:500]}" for i, doc in enumerate(documents)])
        
        prompt = f"""你是一个专业的文档排序专家。请根据查询和评估准则，对候选文档进行排序。

查询: {query}

评估准则:
{rubric}

候选文档:
{doc_str}

请按相关性从高到低排序，输出格式为: [3] > [1] > [5] > [2] > [4]

<think>
分析查询意图和各文档的相关性...
</think>
<answer>
"""
        return prompt
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        doc_ids: List[str],
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        执行重排序
        """
        # 生成准则
        rubric = self._generate_rubric(query) if self.use_rubric else ""
        
        # 构建 prompt
        prompt = self._build_prompt(query, documents, rubric)
        
        # ============================================
        # TODO: 调用你的模型
        # ============================================
        outputs = self.llm.generate([prompt], self.sampling_params)
        output_text = outputs[0].outputs[0].text
        
        # 解析排序结果
        return self._parse_ranking(output_text, doc_ids)
```

### 方法2：直接使用已有的 ReasonRank 代码

如果你想复用项目中已有的 `run_rank_llm.py` 逻辑：

```python
# bright_evaluator.py 中添加

import sys
sys.path.append('/Users/changhao/xhs_paper/PrincipleReasonRank')

from run_rank_llm import Arguments
from rerank.rank_listwise_os_llm import RankListwiseOSLLM
from rerank.reranker import Reranker

class ExistingReranker(BaseReranker):
    """使用项目已有的 Reranker"""
    
    def __init__(self, model_path: str):
        args = Arguments(
            model_path=model_path,
            context_size=32768,
            window_size=20,
            # ... 其他参数
        )
        agent = RankListwiseOSLLM(args=args, ...)
        self.reranker = Reranker(agent)
    
    def rerank(self, query, documents, doc_ids, **kwargs):
        # 调用已有的 reranker 逻辑
        ...
```

### 方法3：使用缓存的检索结果

如果你已经有检索结果（在 `runs/` 目录），可以直接加载：

```python
# 加载已有的检索结果
def load_cached_retrieval_results(task: str, method: str = "bm25_gpt4cot") -> Dict:
    """
    加载缓存的检索结果
    
    文件路径: runs/{task}/{method}_top100.txt
    """
    result_path = f"../runs/{task}/{method}_top100.txt"
    
    results = {}
    with open(result_path, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if qid not in results:
                results[qid] = {}
            results[qid][docid] = float(score)
    
    return results
```

---

## 📈 评估指标详解

### 主要指标

| 指标 | 全称 | 说明 | BRIGHT 使用 |
|------|------|------|-------------|
| **nDCG@10** | Normalized Discounted Cumulative Gain | 主要排序指标，考虑位置加权 | ✅ 主指标 |
| Recall@1 | 召回率@1 | Top-1 命中率 | ✅ 长文档指标 |
| MRR | Mean Reciprocal Rank | 第一个正确结果的平均排名倒数 | 参考 |

### 分数计算示例

```python
# 假设查询有3个相关文档: doc_A, doc_B, doc_C
# 你的模型返回排序: [doc_X, doc_A, doc_Y, doc_B, doc_Z, ...]

# nDCG@10 计算:
# DCG = rel_1/log2(2) + rel_2/log2(3) + rel_3/log2(4) + ...
#     = 0/1 + 1/1.58 + 0/2 + 1/2.32 + ...
# IDCG (理想情况) = 1/1 + 1/1.58 + 1/2 + ...
# nDCG = DCG / IDCG

# 代码中的计算
from pytrec_eval import RelevanceEvaluator

qrels = {"q1": {"doc_A": 1, "doc_B": 1, "doc_C": 1}}
scores = {"q1": {"doc_X": 0.9, "doc_A": 0.8, "doc_Y": 0.7, "doc_B": 0.6, "doc_Z": 0.5}}

evaluator = RelevanceEvaluator(qrels, {"ndcg_cut.10"})
results = evaluator.evaluate(scores)
print(results["q1"]["ndcg_cut_10"])  # 例如: 0.7523
```

---

## 📝 生成提交报告

### 自动生成

```bash
python bright_evaluator.py --method bm25 --all --generate_report
```

会在 `outputs/` 目录生成 `SUBMISSION_REPORT.md`。

### 报告模板

```markdown
# BRIGHT Benchmark 提交报告

## 基本信息

- **模型名称**: PrincipleRank-7B
- **基座模型**: Qwen2.5-7B-Instruct
- **训练方法**: SFT + DPO
- **是否使用推理**: 是 (CoT)

## 分数汇总

| 任务 | nDCG@10 |
|------|---------|
| biology | 12.34 |
| economics | 15.67 |
| ... | ... |
| **平均** | **XX.XX** |

## 模型描述

PrincipleRank 是一个准则引导的生成式排序模型...

## 开源链接

GitHub: https://github.com/xxx/PrincipleRank
```

---

## 📧 提交方法

### Step 1: 完成评估

```bash
# 确保评估了所有12个任务
python bright_evaluator.py --method custom --all --generate_report
```

### Step 2: 检查输出文件

```bash
outputs/
├── biology_long_False/
│   ├── scores.json      # 每个查询的分数
│   └── metrics.json     # 评估指标
├── economics_long_False/
│   └── ...
├── ...
└── SUBMISSION_REPORT.md  # 提交报告
```

### Step 3: 发送邮件

**收件人**: suhongjin96@gmail.com

**邮件主题**: BRIGHT Benchmark Submission - [你的模型名称]

**邮件内容**:

```
Hi,

I would like to submit my model to the BRIGHT benchmark.

Model Name: PrincipleRank-7B
Model Size: 7B parameters
Uses LLM Reasoning: Yes

## Results

| Task | nDCG@10 |
|------|---------|
| biology | XX.XX |
| economics | XX.XX |
| ... | ... |
| Average | XX.XX |

## Model Description

[简要描述你的模型]

## Code Repository (Optional)

https://github.com/xxx/xxx

Best regards,
[你的名字]
```

**附件**: 
- `SUBMISSION_REPORT.md`
- （可选）完整的 metrics.json 文件

### Step 4: 等待上榜

提交后通常 1-3 个工作日会被添加到 [Leaderboard](https://brightbenchmark.github.io/)。

---

## ❓ 常见问题

### Q1: 数据下载很慢怎么办？

```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
python bright_evaluator.py --all
```

### Q2: GPU 内存不足？

```python
# 减小 batch size
self.llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,  # 降低内存使用
    max_model_len=16384,  # 减小上下文长度
)
```

### Q3: 如何使用长文档设置？

```bash
# 添加 --long_context 参数
python bright_evaluator.py --method custom --all --long_context
```

长文档设置使用 Recall@1 作为主要指标（而非 nDCG@10）。

### Q4: 分数很低怎么办？

1. **检查输出格式**: 确保模型输出可以被正确解析为 `[3] > [1] > [5] > ...` 格式
2. **检查推理链**: 如果使用 CoT，确保 `<think>...</think>` 和 `<answer>...</answer>` 格式正确
3. **调试单个查询**: 打印模型输入输出，检查是否合理

```python
# 调试代码
def debug_single_query(query, docs, model):
    prompt = build_prompt(query, docs)
    print("=== INPUT ===")
    print(prompt[:1000])
    
    output = model.generate(prompt)
    print("=== OUTPUT ===")
    print(output)
    
    ranking = parse_ranking(output)
    print("=== PARSED RANKING ===")
    print(ranking)
```

### Q5: 能否只提交部分任务的分数？

不行。BRIGHT 要求提交所有 12 个短文档任务的分数才能上榜。

---

## 📁 文件清单

```
bright_eval/
├── README.md                 # 本文档
├── bright_evaluator.py       # 主评估脚本
├── custom_retriever.py       # 旧版检索器（参考）
├── run_custom.py             # 旧版评估脚本（参考）
└── outputs/                  # 评估结果输出
    ├── {task}_long_False/
    │   ├── scores.json
    │   └── metrics.json
    └── SUBMISSION_REPORT.md
```

---

## 🔗 相关链接

- [BRIGHT 官网](https://brightbenchmark.github.io/)
- [BRIGHT 论文](https://arxiv.org/abs/2407.12883)
- [BRIGHT GitHub](https://github.com/xlang-ai/BRIGHT)
- [HuggingFace 数据集](https://huggingface.co/datasets/xlangai/bright)

---

*最后更新: 2026-01-25*
