#!/usr/bin/env python3
"""
BRIGHT Benchmark 自定义模型评估脚本
====================================

一键运行所有任务并生成可提交的结果！

使用方法:
    # 评估单个任务
    python run_custom.py --task biology

    # 评估所有任务
    python run_custom.py --all
    
    # 评估所有任务（长文档设置）
    python run_custom.py --all --long_context
"""

import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
import pytrec_eval

# 导入自定义检索器
from custom_retriever import retrieval_custom


def calculate_retrieval_metrics(results, qrels, k_values=[1, 5, 10, 25, 50, 100]):
    """
    计算检索指标 (从 BRIGHT retrievers.py 复制)
    """
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    mrr = {"MRR": 0}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(qrels,
                                               {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]
        mrr["MRR"] += scores[query_id]["recip_rank"]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)
    mrr["MRR"] = round(mrr["MRR"] / len(scores), 5)

    output = {**ndcg, **_map, **recall, **precision, **mrr}
    print(output)
    return output


# 所有任务列表
ALL_TASKS = [
    'biology', 'earth_science', 'economics', 'psychology', 'robotics',
    'stackoverflow', 'sustainable_living', 'leetcode', 'pony', 'aops',
    'theoremqa_questions', 'theoremqa_theorems'
]

# 简化的任务列表（用于快速测试）
QUICK_TASKS = ['biology', 'economics', 'leetcode']


def run_single_task(
    task: str,
    long_context: bool = False,
    cache_dir: str = 'cache',
    output_dir: str = 'outputs_custom',
    debug: bool = False
):
    """运行单个任务评估"""
    
    print(f"\n{'='*60}")
    print(f"评估任务: {task}")
    print(f"长文档模式: {long_context}")
    print(f"{'='*60}")
    
    # 创建输出目录
    task_output_dir = os.path.join(output_dir, f"{task}_custom_long_{long_context}")
    os.makedirs(task_output_dir, exist_ok=True)
    
    score_file = os.path.join(task_output_dir, 'score.json')
    result_file = os.path.join(task_output_dir, 'results.json')
    
    # 加载数据
    print(f"加载数据集...")
    examples = load_dataset('xlangai/bright', 'examples', cache_dir=cache_dir)[task]
    
    if long_context:
        doc_pairs = load_dataset('xlangai/bright', 'long_documents', cache_dir=cache_dir)[task]
    else:
        doc_pairs = load_dataset('xlangai/bright', 'documents', cache_dir=cache_dir)[task]
    
    # 准备数据
    documents = [dp['content'] for dp in doc_pairs]
    doc_ids = [dp['id'] for dp in doc_pairs]
    
    queries = [e['query'] for e in examples]
    query_ids = [e['id'] for e in examples]
    excluded_ids = {e['id']: e['excluded_ids'] for e in examples}
    
    if debug:
        documents = documents[:50]
        doc_ids = doc_ids[:50]
        queries = queries[:10]
        query_ids = query_ids[:10]
        excluded_ids = {qid: excluded_ids[qid] for qid in query_ids}
    
    print(f"查询数: {len(queries)}")
    print(f"文档数: {len(documents)}")
    
    # 运行检索
    if not os.path.exists(score_file):
        scores = retrieval_custom(
            queries=queries,
            query_ids=query_ids,
            documents=documents,
            doc_ids=doc_ids,
            excluded_ids=excluded_ids,
            task=task,
            cache_dir=cache_dir,
            long_context=long_context,
            use_reranker=True,
            top_k=100
        )
        
        with open(score_file, 'w') as f:
            json.dump(scores, f, indent=2)
        print(f"分数已保存: {score_file}")
    else:
        with open(score_file) as f:
            scores = json.load(f)
        print(f"使用缓存的分数: {score_file}")
    
    # 构建 ground truth
    key = 'gold_ids_long' if long_context else 'gold_ids'
    ground_truth = {}
    for e in examples:
        if debug and e['id'] not in query_ids:
            continue
        ground_truth[e['id']] = {gid: 1 for gid in e[key]}
    
    # 计算指标
    print(f"\n计算评估指标...")
    results = calculate_retrieval_metrics(results=scores, qrels=ground_truth)
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"结果已保存: {result_file}")
    
    return {
        'task': task,
        'ndcg@10': results.get('NDCG@10', 0),
        'recall@1': results.get('Recall@1', 0),
        'recall@10': results.get('Recall@10', 0),
        'mrr': results.get('MRR', 0)
    }


def run_all_tasks(
    long_context: bool = False,
    cache_dir: str = 'cache',
    output_dir: str = 'outputs_custom',
    debug: bool = False,
    quick: bool = False
):
    """运行所有任务评估"""
    
    tasks = QUICK_TASKS if quick else ALL_TASKS
    all_results = []
    
    print(f"\n{'#'*60}")
    print(f"# BRIGHT Benchmark 全量评估")
    print(f"# 任务数: {len(tasks)}")
    print(f"# 长文档模式: {long_context}")
    print(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] ", end="")
        try:
            result = run_single_task(
                task=task,
                long_context=long_context,
                cache_dir=cache_dir,
                output_dir=output_dir,
                debug=debug
            )
            all_results.append(result)
        except Exception as e:
            print(f"任务 {task} 失败: {e}")
            all_results.append({
                'task': task,
                'ndcg@10': 0,
                'recall@1': 0,
                'recall@10': 0,
                'mrr': 0,
                'error': str(e)
            })
    
    # 汇总结果
    summary = generate_summary(all_results, long_context, output_dir)
    
    return all_results, summary


def generate_summary(results: list, long_context: bool, output_dir: str):
    """生成汇总报告"""
    
    # 计算平均分
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        print("没有有效结果！")
        return None
    
    avg_ndcg = sum(r['ndcg@10'] for r in valid_results) / len(valid_results)
    avg_recall1 = sum(r['recall@1'] for r in valid_results) / len(valid_results)
    avg_recall10 = sum(r['recall@10'] for r in valid_results) / len(valid_results)
    avg_mrr = sum(r['mrr'] for r in valid_results) / len(valid_results)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'long_context': long_context,
        'num_tasks': len(valid_results),
        'average_scores': {
            'nDCG@10': round(avg_ndcg * 100, 2),  # 转为百分比
            'Recall@1': round(avg_recall1 * 100, 2),
            'Recall@10': round(avg_recall10 * 100, 2),
            'MRR': round(avg_mrr * 100, 2)
        },
        'per_task_scores': {r['task']: round(r['ndcg@10'] * 100, 2) for r in valid_results}
    }
    
    # 保存汇总
    summary_file = os.path.join(output_dir, f'summary_long_{long_context}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 打印报告
    print(f"\n")
    print(f"{'='*60}")
    print(f"📊 BRIGHT Benchmark 评估结果汇总")
    print(f"{'='*60}")
    print(f"评估时间: {summary['timestamp']}")
    print(f"长文档模式: {long_context}")
    print(f"有效任务数: {len(valid_results)}")
    print(f"")
    print(f"{'─'*60}")
    print(f"平均分数:")
    print(f"  • nDCG@10:   {summary['average_scores']['nDCG@10']:.2f}")
    print(f"  • Recall@1:  {summary['average_scores']['Recall@1']:.2f}")
    print(f"  • Recall@10: {summary['average_scores']['Recall@10']:.2f}")
    print(f"  • MRR:       {summary['average_scores']['MRR']:.2f}")
    print(f"")
    print(f"{'─'*60}")
    print(f"各任务 nDCG@10 分数:")
    for task, score in summary['per_task_scores'].items():
        print(f"  • {task:25s}: {score:.2f}")
    print(f"{'='*60}")
    print(f"")
    print(f"📁 汇总文件: {summary_file}")
    print(f"")
    print(f"{'─'*60}")
    print(f"📧 提交说明:")
    print(f"   将结果发送至: suhongjin96@gmail.com")
    print(f"   主要指标: 平均 nDCG@10 = {summary['average_scores']['nDCG@10']:.2f}")
    print(f"{'='*60}")
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BRIGHT Benchmark 自定义模型评估')
    parser.add_argument('--task', type=str, default=None,
                        help='单个任务名称')
    parser.add_argument('--all', action='store_true',
                        help='评估所有任务')
    parser.add_argument('--quick', action='store_true',
                        help='快速测试（仅3个任务）')
    parser.add_argument('--long_context', action='store_true',
                        help='使用长文档设置')
    parser.add_argument('--cache_dir', type=str, default='cache',
                        help='数据缓存目录')
    parser.add_argument('--output_dir', type=str, default='outputs_custom',
                        help='结果输出目录')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式（使用少量数据）')
    
    args = parser.parse_args()
    
    if args.all or args.quick:
        run_all_tasks(
            long_context=args.long_context,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            debug=args.debug,
            quick=args.quick
        )
    elif args.task:
        run_single_task(
            task=args.task,
            long_context=args.long_context,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            debug=args.debug
        )
    else:
        print("请指定 --task <任务名> 或 --all 来运行评估")
        print(f"可用任务: {', '.join(ALL_TASKS)}")
