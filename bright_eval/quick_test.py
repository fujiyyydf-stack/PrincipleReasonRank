#!/usr/bin/env python3
"""
快速测试脚本 - 验证环境安装是否正确

运行方式:
    python quick_test.py
"""

import sys

def check_dependencies():
    """检查必要的依赖是否安装"""
    
    print("=" * 60)
    print("BRIGHT Benchmark 环境检查")
    print("=" * 60)
    
    dependencies = {
        "datasets": "datasets (HuggingFace)",
        "pytrec_eval": "pytrec_eval",
        "rank_bm25": "rank_bm25",
        "numpy": "numpy",
        "tqdm": "tqdm",
    }
    
    optional_deps = {
        "transformers": "transformers",
        "torch": "torch",
        "vllm": "vllm",
    }
    
    all_ok = True
    
    print("\n[必要依赖]")
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - 请运行: pip install {module}")
            all_ok = False
    
    print("\n[可选依赖 - 用于 LLM 推理]")
    for module, name in optional_deps.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️ {name} - 未安装 (如需使用自定义模型，请安装)")
    
    return all_ok


def test_data_loading():
    """测试数据加载"""
    
    print("\n" + "=" * 60)
    print("测试数据加载")
    print("=" * 60)
    
    try:
        from datasets import load_dataset
        
        print("\n[加载 BRIGHT 数据集 - biology (可能需要下载)...]")
        
        # 尝试加载一小部分数据
        examples = load_dataset('xlangai/bright', 'examples', split='biology')
        print(f"  ✅ Examples 加载成功: {len(examples)} 条")
        
        documents = load_dataset('xlangai/bright', 'documents', split='biology')
        print(f"  ✅ Documents 加载成功: {len(documents)} 条")
        
        # 显示示例
        sample = examples[0]
        print(f"\n  示例查询: {sample['query'][:100]}...")
        print(f"  相关文档数: {len(sample['gold_ids'])}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 数据加载失败: {e}")
        print("\n  提示: 如果是网络问题，可以尝试:")
        print("    export HF_ENDPOINT=https://hf-mirror.com")
        return False


def test_bm25():
    """测试 BM25 检索器"""
    
    print("\n" + "=" * 60)
    print("测试 BM25 检索器")
    print("=" * 60)
    
    try:
        from rank_bm25 import BM25Okapi
        
        # 简单测试
        corpus = [
            "Hello there good man!",
            "It is quite windy in London",
            "How is the weather today?",
            "The weather in London is rainy",
        ]
        
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        
        query = "weather in London"
        scores = bm25.get_scores(query.split(" "))
        
        print(f"  ✅ BM25 测试成功")
        print(f"  查询: '{query}'")
        print(f"  分数: {scores}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ BM25 测试失败: {e}")
        return False


def test_evaluation():
    """测试评估指标计算"""
    
    print("\n" + "=" * 60)
    print("测试评估指标计算")
    print("=" * 60)
    
    try:
        import pytrec_eval
        
        # 模拟数据
        qrels = {
            "q1": {"doc1": 1, "doc2": 1, "doc3": 0},
            "q2": {"doc4": 1, "doc5": 0},
        }
        
        results = {
            "q1": {"doc1": 0.9, "doc2": 0.7, "doc3": 0.3, "doc4": 0.1},
            "q2": {"doc4": 0.8, "doc5": 0.6, "doc1": 0.2},
        }
        
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.10"})
        metrics = evaluator.evaluate(results)
        
        avg_ndcg = sum(m["ndcg_cut_10"] for m in metrics.values()) / len(metrics)
        
        print(f"  ✅ 评估测试成功")
        print(f"  平均 nDCG@10: {avg_ndcg:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 评估测试失败: {e}")
        return False


def main():
    """主函数"""
    
    print("\n" + "🔍 " * 20)
    print("       BRIGHT Benchmark 环境测试")
    print("🔍 " * 20)
    
    results = []
    
    # 1. 检查依赖
    results.append(("依赖检查", check_dependencies()))
    
    # 2. 测试 BM25
    results.append(("BM25 检索", test_bm25()))
    
    # 3. 测试评估
    results.append(("指标评估", test_evaluation()))
    
    # 4. 测试数据加载 (可选，因为需要下载)
    print("\n是否测试数据加载? (需要下载约 500MB 数据) [y/N]: ", end="")
    try:
        answer = input().strip().lower()
        if answer == 'y':
            results.append(("数据加载", test_data_loading()))
    except:
        pass
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过! 环境配置正确。")
        print("\n下一步:")
        print("  1. 运行基线评估: python bright_evaluator.py --method bm25 --quick")
        print("  2. 替换你的模型: 编辑 bright_evaluator.py 中的 PrincipleRankReranker")
        print("  3. 生成提交报告: python bright_evaluator.py --method custom --all --generate_report")
    else:
        print("⚠️ 部分测试未通过，请根据提示安装缺失的依赖。")
        print("\n快速安装所有依赖:")
        print("  pip install -r requirements.txt")
    
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
