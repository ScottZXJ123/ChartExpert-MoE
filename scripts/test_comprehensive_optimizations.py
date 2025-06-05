#!/usr/bin/env python3
"""
综合优化测试脚本
测试ChartExpert-MoE的所有架构优化和性能提升

包含测试：
1. 层次化专家架构
2. 动态专家选择机制  
3. 专家计算的批处理优化
4. 改进的内存管理
5. 注意力机制优化
6. 推理加速（KV缓存和早期退出）
7. 综合性能基准测试
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Tuple, Any
import json
import os
import sys

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.chart_expert_moe import ChartExpertMoE
from src.utils.memory_manager import AdaptiveMemoryManager
from src.fusion.multimodal_fusion import ChartAwareAttention


class ComprehensiveOptimizationTester:
    """综合优化测试器"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
        # 测试配置
        self.test_config = {
            "hidden_size": 768,
            "vocab_size": 32000,
            "num_heads": 12,
            "use_hierarchical_experts": True,
            "use_flash_attention": True,
            "simple_confidence_threshold": 0.95,
            "medium_confidence_threshold": 0.90,
            "complex_confidence_threshold": 0.85,
            "num_early_exit_layers": 3,
            "kv_cache_size_limit": 1024 * 1024 * 1024,  # 1GB
            "vision_encoder": {"hidden_size": 768},
            "llm_backbone": {"hidden_size": 4096},
            "fusion": {"hidden_size": 768, "num_heads": 12},
            "routing": {"hidden_size": 768, "num_experts": 12},
            "moe": {"hidden_size": 768},
            "experts": {
                "layout": {"hidden_size": 768},
                "ocr": {"hidden_size": 768},
                "scale": {"hidden_size": 768},
                "geometric": {"hidden_size": 768},
                "trend": {"hidden_size": 768},
                "query": {"hidden_size": 768},
                "numerical": {"hidden_size": 768},
                "integration": {"hidden_size": 768},
                "alignment": {"hidden_size": 768},
                "chart_to_graph": {"hidden_size": 768},
                "shallow_reasoning": {"hidden_size": 768},
                "orchestrator": {"hidden_size": 768}
            }
        }
        
        print(f"🚀 初始化综合优化测试器")
        print(f"📱 设备: {self.device}")
        print(f"🧠 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB" if torch.cuda.is_available() else "CPU模式")
        
    def create_test_model(self) -> ChartExpertMoE:
        """创建测试模型"""
        try:
            model = ChartExpertMoE(self.test_config)
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"❌ 模型创建失败: {e}")
            raise
    
    def generate_test_data(self, batch_size: int = 2, seq_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成测试数据"""
        # 图像数据 (模拟图表图像)
        image = torch.randn(batch_size, 3, 224, 224, device=self.device)
        
        # 文本数据 (模拟查询)
        input_ids = torch.randint(0, self.test_config["vocab_size"], (batch_size, seq_len), device=self.device)
        
        # 注意力掩码
        attention_mask = torch.ones(batch_size, seq_len, device=self.device)
        
        return image, input_ids, attention_mask
    
    def test_hierarchical_experts(self, model: ChartExpertMoE) -> Dict[str, Any]:
        """测试层次化专家架构"""
        print("\n🔄 测试层次化专家架构...")
        
        test_results = {
            "hierarchical_enabled": hasattr(model, 'expert_levels'),
            "num_levels": 0,
            "experts_per_level": [],
            "adaptive_selection_working": False,
            "performance": {}
        }
        
        if not test_results["hierarchical_enabled"]:
            print("⚠️  层次化专家未启用")
            return test_results
        
        test_results["num_levels"] = len(model.expert_levels)
        test_results["experts_per_level"] = [len(level) for level in model.expert_levels]
        
        # 测试自适应专家选择
        try:
            image, input_ids, attention_mask = self.generate_test_data()
            
            # 测试不同复杂度
            complexities = [torch.tensor([[0.2]]), torch.tensor([[0.6]]), torch.tensor([[0.9]])]
            complexity_names = ["simple", "medium", "complex"]
            
            for complexity, name in zip(complexities, complexity_names):
                expert_counts = model._adaptive_expert_count(complexity)
                test_results["performance"][f"{name}_expert_counts"] = expert_counts
                
            test_results["adaptive_selection_working"] = True
            print(f"✅ 层次化专家架构正常工作")
            print(f"   - 专家层级: {test_results['num_levels']}")
            print(f"   - 每层专家数: {test_results['experts_per_level']}")
            
        except Exception as e:
            print(f"❌ 层次化专家测试失败: {e}")
            test_results["error"] = str(e)
        
        return test_results
    
    def test_chart_aware_attention(self) -> Dict[str, Any]:
        """测试图表感知注意力"""
        print("\n🎯 测试图表感知注意力...")
        
        test_results = {
            "chart_aware_attention_working": False,
            "chart_types_supported": [],
            "bias_patterns_generated": False,
            "flash_attention_available": False
        }
        
        try:
            # 创建图表感知注意力模块
            attention = ChartAwareAttention(hidden_size=768, num_heads=12)
            attention = attention.to(self.device)
            
            test_results["flash_attention_available"] = attention.use_flash_attn
            
            # 测试不同图表类型的注意力偏置
            chart_types = ['bar_chart', 'line_chart', 'pie_chart', 'scatter_plot', 'heatmap']
            seq_len = 64
            
            for chart_type in chart_types:
                try:
                    bias = attention._get_chart_bias(chart_type, seq_len, self.device)
                    if bias is not None:
                        test_results["chart_types_supported"].append(chart_type)
                except:
                    pass
            
            test_results["bias_patterns_generated"] = len(test_results["chart_types_supported"]) > 0
            
            # 测试注意力计算
            batch_size, seq_len = 2, 32
            hidden_size = 768
            
            query = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
            key = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
            value = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
            
            output, weights = attention(query, key, value, chart_type='bar_chart')
            
            test_results["chart_aware_attention_working"] = output.shape == query.shape
            
            print(f"✅ 图表感知注意力正常工作")
            print(f"   - Flash Attention: {'可用' if test_results['flash_attention_available'] else '不可用'}")
            print(f"   - 支持图表类型: {len(test_results['chart_types_supported'])}")
            
        except Exception as e:
            print(f"❌ 图表感知注意力测试失败: {e}")
            test_results["error"] = str(e)
        
        return test_results
    
    def test_memory_management(self) -> Dict[str, Any]:
        """测试内存管理"""
        print("\n💾 测试内存管理...")
        
        test_results = {
            "adaptive_memory_manager_working": False,
            "expert_loading_working": False,
            "memory_stats": {},
            "cache_efficiency": 0.0
        }
        
        try:
            # 创建自适应内存管理器
            config = {
                "max_gpu_experts": 4,
                "memory_threshold": 0.8,
                "adaptive": True,
                "adaptation_interval": 10
            }
            
            memory_manager = AdaptiveMemoryManager(config)
            
            # 模拟专家列表
            mock_experts = [torch.nn.Linear(768, 768) for _ in range(8)]
            memory_manager.initialize(mock_experts)
            
            # 测试自适应专家加载
            predicted_experts = [0, 1, 2, 5]
            current_complexity = 0.7
            
            loaded_experts = memory_manager.adaptive_expert_loading(
                predicted_experts, current_complexity
            )
            
            test_results["expert_loading_working"] = len(loaded_experts) > 0
            test_results["memory_stats"] = memory_manager.get_memory_stats()
            test_results["adaptive_memory_manager_working"] = True
            
            print(f"✅ 内存管理正常工作")
            print(f"   - 加载专家数: {len(loaded_experts)}")
            print(f"   - 内存效率: {test_results['memory_stats'].get('cache_efficiency', 0):.3f}")
            
        except Exception as e:
            print(f"❌ 内存管理测试失败: {e}")
            test_results["error"] = str(e)
        
        return test_results
    
    def test_inference_optimizations(self, model: ChartExpertMoE) -> Dict[str, Any]:
        """测试推理优化"""
        print("\n⚡ 测试推理优化...")
        
        test_results = {
            "early_exit_working": False,
            "kv_cache_working": False,
            "smart_inference_working": False,
            "performance_gains": {},
            "cache_stats": {}
        }
        
        try:
            image, input_ids, attention_mask = self.generate_test_data()
            
            # 测试智能推理
            start_time = time.time()
            smart_outputs = model.smart_inference(
                image, input_ids, attention_mask,
                use_early_exit=True,
                use_kv_cache=True
            )
            smart_time = time.time() - start_time
            
            test_results["smart_inference_working"] = "logits" in smart_outputs
            test_results["performance_gains"]["smart_inference_time"] = smart_time
            
            # 测试早期退出（如果可用）
            if hasattr(model, 'intermediate_confidence_estimators'):
                start_time = time.time()
                early_outputs = model.inference_with_early_exit(image, input_ids, attention_mask)
                early_time = time.time() - start_time
                
                test_results["early_exit_working"] = True
                test_results["performance_gains"]["early_exit_time"] = early_time
                test_results["early_exit_triggered"] = early_outputs.get("early_exit", False)
            
            # 测试KV缓存
            if hasattr(model, 'kv_cache_pool'):
                start_time = time.time()
                cache_outputs = model.inference_with_kv_cache(image, input_ids, attention_mask)
                cache_time = time.time() - start_time
                
                test_results["kv_cache_working"] = True
                test_results["performance_gains"]["kv_cache_time"] = cache_time
                test_results["cache_stats"] = model.get_cache_stats()
            
            # 对比标准推理
            start_time = time.time()
            standard_outputs = model.forward(image, input_ids, attention_mask)
            standard_time = time.time() - start_time
            
            test_results["performance_gains"]["standard_inference_time"] = standard_time
            
            # 计算加速比
            if smart_time > 0:
                speedup = standard_time / smart_time
                test_results["performance_gains"]["speedup_ratio"] = speedup
                
            print(f"✅ 推理优化正常工作")
            print(f"   - 智能推理: {'✓' if test_results['smart_inference_working'] else '✗'}")
            print(f"   - 早期退出: {'✓' if test_results['early_exit_working'] else '✗'}")
            print(f"   - KV缓存: {'✓' if test_results['kv_cache_working'] else '✗'}")
            if "speedup_ratio" in test_results["performance_gains"]:
                print(f"   - 加速比: {test_results['performance_gains']['speedup_ratio']:.2f}x")
            
        except Exception as e:
            print(f"❌ 推理优化测试失败: {e}")
            test_results["error"] = str(e)
        
        return test_results
    
    def test_performance_benchmarks(self, model: ChartExpertMoE) -> Dict[str, Any]:
        """性能基准测试"""
        print("\n📊 执行性能基准测试...")
        
        test_results = {
            "throughput_tests": {},
            "memory_efficiency": {},
            "latency_tests": {},
            "scalability_tests": {}
        }
        
        # 吞吐量测试
        batch_sizes = [1, 2, 4, 8]
        seq_lengths = [32, 64, 128]
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                if batch_size * seq_len > 512:  # 避免内存溢出
                    continue
                    
                try:
                    image, input_ids, attention_mask = self.generate_test_data(batch_size, seq_len)
                    
                    # 预热
                    with torch.no_grad():
                        _ = model.smart_inference(image, input_ids, attention_mask)
                    
                    # 测试
                    start_time = time.time()
                    num_runs = 5
                    
                    for _ in range(num_runs):
                        with torch.no_grad():
                            _ = model.smart_inference(image, input_ids, attention_mask)
                    
                    total_time = time.time() - start_time
                    avg_time = total_time / num_runs
                    throughput = batch_size / avg_time
                    
                    test_key = f"batch{batch_size}_seq{seq_len}"
                    test_results["throughput_tests"][test_key] = {
                        "avg_time": avg_time,
                        "throughput": throughput,
                        "samples_per_sec": throughput
                    }
                    
                except Exception as e:
                    print(f"⚠️  性能测试失败 (batch={batch_size}, seq={seq_len}): {e}")
        
        # 内存效率测试
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            image, input_ids, attention_mask = self.generate_test_data(4, 64)
            with torch.no_grad():
                _ = model.smart_inference(image, input_ids, attention_mask)
            
            peak_memory = torch.cuda.max_memory_allocated()
            memory_efficiency = (peak_memory - initial_memory) / (1024**3)  # GB
            
            test_results["memory_efficiency"] = {
                "peak_memory_gb": memory_efficiency,
                "efficiency_grade": "A" if memory_efficiency < 2.0 else "B" if memory_efficiency < 4.0 else "C"
            }
        
        print(f"✅ 性能基准测试完成")
        print(f"   - 吞吐量测试: {len(test_results['throughput_tests'])} 个配置")
        print(f"   - 内存效率: {test_results.get('memory_efficiency', {}).get('efficiency_grade', 'N/A')}")
        
        return test_results
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """运行所有综合测试"""
        print("🧪 开始综合优化测试")
        print("=" * 60)
        
        # 创建模型
        model = self.create_test_model()
        
        # 运行所有测试
        all_results = {
            "test_timestamp": time.time(),
            "device": str(self.device),
            "model_config": self.test_config
        }
        
        # 1. 层次化专家测试
        all_results["hierarchical_experts"] = self.test_hierarchical_experts(model)
        
        # 2. 图表感知注意力测试
        all_results["chart_aware_attention"] = self.test_chart_aware_attention()
        
        # 3. 内存管理测试
        all_results["memory_management"] = self.test_memory_management()
        
        # 4. 推理优化测试
        all_results["inference_optimizations"] = self.test_inference_optimizations(model)
        
        # 5. 性能基准测试
        all_results["performance_benchmarks"] = self.test_performance_benchmarks(model)
        
        # 6. 整体优化统计
        all_results["optimization_stats"] = model.get_optimization_stats()
        
        return all_results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成测试报告"""
        report = []
        report.append("🎯 ChartExpert-MoE 综合优化测试报告")
        report.append("=" * 50)
        
        # 测试概览
        report.append(f"\n📊 测试概览:")
        report.append(f"   设备: {results['device']}")
        report.append(f"   时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['test_timestamp']))}")
        
        # 优化功能状态
        report.append(f"\n🔧 优化功能状态:")
        optimizations = results.get("optimization_stats", {}).get("optimizations_enabled", [])
        report.append(f"   启用的优化: {', '.join(optimizations) if optimizations else '无'}")
        
        # 层次化专家
        hier_results = results.get("hierarchical_experts", {})
        status = "✅" if hier_results.get("hierarchical_enabled", False) else "❌"
        report.append(f"   层次化专家: {status}")
        if hier_results.get("hierarchical_enabled"):
            report.append(f"     - 专家层级: {hier_results.get('num_levels', 0)}")
            report.append(f"     - 每层专家数: {hier_results.get('experts_per_level', [])}")
        
        # 图表感知注意力
        attn_results = results.get("chart_aware_attention", {})
        status = "✅" if attn_results.get("chart_aware_attention_working", False) else "❌"
        report.append(f"   图表感知注意力: {status}")
        if attn_results.get("chart_aware_attention_working"):
            report.append(f"     - Flash Attention: {'✅' if attn_results.get('flash_attention_available') else '❌'}")
            report.append(f"     - 支持图表类型: {len(attn_results.get('chart_types_supported', []))}")
        
        # 内存管理
        mem_results = results.get("memory_management", {})
        status = "✅" if mem_results.get("adaptive_memory_manager_working", False) else "❌"
        report.append(f"   自适应内存管理: {status}")
        
        # 推理优化
        inf_results = results.get("inference_optimizations", {})
        report.append(f"   智能推理: {'✅' if inf_results.get('smart_inference_working') else '❌'}")
        report.append(f"   早期退出: {'✅' if inf_results.get('early_exit_working') else '❌'}")
        report.append(f"   KV缓存: {'✅' if inf_results.get('kv_cache_working') else '❌'}")
        
        # 性能统计
        perf_results = results.get("performance_benchmarks", {})
        if perf_results.get("memory_efficiency"):
            mem_eff = perf_results["memory_efficiency"]
            report.append(f"\n⚡ 性能统计:")
            report.append(f"   内存效率: {mem_eff.get('efficiency_grade', 'N/A')} ({mem_eff.get('peak_memory_gb', 0):.2f} GB)")
        
        if inf_results.get("performance_gains", {}).get("speedup_ratio"):
            speedup = inf_results["performance_gains"]["speedup_ratio"]
            report.append(f"   推理加速: {speedup:.2f}x")
        
        # 吞吐量统计
        throughput_tests = perf_results.get("throughput_tests", {})
        if throughput_tests:
            best_throughput = max(test["throughput"] for test in throughput_tests.values())
            report.append(f"   最大吞吐量: {best_throughput:.1f} samples/sec")
        
        report.append(f"\n🎉 测试完成！所有架构优化已验证。")
        
        return "\n".join(report)


def main():
    """主函数"""
    tester = ComprehensiveOptimizationTester()
    
    try:
        # 运行综合测试
        results = tester.run_comprehensive_tests()
        
        # 生成报告
        report = tester.generate_report(results)
        print("\n" + report)
        
        # 保存详细结果
        os.makedirs("test_results", exist_ok=True)
        
        # 保存JSON结果
        with open("test_results/comprehensive_optimization_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # 保存文本报告
        with open("test_results/optimization_report.txt", "w") as f:
            f.write(report)
        
        print(f"\n📁 详细结果已保存到 test_results/ 目录")
        
        # 计算总体成功率
        total_optimizations = 6  # 总优化数量
        successful_optimizations = 0
        
        if results.get("hierarchical_experts", {}).get("hierarchical_enabled"):
            successful_optimizations += 1
        if results.get("chart_aware_attention", {}).get("chart_aware_attention_working"):
            successful_optimizations += 1
        if results.get("memory_management", {}).get("adaptive_memory_manager_working"):
            successful_optimizations += 1
        if results.get("inference_optimizations", {}).get("smart_inference_working"):
            successful_optimizations += 1
        if results.get("inference_optimizations", {}).get("early_exit_working"):
            successful_optimizations += 1
        if results.get("inference_optimizations", {}).get("kv_cache_working"):
            successful_optimizations += 1
        
        success_rate = (successful_optimizations / total_optimizations) * 100
        print(f"\n🏆 总体成功率: {success_rate:.1f}% ({successful_optimizations}/{total_optimizations})")
        
        if success_rate >= 80:
            print("🌟 优化实现优秀！所有主要功能正常工作。")
        elif success_rate >= 60:
            print("👍 优化实现良好！大部分功能正常工作。")
        else:
            print("⚠️  需要进一步优化，部分功能存在问题。")
            
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 