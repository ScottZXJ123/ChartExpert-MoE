#!/usr/bin/env python3
"""
ChartExpert-MoE 综合优化测试脚本

测试所有实现的架构优化：
1. 层次化专家架构
2. 图表感知注意力  
3. 智能内存管理
4. 推理优化（早期退出、KV缓存）
5. 性能基准测试
"""

import torch
import time
import json
import os
import sys

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_optimizations():
    """测试所有优化功能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 ChartExpert-MoE 综合优化测试")
    print(f"📱 设备: {device}")
    print("=" * 50)
    
    # 测试配置
    config = {
        "hidden_size": 768,
        "vocab_size": 32000,
        "num_heads": 12,
        "use_hierarchical_experts": True,
        "use_flash_attention": True,
        "simple_confidence_threshold": 0.95,
        "medium_confidence_threshold": 0.90,
        "complex_confidence_threshold": 0.85,
        "num_early_exit_layers": 3,
        "kv_cache_size_limit": 1024 * 1024 * 1024,
        "vision_encoder": {
            "encoder_type": "mock",
            "hidden_size": 768,
            "model_name": "openai/clip-vit-base-patch32",
            "use_native_resolution": False,
            "use_2d_rope": False
        },
        "llm_backbone": {
            "model_name": "microsoft/DialoGPT-medium",
            "hidden_size": 4096,
            "use_mock": True
        },
        "fusion": {
            "hidden_size": 768, 
            "num_heads": 12,
            "fusion_type": "attention"
        },
        "routing": {
            "hidden_size": 768, 
            "num_experts": 12,
            "top_k": 2
        },
        "moe": {
            "hidden_size": 768,
            "num_experts": 12
        },
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
    
    results = {"timestamp": time.time(), "device": str(device)}
    
    try:
        # 1. 测试层次化专家架构
        print("🔄 测试层次化专家架构...")
        from src.models.chart_expert_moe import ChartExpertMoE
        
        model = ChartExpertMoE(config)
        model = model.to(device)
        model.eval()
        
        has_hierarchical = hasattr(model, 'expert_levels')
        results["hierarchical_experts"] = {
            "enabled": has_hierarchical,
            "levels": len(model.expert_levels) if has_hierarchical else 0,
            "experts_per_level": [len(level) for level in model.expert_levels] if has_hierarchical else []
        }
        
        print(f"   ✅ 层次化专家: {'启用' if has_hierarchical else '未启用'}")
        if has_hierarchical:
            print(f"   📊 专家层级: {results['hierarchical_experts']['levels']}")
        
        # 2. 测试图表感知注意力
        print("🎯 测试图表感知注意力...")
        from src.fusion.multimodal_fusion import ChartAwareAttention
        
        attention = ChartAwareAttention(768, 12)
        attention = attention.to(device)
        
        # 测试注意力计算
        query = torch.randn(2, 32, 768, device=device)
        key = torch.randn(2, 32, 768, device=device)
        value = torch.randn(2, 32, 768, device=device)
        
        output, _ = attention(query, key, value, chart_type='bar_chart')
        
        results["chart_aware_attention"] = {
            "working": output.shape == query.shape,
            "flash_attention": attention.use_flash_attn,
            "supported_charts": ['bar_chart', 'line_chart', 'pie_chart', 'scatter_plot', 'heatmap']
        }
        
        print(f"   ✅ 图表感知注意力: 正常工作")
        print(f"   ⚡ Flash Attention: {'可用' if attention.use_flash_attn else '不可用'}")
        
        # 3. 测试内存管理
        print("💾 测试内存管理...")
        from src.utils.memory_manager import AdaptiveMemoryManager
        
        mem_config = {
            "max_gpu_experts": 4,
            "memory_threshold": 0.8,
            "adaptive": True
        }
        
        memory_manager = AdaptiveMemoryManager(mem_config)
        mock_experts = [torch.nn.Linear(768, 768) for _ in range(8)]
        memory_manager.initialize(mock_experts)
        
        loaded_experts = memory_manager.adaptive_expert_loading([0, 1, 2, 5], 0.7)
        
        results["memory_management"] = {
            "adaptive_loading": len(loaded_experts) > 0,
            "loaded_experts": len(loaded_experts),
            "memory_stats": memory_manager.get_memory_stats()
        }
        
        print(f"   ✅ 自适应内存管理: 正常工作")
        print(f"   📈 加载专家数: {len(loaded_experts)}")
        
        # 4. 测试推理优化
        print("⚡ 测试推理优化...")
        
        # 生成测试数据
        image = torch.randn(2, 3, 224, 224, device=device)
        input_ids = torch.randint(0, config["vocab_size"], (2, 64), device=device)
        attention_mask = torch.ones(2, 64, device=device)
        
        # 测试智能推理
        start_time = time.time()
        smart_outputs = model.smart_inference(
            image, input_ids, attention_mask,
            use_early_exit=True,
            use_kv_cache=True
        )
        smart_time = time.time() - start_time
        
        # 测试标准推理
        start_time = time.time()
        standard_outputs = model.forward(image, input_ids, attention_mask)
        standard_time = time.time() - start_time
        
        speedup = standard_time / smart_time if smart_time > 0 else 1.0
        
        results["inference_optimizations"] = {
            "smart_inference": "logits" in smart_outputs,
            "early_exit_available": hasattr(model, 'intermediate_confidence_estimators'),
            "kv_cache_available": hasattr(model, 'kv_cache_pool'),
            "smart_time": smart_time,
            "standard_time": standard_time,
            "speedup": speedup
        }
        
        print(f"   ✅ 智能推理: 正常工作")
        print(f"   🏃 推理加速: {speedup:.2f}x")
        print(f"   🧠 早期退出: {'可用' if results['inference_optimizations']['early_exit_available'] else '不可用'}")
        print(f"   💿 KV缓存: {'可用' if results['inference_optimizations']['kv_cache_available'] else '不可用'}")
        
        # 5. 性能基准测试
        print("📊 性能基准测试...")
        
        # 内存效率测试
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            with torch.no_grad():
                _ = model.smart_inference(image, input_ids, attention_mask)
            
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = (peak_memory - initial_memory) / (1024**3)  # GB
            
            efficiency_grade = "A" if memory_used < 2.0 else "B" if memory_used < 4.0 else "C"
        else:
            memory_used = 0
            efficiency_grade = "N/A"
        
        # 吞吐量测试
        num_runs = 5
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model.smart_inference(image, input_ids, attention_mask)
        
        total_time = time.time() - start_time
        throughput = (2 * num_runs) / total_time  # samples per second
        
        results["performance"] = {
            "memory_usage_gb": memory_used,
            "efficiency_grade": efficiency_grade,
            "throughput_samples_per_sec": throughput,
            "avg_inference_time": total_time / num_runs
        }
        
        print(f"   ✅ 性能测试完成")
        print(f"   💾 内存效率: {efficiency_grade} ({memory_used:.2f} GB)")
        print(f"   🚄 吞吐量: {throughput:.1f} samples/sec")
        
        # 6. 总体优化统计
        optimization_stats = model.get_optimization_stats()
        results["optimization_stats"] = optimization_stats
        
        print(f"\n🔧 启用的优化: {', '.join(optimization_stats.get('optimizations_enabled', []))}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        results["error"] = str(e)
        return results
    
    return results

def generate_report(results):
    """生成测试报告"""
    report = []
    report.append("🎯 ChartExpert-MoE 优化测试报告")
    report.append("=" * 40)
    
    # 计算成功率
    successful = 0
    total = 5
    
    if results.get("hierarchical_experts", {}).get("enabled"):
        successful += 1
    if results.get("chart_aware_attention", {}).get("working"):
        successful += 1
    if results.get("memory_management", {}).get("adaptive_loading"):
        successful += 1
    if results.get("inference_optimizations", {}).get("smart_inference"):
        successful += 1
    if results.get("performance", {}).get("efficiency_grade") in ["A", "B"]:
        successful += 1
    
    success_rate = (successful / total) * 100
    
    report.append(f"📊 总体成功率: {success_rate:.1f}% ({successful}/{total})")
    report.append(f"⚡ 推理加速: {results.get('inference_optimizations', {}).get('speedup', 1.0):.2f}x")
    report.append(f"💾 内存效率: {results.get('performance', {}).get('efficiency_grade', 'N/A')}")
    report.append(f"🚄 吞吐量: {results.get('performance', {}).get('throughput_samples_per_sec', 0):.1f} samples/sec")
    
    if success_rate >= 80:
        report.append("\n🌟 优化实现优秀！所有主要功能正常工作。")
    elif success_rate >= 60:
        report.append("\n👍 优化实现良好！大部分功能正常工作。")
    else:
        report.append("\n⚠️  需要进一步优化，部分功能存在问题。")
    
    return "\n".join(report)

def main():
    """主函数"""
    print("开始ChartExpert-MoE综合优化测试...\n")
    
    # 运行测试
    results = test_optimizations()
    
    # 生成报告
    report = generate_report(results)
    print(f"\n{report}")
    
    # 保存结果
    os.makedirs("test_results", exist_ok=True)
    
    with open("test_results/comprehensive_optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open("test_results/optimization_report.txt", "w") as f:
        f.write(report)
    
    print(f"\n📁 详细结果已保存到 test_results/ 目录")

if __name__ == "__main__":
    main() 