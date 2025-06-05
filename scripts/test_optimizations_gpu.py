#!/usr/bin/env python3
"""
GPU性能优化测试脚本
ChartExpert-MoE GPU Performance Test Suite
"""

import torch
import time
import os
import sys
import warnings
import gc
from typing import Dict, List, Tuple, Any
import json
from collections import defaultdict

# 确保能找到项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 忽略警告
warnings.filterwarnings('ignore')

# GPU设置
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
    print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')
    print("⚠️ CUDA not available, using CPU")
    exit(1)

def check_gpu_memory():
    """检查GPU内存使用"""
    allocated = torch.cuda.memory_allocated() / 1024**3
    cached = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return {
        'allocated': allocated,
        'cached': cached, 
        'total': total,
        'free': total - allocated
    }

class GPUPerformanceTracker:
    """GPU性能追踪器"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.memory_usage = defaultdict(list)
        
    def start_timing(self, name: str):
        """开始GPU计时"""
        torch.cuda.synchronize()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        
    def end_timing(self, name: str):
        """结束GPU计时"""
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()
        
        gpu_time = self.start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
        self.metrics[name].append(gpu_time)
        
        # 记录内存使用
        memory_info = check_gpu_memory()
        self.memory_usage[name].append(memory_info['allocated'])
        
    def get_average(self, name: str) -> float:
        """获取平均时间"""
        return sum(self.metrics[name]) / len(self.metrics[name])
        
    def get_throughput(self, name: str, batch_size: int) -> float:
        """计算吞吐量"""
        avg_time = self.get_average(name)
        return batch_size / avg_time
        
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {}
        
        for name, times in self.metrics.items():
            summary[f"{name}_avg"] = sum(times) / len(times)
            summary[f"{name}_std"] = torch.tensor(times).std().item()
            summary[f"{name}_min"] = min(times)
            summary[f"{name}_max"] = max(times)
            
        for name, memory_usage in self.memory_usage.items():
            summary[f"{name}_memory_avg"] = sum(memory_usage) / len(memory_usage)
            summary[f"{name}_memory_max"] = max(memory_usage)
            
        return summary

def test_gpu_memory_management():
    """测试GPU内存管理优化"""
    print("\n💾 Testing GPU Memory Management...")
    
    tracker = GPUPerformanceTracker()
    
    # 测试不同大小的张量分配和释放
    tensor_sizes = [
        (1024, 1024),
        (2048, 2048), 
        (4096, 4096),
        (8192, 4096)
    ]
    
    print("Testing tensor allocation/deallocation:")
    
    for size in tensor_sizes:
        # 测试分配
        tracker.start_timing(f"alloc_{size[0]}x{size[1]}")
        
        tensors = []
        for _ in range(10):
            tensor = torch.randn(*size, device=device)
            tensors.append(tensor)
            
        tracker.end_timing(f"alloc_{size[0]}x{size[1]}")
        
        # 测试释放
        tracker.start_timing(f"free_{size[0]}x{size[1]}")
        
        del tensors
        torch.cuda.empty_cache()
        
        tracker.end_timing(f"free_{size[0]}x{size[1]}")
        
        avg_alloc = tracker.get_average(f"alloc_{size[0]}x{size[1]}")
        avg_free = tracker.get_average(f"free_{size[0]}x{size[1]}")
        
        print(f"  {size}: alloc={avg_alloc*1000:.2f}ms, free={avg_free*1000:.2f}ms")
    
    return tracker.get_summary()

def test_mixed_precision_performance():
    """测试混合精度性能"""
    print("\n⚡ Testing Mixed Precision Performance...")
    
    tracker = GPUPerformanceTracker()
    
    # 创建大型模型
    model_fp32 = torch.nn.Sequential(
        torch.nn.Linear(2048, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 8192),
        torch.nn.ReLU(),
        torch.nn.Linear(8192, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 1024)
    ).to(device)
    
    model_fp16 = model_fp32.half()
    
    batch_sizes = [16, 32, 64, 128]
    
         for batch_size in batch_sizes:
         # 测试FP32
         input_data_fp32 = torch.randn(batch_size, 2048, device=device, dtype=torch.float32)
         
         tracker.start_timing(f"fp32_batch_{batch_size}")
         
         for _ in range(10):
             with torch.no_grad():
                 _ = model_fp32(input_data_fp32)
                 
         tracker.end_timing(f"fp32_batch_{batch_size}")
         
         # 测试FP16
         input_data_fp16 = torch.randn(batch_size, 2048, device=device, dtype=torch.half)
         
         tracker.start_timing(f"fp16_batch_{batch_size}")
         
         for _ in range(10):
             with torch.no_grad():
                 _ = model_fp16(input_data_fp16)
                 
         tracker.end_timing(f"fp16_batch_{batch_size}")
         
         fp32_time = tracker.get_average(f"fp32_batch_{batch_size}")
         fp16_time = tracker.get_average(f"fp16_batch_{batch_size}")
         speedup = fp32_time / fp16_time
         
         print(f"  Batch {batch_size}: FP32={fp32_time*1000:.2f}ms, FP16={fp16_time*1000:.2f}ms, Speedup={speedup:.2f}x")
    
    return tracker.get_summary()

def test_batch_processing_optimization():
    """测试批处理优化"""
    print("\n📦 Testing Batch Processing Optimization...")
    
    tracker = GPUPerformanceTracker()
    
    # 创建简单的卷积模型
    conv_model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((8, 8)),
        torch.nn.Flatten(),
        torch.nn.Linear(128 * 8 * 8, 512)
    ).to(device)
    
    image_sizes = [(224, 224), (384, 384), (512, 512)]
    batch_sizes = [8, 16, 32, 64]
    
    for img_size in image_sizes:
        print(f"\n  Testing image size: {img_size}")
        
        for batch_size in batch_sizes:
            # 预热
            warmup_data = torch.randn(batch_size, 3, *img_size, device=device)
            for _ in range(3):
                with torch.no_grad():
                    _ = conv_model(warmup_data)
            
            torch.cuda.empty_cache()
            
            # 正式测试
            test_data = torch.randn(batch_size, 3, *img_size, device=device)
            
            tracker.start_timing(f"conv_{img_size[0]}_{batch_size}")
            
            for _ in range(20):
                with torch.no_grad():
                    output = conv_model(test_data)
                    
            tracker.end_timing(f"conv_{img_size[0]}_{batch_size}")
            
            avg_time = tracker.get_average(f"conv_{img_size[0]}_{batch_size}")
            throughput = tracker.get_throughput(f"conv_{img_size[0]}_{batch_size}", batch_size * 20)
            
            print(f"    Batch {batch_size}: {avg_time/20*1000:.2f}ms/batch, {throughput:.1f} img/sec")
    
    return tracker.get_summary()

def test_expert_routing_performance():
    """测试专家路由性能"""
    print("\n🧠 Testing Expert Routing Performance...")
    
    tracker = GPUPerformanceTracker()
    
    # 模拟多个专家
    num_experts = 12
    experts = []
    
    for i in range(num_experts):
        expert = torch.nn.Sequential(
            torch.nn.Linear(768, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 768)
        ).to(device)
        experts.append(expert)
    
    # 创建路由器
    router = torch.nn.Linear(768, num_experts).to(device)
    
    batch_sizes = [16, 32, 64]
    
    for batch_size in batch_sizes:
        input_data = torch.randn(batch_size, 768, device=device)
        
        # 测试路由决策
        tracker.start_timing(f"routing_{batch_size}")
        
        for _ in range(50):
            with torch.no_grad():
                # 路由决策
                routing_weights = torch.softmax(router(input_data), dim=-1)
                top_experts = torch.topk(routing_weights, k=2, dim=-1)
                
        tracker.end_timing(f"routing_{batch_size}")
        
        # 测试专家执行
        tracker.start_timing(f"expert_exec_{batch_size}")
        
        for _ in range(50):
            with torch.no_grad():
                # 模拟专家执行
                outputs = []
                for i in range(2):  # top-2 experts
                    expert_output = experts[i](input_data)
                    outputs.append(expert_output)
                
                # 聚合输出
                final_output = torch.stack(outputs, dim=0).mean(dim=0)
                
        tracker.end_timing(f"expert_exec_{batch_size}")
        
        routing_time = tracker.get_average(f"routing_{batch_size}")
        exec_time = tracker.get_average(f"expert_exec_{batch_size}")
        
        print(f"  Batch {batch_size}: routing={routing_time/50*1000:.2f}ms, execution={exec_time/50*1000:.2f}ms")
    
    return tracker.get_summary()

def test_attention_optimization():
    """测试注意力机制优化"""
    print("\n🎯 Testing Attention Optimization...")
    
    tracker = GPUPerformanceTracker()
    
    # 创建多头注意力
    attention = torch.nn.MultiheadAttention(
        embed_dim=768,
        num_heads=12,
        batch_first=True
    ).to(device)
    
    sequence_lengths = [128, 256, 512, 1024]
    batch_sizes = [8, 16, 32]
    
    for seq_len in sequence_lengths:
        print(f"\n  Testing sequence length: {seq_len}")
        
        for batch_size in batch_sizes:
            # 创建输入
            query = torch.randn(batch_size, seq_len, 768, device=device)
            key = torch.randn(batch_size, seq_len, 768, device=device)
            value = torch.randn(batch_size, seq_len, 768, device=device)
            
            # 预热
            for _ in range(3):
                with torch.no_grad():
                    _ = attention(query, key, value)
            
            torch.cuda.empty_cache()
            
            # 正式测试
            tracker.start_timing(f"attention_{seq_len}_{batch_size}")
            
            for _ in range(10):
                with torch.no_grad():
                    output, _ = attention(query, key, value)
                    
            tracker.end_timing(f"attention_{seq_len}_{batch_size}")
            
            avg_time = tracker.get_average(f"attention_{seq_len}_{batch_size}")
            
            print(f"    Batch {batch_size}: {avg_time/10*1000:.2f}ms/forward")
    
    return tracker.get_summary()

def benchmark_full_model():
    """完整模型基准测试"""
    print("\n🚀 Full Model Benchmark...")
    
    tracker = GPUPerformanceTracker()
    
    try:
        # 尝试导入主模型
        from models.chart_expert_moe import ChartExpertMoE
        
        model = ChartExpertMoE(
            input_dim=768,
            hidden_dim=2048,
            num_experts=12,
            num_layers=6,
            device=device
        ).half()  # 使用FP16
        
        test_configs = [
            {'batch_size': 4, 'seq_len': 128},
            {'batch_size': 8, 'seq_len': 128},
            {'batch_size': 16, 'seq_len': 128},
            {'batch_size': 32, 'seq_len': 96},
        ]
        
        for config in test_configs:
            batch_size = config['batch_size']
            seq_len = config['seq_len']
            
            # 创建输入
            image_inputs = torch.randn(batch_size, 3, 512, 512, device=device, dtype=torch.half)
            text_inputs = torch.randint(0, 30000, (batch_size, seq_len), device=device)
            
            # 预热
            for _ in range(3):
                with torch.no_grad():
                    _ = model(image_inputs, text_inputs)
            
            torch.cuda.empty_cache()
            
            # 测试
            tracker.start_timing(f"full_model_{batch_size}")
            
            for _ in range(10):
                with torch.no_grad():
                    outputs = model(image_inputs, text_inputs)
                    
            tracker.end_timing(f"full_model_{batch_size}")
            
            avg_time = tracker.get_average(f"full_model_{batch_size}")
            throughput = tracker.get_throughput(f"full_model_{batch_size}", batch_size * 10)
            
            print(f"  Batch {batch_size}: {avg_time/10*1000:.2f}ms/batch, {throughput:.1f} samples/sec")
        
    except ImportError as e:
        print(f"❌ Could not import full model: {e}")
        
        # 创建简化版本
        simplified_model = torch.nn.Sequential(
            torch.nn.Linear(768, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 1024)
        ).to(device).half()
        
        batch_sizes = [16, 32, 64, 128]
        
        for batch_size in batch_sizes:
            input_data = torch.randn(batch_size, 768, device=device, dtype=torch.half)
            
            # 预热
            for _ in range(3):
                with torch.no_grad():
                    _ = simplified_model(input_data)
            
            torch.cuda.empty_cache()
            
            tracker.start_timing(f"simplified_{batch_size}")
            
            for _ in range(20):
                with torch.no_grad():
                    output = simplified_model(input_data)
                    
            tracker.end_timing(f"simplified_{batch_size}")
            
            avg_time = tracker.get_average(f"simplified_{batch_size}")
            throughput = tracker.get_throughput(f"simplified_{batch_size}", batch_size * 20)
            
            print(f"  Simplified batch {batch_size}: {avg_time/20*1000:.2f}ms/batch, {throughput:.1f} samples/sec")
    
    return tracker.get_summary()

def generate_gpu_report(all_metrics: Dict[str, Any]):
    """生成GPU性能报告"""
    print("\n📊 Generating GPU Performance Report...")
    
    # 创建报告目录
    report_dir = "gpu_performance_reports"
    os.makedirs(report_dir, exist_ok=True)
    
    # 保存原始数据
    with open(f"{report_dir}/gpu_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    # 生成markdown报告
    gpu_info = torch.cuda.get_device_properties(0)
    
    report = f"""# ChartExpert-MoE GPU Performance Report

## GPU Configuration
- **Device**: {torch.cuda.get_device_name()}
- **Memory**: {gpu_info.total_memory / 1024**3:.1f} GB
- **Compute Capability**: {gpu_info.major}.{gpu_info.minor}
- **PyTorch Version**: {torch.__version__}
- **CUDA Version**: {torch.version.cuda}

## Performance Summary

### Memory Efficiency
"""
    
    # 内存使用统计
    memory_metrics = {k: v for k, v in all_metrics.items() if 'memory' in k}
    peak_memory = max([v for k, v in memory_metrics.items() if 'max' in k], default=0)
    avg_memory = sum([v for k, v in memory_metrics.items() if 'avg' in k]) / max(len([k for k in memory_metrics.keys() if 'avg' in k]), 1)
    
    report += f"- **Peak Memory Usage**: {peak_memory:.2f} GB\n"
    report += f"- **Average Memory Usage**: {avg_memory:.2f} GB\n"
    report += f"- **Memory Efficiency**: {(avg_memory/gpu_info.total_memory*1024**3)*100:.1f}%\n\n"
    
    # 性能指标
    report += "### Performance Metrics\n\n"
    
    # 提取关键性能指标
    perf_metrics = {k: v for k, v in all_metrics.items() if 'avg' in k and 'memory' not in k}
    
    for metric, value in sorted(perf_metrics.items()):
        component = metric.replace('_avg', '').replace('_', ' ').title()
        report += f"- **{component}**: {value*1000:.2f}ms\n"
    
    # 计算性能等级
    if peak_memory < gpu_info.total_memory / 1024**3 * 0.5:
        memory_grade = "A"
    elif peak_memory < gpu_info.total_memory / 1024**3 * 0.7:
        memory_grade = "B"
    else:
        memory_grade = "C"
    
    avg_latency = sum(perf_metrics.values()) / len(perf_metrics) if perf_metrics else 0
    if avg_latency < 0.01:  # 10ms
        speed_grade = "A"
    elif avg_latency < 0.05:  # 50ms
        speed_grade = "B"
    else:
        speed_grade = "C"
    
    report += f"""
## Performance Grades
- **Memory Efficiency**: {memory_grade}
- **Speed Performance**: {speed_grade}

## Recommendations

### Memory Optimization
"""

    if memory_grade == "C":
        report += "- Consider reducing batch sizes or model complexity\n"
        report += "- Enable gradient checkpointing for training\n"
    else:
        report += "- Memory usage is within acceptable range\n"
        report += "- Consider increasing batch sizes for better throughput\n"

    report += "\n### Speed Optimization\n"
    
    if speed_grade == "C":
        report += "- Enable mixed precision training (FP16)\n"
        report += "- Optimize data loading pipeline\n"
    else:
        report += "- Performance is good, monitor for regression\n"
        report += "- Consider model parallelism for larger models\n"

    report += f"\n---\n*Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n"
    
    with open(f"{report_dir}/gpu_performance_report.md", "w") as f:
        f.write(report)
    
    print(f"📄 GPU performance report saved to {report_dir}/")

def main():
    """主测试函数"""
    print("🚀 ChartExpert-MoE GPU Performance Test Suite")
    print("=" * 60)
    
    # 显示GPU信息
    print(f"🔥 GPU: {torch.cuda.get_device_name()}")
    print(f"💾 Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"⚡ CUDA Version: {torch.version.cuda}")
    print(f"🐍 PyTorch Version: {torch.__version__}")
    
    # 检查初始内存
    initial_memory = check_gpu_memory()
    print(f"📊 Available Memory: {initial_memory['free']:.2f} GB")
    
    all_metrics = {}
    
    # 测试套件
    test_functions = [
        ("GPU Memory Management", test_gpu_memory_management),
        ("Mixed Precision Performance", test_mixed_precision_performance),
        ("Batch Processing Optimization", test_batch_processing_optimization),
        ("Expert Routing Performance", test_expert_routing_performance),
        ("Attention Optimization", test_attention_optimization),
        ("Full Model Benchmark", benchmark_full_model),
    ]
    
    for test_name, test_func in test_functions:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            
            # 清理内存
            torch.cuda.empty_cache()
            gc.collect()
            
            # 运行测试
            metrics = test_func()
            all_metrics.update(metrics)
            
            # 检查内存使用
            current_memory = check_gpu_memory()
            print(f"💾 Memory after test: {current_memory['allocated']:.2f} GB allocated")
            
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成性能报告
    generate_gpu_report(all_metrics)
    
    # 最终清理
    torch.cuda.empty_cache()
    final_memory = check_gpu_memory()
    print(f"\n📊 Final Memory Usage: {final_memory['allocated']:.2f} GB")
    
    print("\n🎉 GPU performance tests completed successfully!")
    print("Check gpu_performance_reports/ for detailed analysis.")

if __name__ == "__main__":
    main() 