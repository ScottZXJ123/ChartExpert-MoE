"""
内存高效的专家管理系统
"""

import torch
import torch.nn as nn
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict, deque
from threading import Lock
import time
import psutil
import gc


class ExpertMemoryManager:
    """内存高效的MoE专家管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_gpu_experts = config.get("max_gpu_experts", 4)
        self.memory_threshold = config.get("memory_threshold", 0.85)  # GPU内存使用阈值
        self.usage_window = config.get("usage_window", 1000)  # 使用历史窗口大小
        
        # 专家状态跟踪
        self.active_experts: Set[int] = set()
        self.expert_usage_history: deque = deque(maxlen=self.usage_window)
        self.expert_usage_count: defaultdict = defaultdict(int)
        self.expert_last_used: Dict[int, float] = {}
        self.expert_load_time: Dict[int, float] = {}
        
        # 内存统计
        self.total_loads = 0
        self.total_offloads = 0
        self.memory_saves = 0
        
        # 线程安全
        self._lock = Lock()
        
        # 专家引用
        self.experts: Optional[List[nn.Module]] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def initialize(self, experts: List[nn.Module]):
        """初始化专家管理器"""
        self.experts = experts
        
        # 初始加载前几个最常用的专家
        for i in range(min(self.max_gpu_experts, len(experts))):
            self._load_expert_to_gpu(i)
    
    def get_expert_for_computation(self, expert_id: int) -> nn.Module:
        """获取用于计算的专家，自动处理加载"""
        with self._lock:
            current_time = time.time()
            
            # 记录使用
            self.expert_usage_history.append((expert_id, current_time))
            self.expert_usage_count[expert_id] += 1
            self.expert_last_used[expert_id] = current_time
            
            # 确保专家在GPU上
            if expert_id not in self.active_experts:
                self._ensure_expert_on_gpu(expert_id)
            
            return self.experts[expert_id]
    
    def _ensure_expert_on_gpu(self, expert_id: int):
        """确保专家在GPU上，必要时进行内存管理"""
        if expert_id in self.active_experts:
            return
        
        # 检查GPU内存使用情况
        if self._should_manage_memory():
            self._free_gpu_memory()
        
        # 加载专家到GPU
        self._load_expert_to_gpu(expert_id)
    
    def _should_manage_memory(self) -> bool:
        """检查是否需要进行内存管理"""
        if len(self.active_experts) < self.max_gpu_experts:
            return False
        
        # 检查GPU内存使用率
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if memory_used > self.memory_threshold:
                return True
        
        return len(self.active_experts) >= self.max_gpu_experts
    
    def _free_gpu_memory(self):
        """释放GPU内存 - 卸载最少使用的专家"""
        if not self.active_experts:
            return
        
        # 选择要卸载的专家 - LRU策略
        lru_expert = self._get_lru_expert()
        if lru_expert is not None:
            self._offload_expert_to_cpu(lru_expert)
    
    def _get_lru_expert(self) -> Optional[int]:
        """获取最少最近使用的专家"""
        if not self.active_experts:
            return None
        
        # 基于最后使用时间的LRU
        current_time = time.time()
        oldest_time = current_time
        lru_expert = None
        
        for expert_id in self.active_experts:
            last_used = self.expert_last_used.get(expert_id, 0)
            if last_used < oldest_time:
                oldest_time = last_used
                lru_expert = expert_id
        
        return lru_expert
    
    def _load_expert_to_gpu(self, expert_id: int):
        """加载专家到GPU"""
        if expert_id in self.active_experts or self.experts is None:
            return
        
        start_time = time.time()
        
        try:
            self.experts[expert_id].to(self.device)
            self.active_experts.add(expert_id)
            self.expert_load_time[expert_id] = time.time() - start_time
            self.total_loads += 1
            
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                # GPU内存不足，强制释放内存
                self._emergency_memory_cleanup()
                # 重试
                self.experts[expert_id].to(self.device)
                self.active_experts.add(expert_id)
            else:
                raise e
    
    def _offload_expert_to_cpu(self, expert_id: int):
        """卸载专家到CPU"""
        if expert_id not in self.active_experts or self.experts is None:
            return
        
        try:
            self.experts[expert_id].to('cpu')
            self.active_experts.remove(expert_id)
            self.total_offloads += 1
            self.memory_saves += 1
            
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error offloading expert {expert_id}: {e}")
    
    def _emergency_memory_cleanup(self):
        """紧急内存清理"""
        # 卸载一半的专家
        experts_to_offload = list(self.active_experts)[:len(self.active_experts)//2]
        
        for expert_id in experts_to_offload:
            self._offload_expert_to_cpu(expert_id)
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def preload_experts(self, expert_ids: List[int]):
        """预加载专家列表"""
        with self._lock:
            for expert_id in expert_ids:
                if len(self.active_experts) < self.max_gpu_experts:
                    if expert_id not in self.active_experts:
                        self._load_expert_to_gpu(expert_id)
                else:
                    break
    
    def optimize_memory_layout(self):
        """基于使用模式优化内存布局"""
        # 计算专家使用频率
        usage_freq = defaultdict(int)
        recent_history = list(self.expert_usage_history)[-self.usage_window//2:]
        
        for expert_id, _ in recent_history:
            usage_freq[expert_id] += 1
        
        # 获取最常用的专家
        most_used = sorted(usage_freq.items(), key=lambda x: x[1], reverse=True)
        target_experts = [expert_id for expert_id, _ in most_used[:self.max_gpu_experts]]
        
        # 卸载不常用的专家
        current_experts = list(self.active_experts)
        for expert_id in current_experts:
            if expert_id not in target_experts:
                self._offload_expert_to_cpu(expert_id)
        
        # 加载常用专家
        for expert_id in target_experts:
            if expert_id not in self.active_experts:
                self._load_expert_to_gpu(expert_id)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存管理统计信息"""
        stats = {
            "active_experts": len(self.active_experts),
            "max_gpu_experts": self.max_gpu_experts,
            "total_loads": self.total_loads,
            "total_offloads": self.total_offloads,
            "memory_saves": self.memory_saves,
            "expert_usage_count": dict(self.expert_usage_count),
            "cache_efficiency": self.memory_saves / max(1, self.total_loads + self.total_offloads)
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_reserved": torch.cuda.memory_reserved(),
                "gpu_memory_max": torch.cuda.max_memory_allocated(),
            })
        
        # 系统内存
        memory = psutil.virtual_memory()
        stats.update({
            "system_memory_used": memory.used,
            "system_memory_available": memory.available,
            "system_memory_percent": memory.percent
        })
        
        return stats
    
    def get_expert_load_times(self) -> Dict[int, float]:
        """获取专家加载时间统计"""
        return dict(self.expert_load_time)
    
    def reset_stats(self):
        """重置统计信息"""
        self.total_loads = 0
        self.total_offloads = 0
        self.memory_saves = 0
        self.expert_usage_count.clear()
        self.expert_usage_history.clear()
        self.expert_last_used.clear()
        self.expert_load_time.clear()
    
    def cleanup(self):
        """清理资源"""
        # 卸载所有GPU专家
        if self.experts is not None:
            current_experts = list(self.active_experts)
            for expert_id in current_experts:
                self._offload_expert_to_cpu(expert_id)
        
        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()


class AdaptiveMemoryManager(ExpertMemoryManager):
    """自适应内存管理器 - 根据工作负载动态调整"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.adaptation_interval = config.get("adaptation_interval", 100)  # 自适应间隔
        self.adaptation_counter = 0
        
        # 自适应参数
        self.load_latency_threshold = config.get("load_latency_threshold", 0.1)  # 加载延迟阈值
        self.memory_pressure_threshold = config.get("memory_pressure_threshold", 0.9)
        
        # 专家重要性和优先级管理
        self.expert_importance_scores = defaultdict(float)
        self.expert_priority_queue = []
        self.importance_decay = config.get("importance_decay", 0.95)
        
        # 任务复杂度历史追踪
        self.complexity_history = deque(maxlen=100)
        self.complexity_weights = {
            'simple': 0.3,
            'medium': 0.7, 
            'complex': 1.0
        }
        
        # CUDA流池用于异步操作
        if torch.cuda.is_available():
            self.loading_streams = [torch.cuda.Stream() for _ in range(4)]
        else:
            self.loading_streams = []
            
        # 预测模型状态
        self.predicted_experts_cache = {}
        self.prediction_accuracy = 0.8
    
    def get_expert_for_computation(self, expert_id: int) -> nn.Module:
        """重写以支持自适应优化"""
        expert = super().get_expert_for_computation(expert_id)
        
        self.adaptation_counter += 1
        if self.adaptation_counter >= self.adaptation_interval:
            self._adapt_memory_strategy()
            self.adaptation_counter = 0
        
        return expert
    
    def _adapt_memory_strategy(self):
        """自适应调整内存策略"""
        stats = self.get_memory_stats()
        
        # 如果加载延迟过高，增加GPU专家数量
        avg_load_time = sum(self.expert_load_time.values()) / max(1, len(self.expert_load_time))
        if avg_load_time > self.load_latency_threshold:
            if self.max_gpu_experts < len(self.experts):
                self.max_gpu_experts = min(self.max_gpu_experts + 1, len(self.experts))
                print(f"Increased max_gpu_experts to {self.max_gpu_experts} due to high load latency")
        
        # 如果内存压力过高，减少GPU专家数量
        if torch.cuda.is_available():
            memory_pressure = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if memory_pressure > self.memory_pressure_threshold:
                if self.max_gpu_experts > 1:
                    self.max_gpu_experts = max(self.max_gpu_experts - 1, 1)
                    print(f"Decreased max_gpu_experts to {self.max_gpu_experts} due to memory pressure")
                    # 立即释放内存
                    self._emergency_memory_cleanup()
        
        # 优化专家布局
        self.optimize_memory_layout()
    
    def adaptive_expert_loading(
        self, 
        predicted_experts: List[int], 
        current_task_complexity: float
    ) -> List[int]:
        """
        根据任务复杂度和专家重要性自适应加载专家
        
        Args:
            predicted_experts: 预测需要的专家ID列表
            current_task_complexity: 当前任务复杂度 [0, 1]
            
        Returns:
            实际加载的专家ID列表
        """
        # 更新复杂度历史
        self.complexity_history.append(current_task_complexity)
        
        # 计算每个专家的综合优先级分数
        expert_scores = {}
        current_time = time.time()
        
        for expert_id in predicted_experts:
            if expert_id >= len(self.experts) or expert_id < 0:
                continue
                
            # 使用频率得分
            usage_freq = self.expert_usage_count.get(expert_id, 0) / max(1, sum(self.expert_usage_count.values()))
            
            # 重要性得分
            importance = self.expert_importance_scores[expert_id]
            
            # 最近使用得分 (recency bias)
            last_used = self.expert_last_used.get(expert_id, 0)
            recency = 1.0 / (current_time - last_used + 1)
            
            # 复杂度适应得分
            complexity_boost = min(2.0, current_task_complexity + 0.5)
            
            # 内存效率得分 (更小的专家有轻微优势)
            memory_cost = self._estimate_expert_memory_usage(expert_id)
            efficiency = 1.0 / (memory_cost + 0.1)
            
            # 综合评分 (权重可调)
            expert_scores[expert_id] = (
                0.3 * usage_freq +           # 使用频率
                0.25 * importance +          # 历史重要性
                0.2 * recency +              # 最近使用
                0.15 * complexity_boost +    # 复杂度适应
                0.1 * efficiency             # 内存效率
            )
        
        # 按评分排序
        sorted_experts = sorted(expert_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 根据可用内存和复杂度决定加载数量
        available_memory = self._get_available_gpu_memory()
        max_concurrent_experts = self._calculate_max_experts(available_memory, current_task_complexity)
        
        loaded_experts = []
        total_memory_used = 0
        
        # 智能加载策略
        for expert_id, score in sorted_experts:
            if len(loaded_experts) >= max_concurrent_experts:
                break
                
            expert_memory = self._estimate_expert_memory_usage(expert_id)
            
            if total_memory_used + expert_memory <= available_memory * 0.9:  # 留10%缓冲
                if expert_id not in self.active_experts:
                    self._async_load_expert_to_gpu(expert_id)
                
                loaded_experts.append(expert_id)
                total_memory_used += expert_memory
                
                # 更新重要性得分
                self.expert_importance_scores[expert_id] *= self.importance_decay
                self.expert_importance_scores[expert_id] += score * 0.1
        
        # 异步卸载不需要的专家
        self._async_unload_unused_experts(loaded_experts)
        
        return loaded_experts
    
    def _calculate_max_experts(self, available_memory: float, task_complexity: float) -> int:
        """根据可用内存和任务复杂度计算最大专家数"""
        # 基础专家数根据复杂度调整
        base_experts = int(2 + task_complexity * 4)  # 2-6个专家
        
        # 根据内存可用性调整
        memory_factor = available_memory / (2 * 1024**3)  # 假设2GB为基准
        memory_adjusted = int(base_experts * min(2.0, memory_factor))
        
        return max(1, min(self.max_gpu_experts, memory_adjusted))
    
    def _get_available_gpu_memory(self) -> float:
        """获取可用GPU内存 (字节)"""
        if not torch.cuda.is_available():
            return float('inf')  # CPU模式下不限制
            
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        
        return total_memory - allocated_memory
    
    def _estimate_expert_memory_usage(self, expert_id: int) -> float:
        """估算专家的内存使用量 (字节)"""
        if expert_id >= len(self.experts) or expert_id < 0:
            return 0.0
        
        # 简化估算：根据专家类型估算
        expert_type_memory = {
            0: 0.8 * 1024**3,  # layout_expert
            1: 1.2 * 1024**3,  # ocr_expert (包含OCR模型)
            2: 0.6 * 1024**3,  # scale_expert
            3: 0.7 * 1024**3,  # geometric_expert
            4: 0.9 * 1024**3,  # trend_expert
            5: 1.0 * 1024**3,  # query_expert
            6: 1.1 * 1024**3,  # numerical_expert
            7: 1.3 * 1024**3,  # integration_expert (复杂)
            8: 1.0 * 1024**3,  # alignment_expert
            9: 1.2 * 1024**3,  # chart_to_graph_expert
            10: 0.8 * 1024**3, # shallow_reasoning_expert
            11: 1.5 * 1024**3  # orchestrator_expert (最复杂)
        }
        
        return expert_type_memory.get(expert_id, 1.0 * 1024**3)
    
    def _async_load_expert_to_gpu(self, expert_id: int):
        """异步加载专家到GPU"""
        if not torch.cuda.is_available() or not self.loading_streams:
            self._load_expert_to_gpu(expert_id)
            return
        
        # 使用可用的CUDA流
        stream_idx = expert_id % len(self.loading_streams)
        stream = self.loading_streams[stream_idx]
        
        with torch.cuda.stream(stream):
            self._load_expert_to_gpu(expert_id)
    
    def _async_unload_unused_experts(self, keep_experts: List[int]):
        """异步卸载不需要的专家"""
        for expert_id in list(self.active_experts):
            if (expert_id not in keep_experts):
                
                # 检查是否可以安全卸载 (不在最近使用列表中)
                current_time = time.time()
                last_used = self.expert_last_used.get(expert_id, 0)
                
                if current_time - last_used > 60:  # 60秒未使用
                    self._offload_expert_to_cpu(expert_id) 