"""
动态批处理系统 - 根据样本复杂度智能分组
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import time
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ComplexityLevel(Enum):
    """样本复杂度级别"""
    SIMPLE = 0
    MEDIUM = 1
    COMPLEX = 2
    VERY_COMPLEX = 3


@dataclass
class BatchSample:
    """批处理样本"""
    sample_id: int
    image: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    complexity: ComplexityLevel
    estimated_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class ComplexityEstimator:
    """样本复杂度估计器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.complexity_cache = {}
        self.estimation_history = deque(maxlen=1000)
        
        # 复杂度特征权重
        self.feature_weights = {
            'image_resolution': 0.3,
            'text_length': 0.2,
            'visual_complexity': 0.25,
            'semantic_complexity': 0.25
        }
    
    def estimate_complexity(self, sample: BatchSample) -> ComplexityLevel:
        """估计样本复杂度"""
        # 生成缓存键
        cache_key = self._generate_cache_key(sample)
        if cache_key in self.complexity_cache:
            return self.complexity_cache[cache_key]
        
        # 计算复杂度特征
        features = self._extract_complexity_features(sample)
        
        # 计算加权复杂度分数
        complexity_score = 0.0
        for feature_name, weight in self.feature_weights.items():
            complexity_score += features.get(feature_name, 0.0) * weight
        
        # 映射到复杂度级别
        if complexity_score < 0.25:
            level = ComplexityLevel.SIMPLE
        elif complexity_score < 0.5:
            level = ComplexityLevel.MEDIUM
        elif complexity_score < 0.75:
            level = ComplexityLevel.COMPLEX
        else:
            level = ComplexityLevel.VERY_COMPLEX
        
        # 缓存结果
        self.complexity_cache[cache_key] = level
        
        return level
    
    def _extract_complexity_features(self, sample: BatchSample) -> Dict[str, float]:
        """提取复杂度特征"""
        features = {}
        
        # 图像分辨率复杂度 (0-1)
        if sample.image is not None:
            h, w = sample.image.shape[-2:]
            resolution_score = min((h * w) / (512 * 512), 1.0)
            features['image_resolution'] = resolution_score
            
            # 视觉复杂度 (基于图像统计)
            features['visual_complexity'] = self._estimate_visual_complexity(sample.image)
        else:
            features['image_resolution'] = 0.0
            features['visual_complexity'] = 0.0
        
        # 文本长度复杂度 (0-1)
        if sample.input_ids is not None:
            text_length = sample.input_ids.shape[-1]
            length_score = min(text_length / 512, 1.0)
            features['text_length'] = length_score
            
            # 语义复杂度 (基于词汇多样性)
            features['semantic_complexity'] = self._estimate_semantic_complexity(sample.input_ids)
        else:
            features['text_length'] = 0.0
            features['semantic_complexity'] = 0.0
        
        return features
    
    def _estimate_visual_complexity(self, image: torch.Tensor) -> float:
        """估计视觉复杂度"""
        if image.dim() < 3:
            return 0.0
        
        # 计算图像梯度作为复杂度指标
        with torch.no_grad():
            if image.dim() == 4:
                image = image[0]  # 取第一个样本
            
            # 转换为灰度
            if image.shape[0] == 3:
                gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            else:
                gray = image[0] if image.shape[0] > 0 else image
            
            # 计算Sobel梯度
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            
            # 应用卷积
            if gray.dim() == 2:
                gray = gray.unsqueeze(0).unsqueeze(0)
            sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).to(gray.device)
            sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).to(gray.device)
            
            grad_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
            grad_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
            
            # 计算梯度幅度
            gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
            complexity = gradient_magnitude.mean().item()
            
            # 归一化到0-1
            return min(complexity / 100.0, 1.0)
    
    def _estimate_semantic_complexity(self, input_ids: torch.Tensor) -> float:
        """估计语义复杂度"""
        if input_ids.dim() == 0 or input_ids.numel() == 0:
            return 0.0
        
        # 计算词汇多样性 (唯一token比例)
        if input_ids.dim() > 1:
            input_ids = input_ids.flatten()
        
        unique_tokens = torch.unique(input_ids)
        diversity = len(unique_tokens) / max(len(input_ids), 1)
        
        # 计算重复模式复杂度
        repetition_penalty = self._calculate_repetition_penalty(input_ids)
        
        # 综合复杂度
        complexity = (diversity + repetition_penalty) / 2.0
        return min(complexity, 1.0)
    
    def _calculate_repetition_penalty(self, input_ids: torch.Tensor) -> float:
        """计算重复模式惩罚"""
        if len(input_ids) < 3:
            return 0.5
        
        # 检查n-gram重复
        ngram_counts = defaultdict(int)
        for n in [2, 3]:
            for i in range(len(input_ids) - n + 1):
                ngram = tuple(input_ids[i:i+n].tolist())
                ngram_counts[ngram] += 1
        
        # 计算重复率
        total_ngrams = sum(ngram_counts.values())
        repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)
        
        if total_ngrams == 0:
            return 0.5
        
        repetition_rate = repeated_ngrams / total_ngrams
        return 1.0 - repetition_rate  # 重复率低 = 复杂度高
    
    def _generate_cache_key(self, sample: BatchSample) -> str:
        """生成样本缓存键"""
        image_hash = "none"
        if sample.image is not None:
            image_hash = str(hash(sample.image.shape))
        
        text_hash = "none"
        if sample.input_ids is not None:
            text_hash = str(hash(tuple(sample.input_ids.flatten()[:10].tolist())))
        
        return f"{image_hash}_{text_hash}"


class DynamicBatcher:
    """动态批处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.complexity_estimator = ComplexityEstimator(config)
        
        # 批处理参数
        self.max_batch_size = config.get("max_batch_size", 8)
        self.min_batch_size = config.get("min_batch_size", 2)
        self.timeout_ms = config.get("batch_timeout_ms", 100)
        self.complexity_tolerance = config.get("complexity_tolerance", 1)
        
        # 样本队列 - 按复杂度分组
        self.sample_queues = {
            level: deque() for level in ComplexityLevel
        }
        
        # 批处理统计
        self.batch_stats = {
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "complexity_distribution": defaultdict(int),
            "processing_times": defaultdict(list)
        }
        
        self.last_batch_time = time.time()
    
    def add_sample(self, sample: BatchSample):
        """添加样本到批处理队列"""
        # 估计复杂度
        complexity = self.complexity_estimator.estimate_complexity(sample)
        sample.complexity = complexity
        
        # 添加到对应队列
        self.sample_queues[complexity].append(sample)
        
        # 更新统计
        self.batch_stats["complexity_distribution"][complexity] += 1
    
    def get_next_batch(self) -> Optional[List[BatchSample]]:
        """获取下一个优化的批次"""
        current_time = time.time()
        
        # 检查是否超时需要强制出批
        time_since_last = (current_time - self.last_batch_time) * 1000
        force_batch = time_since_last > self.timeout_ms
        
        # 尝试构建同质化批次
        batch = self._build_homogeneous_batch(force_batch)
        
        if batch:
            self.last_batch_time = current_time
            self._update_batch_stats(batch)
            return batch
        
        return None
    
    def _build_homogeneous_batch(self, force_batch: bool = False) -> Optional[List[BatchSample]]:
        """构建同质化批次"""
        best_batch = None
        best_score = -1
        
        # 优先级：简单 > 中等 > 复杂 > 非常复杂
        priority_order = [
            ComplexityLevel.SIMPLE,
            ComplexityLevel.MEDIUM, 
            ComplexityLevel.COMPLEX,
            ComplexityLevel.VERY_COMPLEX
        ]
        
        for complexity in priority_order:
            queue = self.sample_queues[complexity]
            
            if not queue:
                continue
            
            # 计算这个复杂度级别的批次大小
            available_samples = len(queue)
            
            if force_batch and available_samples >= self.min_batch_size:
                # 强制出批
                batch_size = min(available_samples, self.max_batch_size)
                batch = [queue.popleft() for _ in range(batch_size)]
                return batch
            
            elif available_samples >= self.max_batch_size:
                # 满批次
                batch = [queue.popleft() for _ in range(self.max_batch_size)]
                return batch
            
            # 检查是否可以与相邻复杂度混合
            mixed_batch = self._try_mixed_batch(complexity, force_batch)
            if mixed_batch:
                return mixed_batch
        
        return best_batch
    
    def _try_mixed_batch(self, primary_complexity: ComplexityLevel, force_batch: bool) -> Optional[List[BatchSample]]:
        """尝试构建混合复杂度批次"""
        primary_queue = self.sample_queues[primary_complexity]
        primary_count = len(primary_queue)
        
        if primary_count == 0:
            return None
        
        # 确定可以混合的复杂度级别
        mixable_levels = self._get_mixable_levels(primary_complexity)
        
        # 收集可用样本
        available_samples = []
        
        # 添加主要复杂度的样本
        for _ in range(min(primary_count, self.max_batch_size)):
            if primary_queue:
                available_samples.append(primary_queue.popleft())
        
        # 添加可混合复杂度的样本
        remaining_slots = self.max_batch_size - len(available_samples)
        for level in mixable_levels:
            if remaining_slots <= 0:
                break
            
            level_queue = self.sample_queues[level]
            samples_to_add = min(len(level_queue), remaining_slots, self.max_batch_size // 3)
            
            for _ in range(samples_to_add):
                if level_queue:
                    available_samples.append(level_queue.popleft())
                    remaining_slots -= 1
        
        # 检查批次是否满足最小要求
        if len(available_samples) >= self.min_batch_size or force_batch:
            return available_samples
        
        # 如果不满足，将样本放回队列
        for sample in available_samples:
            self.sample_queues[sample.complexity].appendleft(sample)
        
        return None
    
    def _get_mixable_levels(self, primary_complexity: ComplexityLevel) -> List[ComplexityLevel]:
        """获取可以与主要复杂度混合的级别"""
        complexity_values = {
            ComplexityLevel.SIMPLE: 0,
            ComplexityLevel.MEDIUM: 1,
            ComplexityLevel.COMPLEX: 2,
            ComplexityLevel.VERY_COMPLEX: 3
        }
        
        primary_value = complexity_values[primary_complexity]
        mixable = []
        
        # 允许相邻复杂度混合
        for level, value in complexity_values.items():
            if level != primary_complexity and abs(value - primary_value) <= self.complexity_tolerance:
                mixable.append(level)
        
        return mixable
    
    def _update_batch_stats(self, batch: List[BatchSample]):
        """更新批处理统计信息"""
        self.batch_stats["total_batches"] += 1
        
        # 更新平均批次大小
        total_samples = self.batch_stats["total_batches"] * self.batch_stats["avg_batch_size"]
        total_samples += len(batch)
        self.batch_stats["avg_batch_size"] = total_samples / self.batch_stats["total_batches"]
        
        # 记录复杂度分布
        for sample in batch:
            self.batch_stats["complexity_distribution"][sample.complexity] += 1
    
    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        status = {}
        total_pending = 0
        
        for level, queue in self.sample_queues.items():
            count = len(queue)
            status[level.name] = count
            total_pending += count
        
        status["total_pending"] = total_pending
        status["estimated_wait_time"] = self._estimate_wait_time()
        
        return status
    
    def _estimate_wait_time(self) -> float:
        """估计等待时间"""
        total_pending = sum(len(queue) for queue in self.sample_queues.values())
        
        if total_pending == 0:
            return 0.0
        
        # 基于历史处理时间估计
        avg_processing_time = self._get_avg_processing_time()
        estimated_batches = (total_pending + self.max_batch_size - 1) // self.max_batch_size
        
        return estimated_batches * avg_processing_time
    
    def _get_avg_processing_time(self) -> float:
        """获取平均处理时间"""
        all_times = []
        for times in self.batch_stats["processing_times"].values():
            all_times.extend(times)
        
        return np.mean(all_times) if all_times else 1.0
    
    def record_batch_processing_time(self, batch: List[BatchSample], processing_time: float):
        """记录批次处理时间"""
        if batch:
            # 记录主要复杂度的处理时间
            complexity_counts = defaultdict(int)
            for sample in batch:
                complexity_counts[sample.complexity] += 1
            
            # 找到主要复杂度
            primary_complexity = max(complexity_counts.items(), key=lambda x: x[1])[0]
            
            # 记录处理时间
            self.batch_stats["processing_times"][primary_complexity].append(processing_time)
            
            # 限制历史记录长度
            if len(self.batch_stats["processing_times"][primary_complexity]) > 100:
                self.batch_stats["processing_times"][primary_complexity] = \
                    self.batch_stats["processing_times"][primary_complexity][-50:]
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """获取批处理统计信息"""
        return dict(self.batch_stats)
    
    def optimize_batch_parameters(self):
        """根据统计信息优化批处理参数"""
        stats = self.get_batch_stats()
        
        # 根据复杂度分布调整批次大小
        complexity_dist = stats["complexity_distribution"]
        total_samples = sum(complexity_dist.values())
        
        if total_samples > 0:
            # 如果简单样本较多，可以增加批次大小
            simple_ratio = complexity_dist.get(ComplexityLevel.SIMPLE, 0) / total_samples
            if simple_ratio > 0.6 and self.max_batch_size < 16:
                self.max_batch_size += 1
                print(f"Increased max_batch_size to {self.max_batch_size} (simple ratio: {simple_ratio:.2f})")
            
            # 如果复杂样本较多，减少批次大小
            complex_ratio = (complexity_dist.get(ComplexityLevel.COMPLEX, 0) + 
                           complexity_dist.get(ComplexityLevel.VERY_COMPLEX, 0)) / total_samples
            if complex_ratio > 0.4 and self.max_batch_size > 2:
                self.max_batch_size = max(self.max_batch_size - 1, 2)
                print(f"Decreased max_batch_size to {self.max_batch_size} (complex ratio: {complex_ratio:.2f})")
    
    def clear_queues(self):
        """清空所有队列"""
        for queue in self.sample_queues.values():
            queue.clear()
    
    def get_pending_count(self) -> int:
        """获取待处理样本数量"""
        return sum(len(queue) for queue in self.sample_queues.values()) 