"""
训练效率优化系统
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from collections import defaultdict, deque
import time
import math


class MixedPrecisionTrainer:
    """混合精度训练管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_amp = config.get("use_amp", True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # 精度策略
        self.expert_precision = config.get("expert_precision", {})
        self.default_precision = config.get("default_precision", "fp16")
        
        # 稳定性监控
        self.loss_scale_history = deque(maxlen=100)
        self.overflow_count = 0
        self.underflow_count = 0
    
    def forward_with_amp(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """带自动混合精度的前向传播"""
        if self.use_amp:
            with autocast():
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
        
        return outputs
    
    def backward_with_amp(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """带自动混合精度的反向传播"""
        if self.use_amp:
            self.scaler.scale(loss).backward()
            
            # 检查梯度溢出
            if self.scaler.get_scale() < self.scaler.get_scale():
                self.overflow_count += 1
            
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # 记录缩放因子
            self.loss_scale_history.append(self.scaler.get_scale())
        else:
            loss.backward()
            optimizer.step()
    
    def get_precision_stats(self) -> Dict[str, Any]:
        """获取精度统计信息"""
        stats = {
            "use_amp": self.use_amp,
            "overflow_count": self.overflow_count,
            "underflow_count": self.underflow_count
        }
        
        if self.loss_scale_history:
            stats.update({
                "current_scale": self.loss_scale_history[-1],
                "avg_scale": np.mean(self.loss_scale_history),
                "scale_variance": np.var(self.loss_scale_history)
            })
        
        return stats


class CurriculumLearning:
    """课程学习管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.curriculum_strategy = config.get("curriculum_strategy", "difficulty_based")
        self.stages = config.get("curriculum_stages", 3)
        self.current_stage = 0
        
        # 难度评估
        self.difficulty_weights = {
            "image_resolution": 0.3,
            "text_length": 0.2,
            "complexity_score": 0.3,
            "reasoning_depth": 0.2
        }
        
        # 样本缓存
        self.sample_buffer = defaultdict(list)
        self.difficulty_scores = {}
        
        # 进度跟踪
        self.stage_performance = defaultdict(list)
        self.stage_start_time = time.time()
        self.adaptation_threshold = config.get("adaptation_threshold", 0.8)
    
    def assess_sample_difficulty(self, sample: Dict[str, Any]) -> float:
        """评估样本难度"""
        difficulty = 0.0
        
        # 图像分辨率难度
        if 'image' in sample:
            image = sample['image']
            if isinstance(image, torch.Tensor):
                h, w = image.shape[-2:]
                resolution_difficulty = min((h * w) / (512 * 512), 1.0)
                difficulty += resolution_difficulty * self.difficulty_weights["image_resolution"]
        
        # 文本长度难度
        if 'input_ids' in sample:
            text_length = len(sample['input_ids'])
            length_difficulty = min(text_length / 512, 1.0)
            difficulty += length_difficulty * self.difficulty_weights["text_length"]
        
        # 复杂度评分
        if 'complexity_score' in sample:
            complexity_difficulty = sample['complexity_score']
            difficulty += complexity_difficulty * self.difficulty_weights["complexity_score"]
        
        # 推理深度
        if 'reasoning_depth' in sample:
            reasoning_difficulty = min(sample['reasoning_depth'] / 5.0, 1.0)
            difficulty += reasoning_difficulty * self.difficulty_weights["reasoning_depth"]
        
        return min(difficulty, 1.0)
    
    def should_include_sample(self, sample: Dict[str, Any]) -> bool:
        """判断是否应该包含样本"""
        difficulty = self.assess_sample_difficulty(sample)
        
        # 根据当前阶段决定难度阈值
        if self.curriculum_strategy == "difficulty_based":
            stage_threshold = (self.current_stage + 1) / self.stages
            return difficulty <= stage_threshold
        
        elif self.curriculum_strategy == "progressive":
            # 渐进式增加难度
            max_difficulty = 0.3 + 0.7 * (self.current_stage / max(1, self.stages - 1))
            return difficulty <= max_difficulty
        
        return True
    
    def update_stage_performance(self, accuracy: float, loss: float):
        """更新阶段性能"""
        self.stage_performance[self.current_stage].append({
            "accuracy": accuracy,
            "loss": loss,
            "timestamp": time.time()
        })
        
        # 检查是否应该进入下一阶段
        if self._should_advance_stage():
            self._advance_to_next_stage()
    
    def _should_advance_stage(self) -> bool:
        """判断是否应该进入下一阶段"""
        if self.current_stage >= self.stages - 1:
            return False
        
        current_performance = self.stage_performance[self.current_stage]
        if len(current_performance) < 10:  # 需要足够的样本
            return False
        
        # 计算最近的平均性能
        recent_performance = current_performance[-10:]
        avg_accuracy = np.mean([p["accuracy"] for p in recent_performance])
        
        return avg_accuracy >= self.adaptation_threshold
    
    def _advance_to_next_stage(self):
        """进入下一阶段"""
        self.current_stage += 1
        self.stage_start_time = time.time()
        print(f"Advancing to curriculum stage {self.current_stage + 1}/{self.stages}")
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """获取课程学习统计信息"""
        stats = {
            "current_stage": self.current_stage + 1,
            "total_stages": self.stages,
            "strategy": self.curriculum_strategy,
            "stage_performance": dict(self.stage_performance)
        }
        
        # 计算每个阶段的平均性能
        for stage, performance in self.stage_performance.items():
            if performance:
                avg_acc = np.mean([p["accuracy"] for p in performance])
                avg_loss = np.mean([p["loss"] for p in performance])
                stats[f"stage_{stage}_avg_accuracy"] = avg_acc
                stats[f"stage_{stage}_avg_loss"] = avg_loss
        
        return stats


class GradientAccumulator:
    """梯度累积管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.accumulation_steps = config.get("gradient_accumulation_steps", 4)
        self.current_step = 0
        
        # 自适应累积
        self.adaptive_accumulation = config.get("adaptive_accumulation", True)
        self.min_accumulation_steps = config.get("min_accumulation_steps", 2)
        self.max_accumulation_steps = config.get("max_accumulation_steps", 16)
        
        # 内存监控
        self.memory_threshold = config.get("memory_threshold", 0.9)
        self.batch_sizes_history = deque(maxlen=100)
        
        # 梯度统计
        self.gradient_norms = deque(maxlen=1000)
        self.accumulated_gradients = 0
    
    def should_accumulate(self) -> bool:
        """判断是否应该累积梯度"""
        return self.current_step < self.accumulation_steps - 1
    
    def step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> bool:
        """执行梯度累积步骤"""
        # 缩放损失
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.current_step += 1
        self.accumulated_gradients += 1
        
        # 记录梯度范数
        total_norm = self._compute_gradient_norm(optimizer)
        if total_norm > 0:
            self.gradient_norms.append(total_norm)
        
        # 检查是否应该更新
        if not self.should_accumulate():
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
            
            # 重置累积计数器
            self.current_step = 0
            
            # 自适应调整累积步数
            if self.adaptive_accumulation:
                self._adapt_accumulation_steps()
            
            return True
        
        return False
    
    def _compute_gradient_norm(self, optimizer: torch.optim.Optimizer) -> float:
        """计算梯度范数"""
        total_norm = 0.0
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _adapt_accumulation_steps(self):
        """自适应调整累积步数"""
        if len(self.gradient_norms) < 10:
            return
        
        # 计算梯度方差
        recent_norms = list(self.gradient_norms)[-10:]
        gradient_variance = np.var(recent_norms)
        
        # 检查GPU内存使用
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
            if memory_usage > self.memory_threshold:
                # 内存压力大，增加累积步数
                self.accumulation_steps = min(
                    self.accumulation_steps + 1,
                    self.max_accumulation_steps
                )
            elif memory_usage < 0.5 and gradient_variance < 1.0:
                # 内存充足且梯度稳定，减少累积步数
                self.accumulation_steps = max(
                    self.accumulation_steps - 1,
                    self.min_accumulation_steps
                )
    
    def get_accumulation_stats(self) -> Dict[str, Any]:
        """获取累积统计信息"""
        stats = {
            "accumulation_steps": self.accumulation_steps,
            "current_step": self.current_step,
            "accumulated_gradients": self.accumulated_gradients,
            "adaptive_accumulation": self.adaptive_accumulation
        }
        
        if self.gradient_norms:
            stats.update({
                "avg_gradient_norm": np.mean(self.gradient_norms),
                "gradient_variance": np.var(self.gradient_norms),
                "max_gradient_norm": np.max(self.gradient_norms)
            })
        
        return stats


class AdvancedOptimizer:
    """高级优化器包装器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimizer_type = config.get("optimizer_type", "adamw")
        
        # 学习率调度
        self.lr_scheduler_type = config.get("lr_scheduler", "cosine")
        self.warmup_steps = config.get("warmup_steps", 1000)
        self.max_lr = config.get("max_lr", 5e-5)
        self.min_lr = config.get("min_lr", 1e-6)
        
        # 权重衰减策略
        self.weight_decay_strategy = config.get("weight_decay_strategy", "layerwise")
        self.layerwise_decay_rate = config.get("layerwise_decay_rate", 0.95)
        
        # 优化器状态
        self.step_count = 0
        self.lr_history = deque(maxlen=1000)
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """创建优化器"""
        # 分层权重衰减
        param_groups = self._create_param_groups(model)
        
        if self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.max_lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
        elif self.optimizer_type == "lion":
            # Lion优化器 (如果可用)
            try:
                from lion_pytorch import Lion
                optimizer = Lion(
                    param_groups,
                    lr=self.max_lr * 0.3,  # Lion通常需要更小的学习率
                    betas=(0.9, 0.99),
                    weight_decay=0.01
                )
            except ImportError:
                print("Lion optimizer not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(param_groups, lr=self.max_lr)
        else:
            optimizer = torch.optim.AdamW(param_groups, lr=self.max_lr)
        
        return optimizer
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int):
        """创建学习率调度器"""
        if self.lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps - self.warmup_steps,
                eta_min=self.min_lr
            )
        elif self.lr_scheduler_type == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        
        return scheduler
    
    def _create_param_groups(self, model: nn.Module) -> List[Dict[str, Any]]:
        """创建分层参数组"""
        if self.weight_decay_strategy == "layerwise":
            return self._layerwise_param_groups(model)
        else:
            return [{"params": model.parameters()}]
    
    def _layerwise_param_groups(self, model: nn.Module) -> List[Dict[str, Any]]:
        """创建分层权重衰减参数组"""
        param_groups = []
        
        # 获取模型层数
        layer_names = [name for name, _ in model.named_parameters()]
        unique_layers = set()
        
        for name in layer_names:
            parts = name.split('.')
            if len(parts) > 2:
                layer_prefix = '.'.join(parts[:2])
                unique_layers.add(layer_prefix)
        
        # 为每层分配不同的权重衰减
        for layer_idx, layer_prefix in enumerate(sorted(unique_layers)):
            layer_params = []
            for name, param in model.named_parameters():
                if name.startswith(layer_prefix):
                    layer_params.append(param)
            
            if layer_params:
                # 更深的层使用更小的权重衰减
                decay_rate = self.layerwise_decay_rate ** layer_idx
                param_groups.append({
                    "params": layer_params,
                    "weight_decay": 0.01 * decay_rate,
                    "layer_name": layer_prefix
                })
        
        return param_groups if param_groups else [{"params": model.parameters()}]
    
    def update_learning_rate(self, optimizer: torch.optim.Optimizer, step: int):
        """更新学习率"""
        self.step_count = step
        
        # Warmup阶段
        if step < self.warmup_steps:
            lr = self.max_lr * (step / self.warmup_steps)
        else:
            # Cosine退火
            progress = (step - self.warmup_steps) / max(1, self.step_count - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        # 更新所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        self.lr_history.append(lr)
    
    def get_optimizer_stats(self) -> Dict[str, Any]:
        """获取优化器统计信息"""
        stats = {
            "optimizer_type": self.optimizer_type,
            "step_count": self.step_count,
            "current_lr": self.lr_history[-1] if self.lr_history else 0,
            "weight_decay_strategy": self.weight_decay_strategy
        }
        
        if self.lr_history:
            stats.update({
                "avg_lr": np.mean(self.lr_history),
                "max_lr_reached": np.max(self.lr_history),
                "min_lr_reached": np.min(self.lr_history)
            })
        
        return stats


class TrainingOptimizer:
    """综合训练优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化各个组件
        self.mixed_precision = MixedPrecisionTrainer(config.get("mixed_precision", {}))
        self.curriculum = CurriculumLearning(config.get("curriculum_learning", {}))
        self.gradient_accumulator = GradientAccumulator(config.get("gradient_accumulation", {}))
        self.advanced_optimizer = AdvancedOptimizer(config.get("advanced_optimizer", {}))
        
        # 训练统计
        self.training_stats = {
            "total_steps": 0,
            "effective_batch_size": 0,
            "training_efficiency": 0.0
        }
    
    def optimize_training_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        step: int
    ) -> Dict[str, Any]:
        """优化训练步骤"""
        start_time = time.time()
        
        # 检查样本是否符合课程学习要求
        if not self.curriculum.should_include_sample(batch):
            return {"skipped": True, "reason": "curriculum_learning"}
        
        # 混合精度前向传播
        outputs = self.mixed_precision.forward_with_amp(model, batch)
        loss = outputs.get("loss", outputs.get("logits", torch.tensor(0.0)))
        
        # 梯度累积
        should_update = self.gradient_accumulator.step(loss, optimizer)
        
        # 更新学习率
        self.advanced_optimizer.update_learning_rate(optimizer, step)
        
        # 更新统计信息
        step_time = time.time() - start_time
        self.training_stats["total_steps"] += 1
        
        # 计算训练效率（samples per second）
        if should_update:
            batch_size = len(batch.get("input_ids", [1]))
            self.training_stats["effective_batch_size"] = batch_size * self.gradient_accumulator.accumulation_steps
            self.training_stats["training_efficiency"] = self.training_stats["effective_batch_size"] / step_time
        
        return {
            "loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "step_time": step_time,
            "should_update": should_update,
            "outputs": outputs
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合统计信息"""
        return {
            "training_stats": self.training_stats,
            "mixed_precision": self.mixed_precision.get_precision_stats(),
            "curriculum_learning": self.curriculum.get_curriculum_stats(),
            "gradient_accumulation": self.gradient_accumulator.get_accumulation_stats(),
            "optimizer": self.advanced_optimizer.get_optimizer_stats()
        } 