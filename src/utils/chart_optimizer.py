"""
图表特定优化系统 - 检测图表类型并应用专门处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import numpy as np
from collections import defaultdict


class ChartType(Enum):
    """图表类型枚举"""
    BAR_CHART = "bar"
    LINE_CHART = "line"
    PIE_CHART = "pie"
    SCATTER_PLOT = "scatter"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box"
    AREA_CHART = "area"
    HEATMAP = "heatmap"
    UNKNOWN = "unknown"


class ChartDetector:
    """图表类型检测器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detection_cache = {}
        
        # 检测阈值
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        
        # 形状检测参数
        self.shape_detection_params = {
            'min_line_length': 30,
            'max_line_gap': 10,
            'circle_dp': 1,
            'circle_min_dist': 50,
            'rectangle_epsilon': 0.02,
            'contour_min_area': 100
        }
    
    def detect_chart_type(self, image: torch.Tensor) -> Tuple[ChartType, float]:
        """检测图表类型"""
        # 简化的图表类型检测
        features = self._extract_features(image)
        
        # 基于特征的简单分类
        if features['edge_density'] > 0.3:
            return ChartType.BAR_CHART, 0.8
        elif features['color_variance'] > 0.5:
            return ChartType.PIE_CHART, 0.7
        elif features['gradient_magnitude'] > 0.2:
            return ChartType.LINE_CHART, 0.6
        else:
            return ChartType.UNKNOWN, 0.3
    
    def _extract_features(self, image: torch.Tensor) -> Dict[str, float]:
        """提取图像特征"""
        if image.dim() == 4:
            image = image[0]
        
        # 计算基础特征
        edge_density = torch.std(image).item()
        color_variance = torch.var(image).item()
        gradient_magnitude = torch.mean(torch.abs(torch.diff(image, dim=-1))).item()
        
        return {
            'edge_density': edge_density,
            'color_variance': color_variance,
            'gradient_magnitude': gradient_magnitude
        }
    
    def _multi_strategy_detection(self, image: np.ndarray) -> Dict[ChartType, float]:
        """多策略检测"""
        scores = defaultdict(float)
        
        # 1. 几何形状检测
        geometric_scores = self._geometric_detection(image)
        for chart_type, score in geometric_scores.items():
            scores[chart_type] += score * 0.4
        
        # 2. 颜色分布检测
        color_scores = self._color_distribution_detection(image)
        for chart_type, score in color_scores.items():
            scores[chart_type] += score * 0.3
        
        # 3. 纹理特征检测
        texture_scores = self._texture_detection(image)
        for chart_type, score in texture_scores.items():
            scores[chart_type] += score * 0.3
        
        return dict(scores)
    
    def _geometric_detection(self, image: np.ndarray) -> Dict[ChartType, float]:
        """基于几何形状的检测"""
        scores = defaultdict(float)
        
        # 边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 直线检测 - 适用于条形图、折线图
        lines = cv2.HoughLinesP(
            edges, 
            1, 
            np.pi/180, 
            threshold=50,
            minLineLength=self.shape_detection_params['min_line_length'],
            maxLineGap=self.shape_detection_params['max_line_gap']
        )
        
        if lines is not None:
            horizontal_lines = 0
            vertical_lines = 0
            diagonal_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                if abs(angle) < 10 or abs(angle) > 170:
                    horizontal_lines += 1
                elif abs(angle - 90) < 10 or abs(angle + 90) < 10:
                    vertical_lines += 1
                else:
                    diagonal_lines += 1
            
            # 条形图：大量垂直或水平线
            if vertical_lines > 5 and horizontal_lines > 2:
                scores[ChartType.BAR_CHART] += 0.8
            
            # 折线图：较多对角线
            if diagonal_lines > 3:
                scores[ChartType.LINE_CHART] += 0.7
        
        # 圆形检测 - 适用于饼图
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.shape_detection_params['circle_dp'],
            minDist=self.shape_detection_params['circle_min_dist'],
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=200
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            if len(circles) >= 1:
                scores[ChartType.PIE_CHART] += 0.9
        
        # 矩形检测 - 适用于热力图、箱线图
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangle_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.shape_detection_params['contour_min_area']:
                epsilon = self.shape_detection_params['rectangle_epsilon'] * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:
                    rectangle_count += 1
        
        if rectangle_count > 10:
            scores[ChartType.HEATMAP] += 0.8
        elif rectangle_count > 3:
            scores[ChartType.BOX_PLOT] += 0.6
        
        return scores
    
    def _color_distribution_detection(self, image: np.ndarray) -> Dict[ChartType, float]:
        """基于颜色分布的检测"""
        scores = defaultdict(float)
        
        # 计算颜色直方图
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
        
        # 颜色多样性
        color_diversity = self._calculate_color_diversity(image)
        
        # 饼图：颜色丰富，分段明显
        if color_diversity > 0.7:
            scores[ChartType.PIE_CHART] += 0.6
        
        # 热力图：颜色渐变
        gradient_score = self._detect_color_gradient(image)
        if gradient_score > 0.8:
            scores[ChartType.HEATMAP] += 0.7
        
        # 散点图：点状分布
        point_density = self._detect_point_distribution(image)
        if point_density > 0.6:
            scores[ChartType.SCATTER_PLOT] += 0.7
        
        return scores
    
    def _texture_detection(self, image: np.ndarray) -> Dict[ChartType, float]:
        """基于纹理特征的检测"""
        scores = defaultdict(float)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算LBP纹理
        lbp = self._calculate_lbp(gray)
        
        # 纹理统计
        texture_variance = np.var(lbp)
        texture_entropy = self._calculate_entropy(lbp)
        
        # 条形图：规则纹理，低方差
        if texture_variance < 50 and texture_entropy < 4:
            scores[ChartType.BAR_CHART] += 0.5
        
        # 散点图：高方差，高熵
        if texture_variance > 100 and texture_entropy > 6:
            scores[ChartType.SCATTER_PLOT] += 0.6
        
        return scores
    
    def _calculate_color_diversity(self, image: np.ndarray) -> float:
        """计算颜色多样性"""
        # 转换到HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 计算色调直方图
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        
        # 计算香农熵
        hue_hist = hue_hist.flatten()
        hue_hist = hue_hist / hue_hist.sum()
        entropy = -np.sum(hue_hist * np.log(hue_hist + 1e-10))
        
        # 归一化
        max_entropy = np.log(180)
        return entropy / max_entropy
    
    def _detect_color_gradient(self, image: np.ndarray) -> float:
        """检测颜色渐变"""
        # 计算图像梯度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 计算梯度平滑度
        gradient_std = np.std(gradient_magnitude)
        gradient_mean = np.mean(gradient_magnitude)
        
        if gradient_mean > 0:
            smoothness = 1 - (gradient_std / gradient_mean)
            return max(0, smoothness)
        
        return 0
    
    def _detect_point_distribution(self, image: np.ndarray) -> float:
        """检测点状分布"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用形态学操作检测小点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # 阈值化
        _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 计算点密度
        point_pixels = np.sum(binary > 0)
        total_pixels = binary.size
        
        return point_pixels / total_pixels
    
    def _calculate_lbp(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """计算局部二值模式"""
        height, width = image.shape
        lbp = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center = image[i, j]
                binary_string = ""
                
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    
                    if x >= 0 and x < height and y >= 0 and y < width:
                        if image[x, y] >= center:
                            binary_string += "1"
                        else:
                            binary_string += "0"
                
                lbp[i, j] = int(binary_string, 2)
        
        return lbp
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """计算图像熵"""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        return entropy
    
    def _tensor_to_cv2(self, tensor: torch.Tensor) -> np.ndarray:
        """将张量转换为OpenCV格式"""
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        
        # 转换到0-255范围
        if tensor.max() <= 1.0:
            tensor = tensor * 255
        
        # 转换为numpy数组
        image = tensor.cpu().numpy().astype(np.uint8)
        
        # 确保3通道
        if image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        return image
    
    def _generate_cache_key(self, tensor: torch.Tensor) -> str:
        """生成缓存键"""
        shape_str = "_".join(map(str, tensor.shape))
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
        return f"{shape_str}_{mean_val:.4f}_{std_val:.4f}"


class ChartOptimizer:
    """图表特定优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detector = ChartDetector(config)
        
        # 优化策略
        self.optimization_strategies = {
            ChartType.BAR_CHART: self._optimize_bar_chart,
            ChartType.LINE_CHART: self._optimize_line_chart,
            ChartType.PIE_CHART: self._optimize_pie_chart,
            ChartType.SCATTER_PLOT: self._optimize_scatter_plot,
            ChartType.HEATMAP: self._optimize_heatmap,
            ChartType.BOX_PLOT: self._optimize_box_plot,
        }
        
        # 性能统计
        self.optimization_stats = defaultdict(list)
    
    def optimize_processing(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, Any]:
        """根据图表类型优化处理"""
        # 检测图表类型
        chart_type, confidence = self.detector.detect_chart_type(image)
        
        # 获取优化策略
        if chart_type in self.optimization_strategies:
            optimization_func = self.optimization_strategies[chart_type]
            optimizations = optimization_func(image, input_ids, attention_mask, confidence)
        else:
            optimizations = self._default_optimization(image, input_ids, attention_mask)
        
        # 添加通用优化建议
        optimizations.update({
            "chart_type": chart_type,
            "confidence": confidence,
            "processing_hints": self._get_processing_hints(chart_type)
        })
        
        return optimizations
    
    def _optimize_bar_chart(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        confidence: float
    ) -> Dict[str, Any]:
        """条形图优化"""
        return {
            "suggested_experts": ["layout_expert", "scale_expert", "numerical_expert"],
            "attention_focus": "vertical_structures",
            "resolution_requirements": "medium",
            "processing_priority": "layout_first",
            "complexity_reduction": 0.3,
            "expert_weights": {
                "layout_expert": 0.4,
                "scale_expert": 0.3,
                "numerical_expert": 0.3
            }
        }
    
    def _optimize_line_chart(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        confidence: float
    ) -> Dict[str, Any]:
        """折线图优化"""
        return {
            "suggested_experts": ["trend_expert", "scale_expert", "geometric_expert"],
            "attention_focus": "trend_patterns",
            "resolution_requirements": "high",
            "processing_priority": "trend_first",
            "complexity_reduction": 0.2,
            "expert_weights": {
                "trend_expert": 0.5,
                "scale_expert": 0.3,
                "geometric_expert": 0.2
            }
        }
    
    def _optimize_pie_chart(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        confidence: float
    ) -> Dict[str, Any]:
        """饼图优化"""
        return {
            "suggested_experts": ["geometric_expert", "ocr_expert", "numerical_expert"],
            "attention_focus": "circular_regions",
            "resolution_requirements": "medium",
            "processing_priority": "geometric_first",
            "complexity_reduction": 0.4,
            "expert_weights": {
                "geometric_expert": 0.4,
                "ocr_expert": 0.3,
                "numerical_expert": 0.3
            }
        }
    
    def _optimize_scatter_plot(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        confidence: float
    ) -> Dict[str, Any]:
        """散点图优化"""
        return {
            "suggested_experts": ["geometric_expert", "trend_expert", "scale_expert"],
            "attention_focus": "point_patterns",
            "resolution_requirements": "high",
            "processing_priority": "pattern_detection",
            "complexity_reduction": 0.1,
            "expert_weights": {
                "geometric_expert": 0.4,
                "trend_expert": 0.3,
                "scale_expert": 0.3
            }
        }
    
    def _optimize_heatmap(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        confidence: float
    ) -> Dict[str, Any]:
        """热力图优化"""
        return {
            "suggested_experts": ["scale_expert", "layout_expert", "numerical_expert"],
            "attention_focus": "color_gradients",
            "resolution_requirements": "medium",
            "processing_priority": "scale_first",
            "complexity_reduction": 0.3,
            "expert_weights": {
                "scale_expert": 0.4,
                "layout_expert": 0.3,
                "numerical_expert": 0.3
            }
        }
    
    def _optimize_box_plot(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        confidence: float
    ) -> Dict[str, Any]:
        """箱线图优化"""
        return {
            "suggested_experts": ["geometric_expert", "scale_expert", "numerical_expert"],
            "attention_focus": "box_structures",
            "resolution_requirements": "medium",
            "processing_priority": "structure_first",
            "complexity_reduction": 0.2,
            "expert_weights": {
                "geometric_expert": 0.4,
                "scale_expert": 0.3,
                "numerical_expert": 0.3
            }
        }
    
    def _default_optimization(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, Any]:
        """默认优化策略"""
        return {
            "suggested_experts": ["layout_expert", "ocr_expert", "integration_expert"],
            "attention_focus": "general",
            "resolution_requirements": "medium",
            "processing_priority": "balanced",
            "complexity_reduction": 0.0,
            "expert_weights": {
                "layout_expert": 0.33,
                "ocr_expert": 0.33,
                "integration_expert": 0.34
            }
        }
    
    def _get_processing_hints(self, chart_type: ChartType) -> Dict[str, Any]:
        """获取处理提示"""
        hints = {
            ChartType.BAR_CHART: {
                "focus_regions": ["bars", "axes", "labels"],
                "skip_regions": ["background", "legend"],
                "key_features": ["bar_height", "axis_values"]
            },
            ChartType.LINE_CHART: {
                "focus_regions": ["lines", "points", "axes"],
                "skip_regions": ["background"],
                "key_features": ["trend_direction", "peak_values"]
            },
            ChartType.PIE_CHART: {
                "focus_regions": ["sectors", "center", "labels"],
                "skip_regions": ["background"],
                "key_features": ["sector_angles", "percentages"]
            },
            ChartType.SCATTER_PLOT: {
                "focus_regions": ["points", "axes", "clusters"],
                "skip_regions": ["background"],
                "key_features": ["point_distribution", "correlations"]
            },
            ChartType.HEATMAP: {
                "focus_regions": ["cells", "color_scale", "axes"],
                "skip_regions": ["background"],
                "key_features": ["intensity_values", "patterns"]
            }
        }
        
        return hints.get(chart_type, {})
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        stats = {}
        for chart_type, times in self.optimization_stats.items():
            if times:
                stats[chart_type.name] = {
                    "count": len(times),
                    "avg_time": np.mean(times),
                    "total_time": np.sum(times)
                }
        
        return stats
    
    def update_optimization_strategy(self, chart_type: ChartType, performance_feedback: Dict[str, float]):
        """根据性能反馈更新优化策略"""
        # 基于反馈调整权重
        if chart_type in self.optimization_strategies:
            # 这里可以实现自适应权重调整逻辑
            pass 