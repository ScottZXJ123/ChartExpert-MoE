"""
Dataset classes for ChartExpert-MoE

Implements dataset loaders for various chart understanding datasets including
ChartMuseum, ChartQA, and PlotQA.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
import json
import os
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer


class ChartMuseumDataset(Dataset):
    """
    Dataset loader for ChartMuseum benchmark
    
    ChartMuseum is a challenging benchmark for chart understanding that focuses
    on complex visual reasoning tasks.
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        split: str = "test",
        max_length: int = 512,
        image_size: Tuple[int, int] = (224, 224),
        cache_dir: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.image_size = image_size
        
        # Load ChartMuseum dataset
        print(f"Loading ChartMuseum dataset (split: {split})...")
        self.dataset = load_dataset("lytang/ChartMuseum", split=split, cache_dir=cache_dir)
        print(f"Loaded {len(self.dataset)} examples")
        
        # Image preprocessing
        from torchvision import transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example from the dataset"""
        item = self.dataset[idx]
        
        # Extract components
        image = item.get("image")
        question = item.get("question", "")
        answer = item.get("answer", "")
        chart_type = item.get("chart_type", "unknown")
        reasoning_type = item.get("reasoning_type", "unknown")
        
        # Process image
        if image is not None:
            if isinstance(image, str):
                # If image is a path, load it
                image = Image.open(image).convert("RGB")
            processed_image = self.image_transform(image)
        else:
            # Create dummy image if missing
            processed_image = torch.zeros(3, *self.image_size)
        
        # Process text
        prompt = self._create_prompt(question, chart_type)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Process answer for training
        if answer:
            answer_inputs = self.tokenizer(
                answer,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            labels = answer_inputs["input_ids"].squeeze()
            # Set padding tokens to -100 (ignored in loss)
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            labels = torch.tensor([-100] * self.max_length)
        
        return {
            "image": processed_image,
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels,
            "question": question,
            "answer": answer,
            "chart_type": chart_type,
            "reasoning_type": reasoning_type,
            "idx": idx
        }
    
    def _create_prompt(self, question: str, chart_type: str) -> str:
        """Create formatted prompt for the model"""
        prompt = f"Chart Type: {chart_type}\n"
        prompt += f"Question: {question}\n"
        prompt += "Answer:"
        return prompt
    
    def get_reasoning_types(self) -> List[str]:
        """Get all unique reasoning types in the dataset"""
        reasoning_types = set()
        for item in self.dataset:
            reasoning_types.add(item.get("reasoning_type", "unknown"))
        return list(reasoning_types)
    
    def get_chart_types(self) -> List[str]:
        """Get all unique chart types in the dataset"""
        chart_types = set()
        for item in self.dataset:
            chart_types.add(item.get("chart_type", "unknown"))
        return list(chart_types)
    
    def filter_by_reasoning_type(self, reasoning_type: str) -> "ChartMuseumDataset":
        """Create a filtered dataset with only specified reasoning type"""
        filtered_data = [
            item for item in self.dataset 
            if item.get("reasoning_type") == reasoning_type
        ]
        
        # Create new dataset instance with filtered data
        new_dataset = ChartMuseumDataset(
            tokenizer=self.tokenizer,
            split=self.split,
            max_length=self.max_length,
            image_size=self.image_size
        )
        new_dataset.dataset = filtered_data
        return new_dataset
    
    def filter_by_chart_type(self, chart_type: str) -> "ChartMuseumDataset":
        """Create a filtered dataset with only specified chart type"""
        filtered_data = [
            item for item in self.dataset 
            if item.get("chart_type") == chart_type
        ]
        
        # Create new dataset instance with filtered data
        new_dataset = ChartMuseumDataset(
            tokenizer=self.tokenizer,
            split=self.split,
            max_length=self.max_length,
            image_size=self.image_size
        )
        new_dataset.dataset = filtered_data
        return new_dataset


class ChartQADataset(Dataset):
    """
    Dataset loader for ChartQA benchmark
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        split: str = "test",
        max_length: int = 512,
        image_size: Tuple[int, int] = (224, 224),
        cache_dir: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.image_size = image_size
        
        # Load ChartQA dataset
        print(f"Loading ChartQA dataset (split: {split})...")
        self.dataset = load_dataset("HuggingFaceM4/ChartQA", split=split, cache_dir=cache_dir)
        print(f"Loaded {len(self.dataset)} examples")
        
        # Image preprocessing
        from torchvision import transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example from the dataset"""
        item = self.dataset[idx]
        
        # Extract components
        image = item.get("image")
        question = item.get("query", "")
        answer = item.get("answer", "")
        
        # Process image
        if image is not None:
            processed_image = self.image_transform(image)
        else:
            processed_image = torch.zeros(3, *self.image_size)
        
        # Process text
        prompt = f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Process answer for training
        if answer:
            answer_inputs = self.tokenizer(
                str(answer),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            labels = answer_inputs["input_ids"].squeeze()
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            labels = torch.tensor([-100] * self.max_length)
        
        return {
            "image": processed_image,
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels,
            "question": question,
            "answer": str(answer),
            "idx": idx
        }


class PlotQADataset(Dataset):
    """
    Dataset loader for PlotQA benchmark
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        split: str = "test",
        max_length: int = 512,
        image_size: Tuple[int, int] = (224, 224),
        data_dir: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.image_size = image_size
        
        # Load PlotQA dataset (assuming local files)
        if data_dir is None:
            raise ValueError("data_dir must be provided for PlotQA dataset")
        
        self.data_dir = data_dir
        self.annotations_file = os.path.join(data_dir, f"{split}.json")
        self.images_dir = os.path.join(data_dir, "images")
        
        # Load annotations
        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        print(f"Loaded {len(self.annotations)} PlotQA examples")
        
        # Image preprocessing
        from torchvision import transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example from the dataset"""
        item = self.annotations[idx]
        
        # Extract components
        image_file = item.get("image_file", "")
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        # Load and process image
        image_path = os.path.join(self.images_dir, image_file)
        try:
            image = Image.open(image_path).convert("RGB")
            processed_image = self.image_transform(image)
        except:
            processed_image = torch.zeros(3, *self.image_size)
        
        # Process text
        prompt = f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Process answer for training
        if answer:
            answer_inputs = self.tokenizer(
                str(answer),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            labels = answer_inputs["input_ids"].squeeze()
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            labels = torch.tensor([-100] * self.max_length)
        
        return {
            "image": processed_image,
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels,
            "question": question,
            "answer": str(answer),
            "image_file": image_file,
            "idx": idx
        }


class MultiDatasetLoader:
    """
    Utility class for loading and managing multiple chart datasets
    """
    
    def __init__(self, tokenizer: AutoTokenizer, config: Dict[str, Any]):
        self.tokenizer = tokenizer
        self.config = config
        self.datasets = {}
    
    def load_chartmuseum(self, split: str = "test") -> ChartMuseumDataset:
        """Load ChartMuseum dataset"""
        if "chartmuseum" not in self.datasets:
            self.datasets["chartmuseum"] = {}
        
        if split not in self.datasets["chartmuseum"]:
            self.datasets["chartmuseum"][split] = ChartMuseumDataset(
                tokenizer=self.tokenizer,
                split=split,
                max_length=self.config.get("max_length", 512),
                image_size=self.config.get("image_size", (224, 224))
            )
        
        return self.datasets["chartmuseum"][split]
    
    def load_chartqa(self, split: str = "test") -> ChartQADataset:
        """Load ChartQA dataset"""
        if "chartqa" not in self.datasets:
            self.datasets["chartqa"] = {}
        
        if split not in self.datasets["chartqa"]:
            self.datasets["chartqa"][split] = ChartQADataset(
                tokenizer=self.tokenizer,
                split=split,
                max_length=self.config.get("max_length", 512),
                image_size=self.config.get("image_size", (224, 224))
            )
        
        return self.datasets["chartqa"][split]
    
    def load_plotqa(self, split: str = "test", data_dir: str = None) -> PlotQADataset:
        """Load PlotQA dataset"""
        if "plotqa" not in self.datasets:
            self.datasets["plotqa"] = {}
        
        if split not in self.datasets["plotqa"]:
            self.datasets["plotqa"][split] = PlotQADataset(
                tokenizer=self.tokenizer,
                split=split,
                max_length=self.config.get("max_length", 512),
                image_size=self.config.get("image_size", (224, 224)),
                data_dir=data_dir or self.config.get("plotqa_data_dir")
            )
        
        return self.datasets["plotqa"][split]
    
    def get_combined_dataset(self, datasets: List[str], split: str = "test") -> Dataset:
        """Combine multiple datasets into one"""
        combined_data = []
        
        for dataset_name in datasets:
            if dataset_name == "chartmuseum":
                dataset = self.load_chartmuseum(split)
            elif dataset_name == "chartqa":
                dataset = self.load_chartqa(split)
            elif dataset_name == "plotqa":
                dataset = self.load_plotqa(split)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            # Add dataset source to each item
            for i in range(len(dataset)):
                item = dataset[i]
                item["dataset_source"] = dataset_name
                combined_data.append(item)
        
        return CombinedDataset(combined_data)


class CombinedDataset(Dataset):
    """
    Dataset that combines multiple chart datasets
    """
    
    def __init__(self, combined_data: List[Dict[str, Any]]):
        self.data = combined_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx] 