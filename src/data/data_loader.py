"""
Data loader for ChartExpert-MoE
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional


class ChartDataLoader:
    """Custom data loader for chart datasets"""
    
    def __init__(
        self,
        dataset,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn
        )
        
    def collate_fn(self, batch):
        """Custom collate function for chart data"""
        # Handle different data formats
        if isinstance(batch[0], dict):
            return self._collate_dict_batch(batch)
        else:
            return torch.utils.data.default_collate(batch)
            
    def _collate_dict_batch(self, batch):
        """Collate dictionary batch"""
        collated = {}
        
        for key in batch[0].keys():
            if key in ['image', 'input_ids', 'attention_mask', 'labels']:
                # Stack tensors
                collated[key] = torch.stack([item[key] for item in batch])
            else:
                # Keep as list for other fields
                collated[key] = [item[key] for item in batch]
                
        return collated
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader) 