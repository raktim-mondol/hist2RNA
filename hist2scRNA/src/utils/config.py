"""
Configuration utilities for hist2scRNA

Handles loading and managing configuration files.
"""

import yaml
import json
from typing import Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Model configuration"""
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 384
    vit_depth: int = 6
    vit_heads: int = 6
    gnn_hidden: int = 512
    gnn_heads: int = 4
    n_genes: int = 2000
    n_cell_types: int = 10
    use_spatial_graph: bool = True
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 50
    batch_size: int = 8
    lr: float = 0.0001
    weight_decay: float = 0.01
    alpha: float = 0.1  # Weight for cell type loss
    seed: int = 42


@dataclass
class DataConfig:
    """Data configuration"""
    data_dir: str = './dummy_data/small'
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    use_images: bool = True
    augment: bool = False


@dataclass
class Config:
    """Complete configuration"""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    output_dir: str = './output_scrna'
    checkpoint_dir: str = './checkpoints'
    device: str = 'cuda'

    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            output_dir=config_dict.get('output_dir', './output_scrna'),
            checkpoint_dir=config_dict.get('checkpoint_dir', './checkpoints'),
            device=config_dict.get('device', 'cuda')
        )

    @classmethod
    def from_json(cls, path: str):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)

        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            output_dir=config_dict.get('output_dir', './output_scrna'),
            checkpoint_dir=config_dict.get('checkpoint_dir', './checkpoints'),
            device=config_dict.get('device', 'cuda')
        )

    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'device': self.device
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def to_json(self, path: str):
        """Save configuration to JSON file"""
        config_dict = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'device': self.device
        }

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)


def get_default_config() -> Config:
    """Get default configuration"""
    return Config(
        model=ModelConfig(),
        training=TrainingConfig(),
        data=DataConfig()
    )
