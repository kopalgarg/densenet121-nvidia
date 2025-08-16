"""
Centralized imports with error handling and fallbacks
"""
import logging
import warnings
from typing import Optional, Union
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU-accelerated libraries with fallbacks
class GPUImports:
    """Manages GPU library imports with fallbacks"""
    
    def __init__(self):
        self.cucim_available = False
        self.cupy_available = False
        self.cudf_available = False
        self.cuml_available = False
        self.cugraph_available = False
        self.monai_available = False
        self.dali_available = False
        self.wandb_available = False
        
        self._import_libraries()
    
    def _import_libraries(self):
        """Import all libraries with error handling"""
        
        # Import cucim
        try:
            import cucim
            self.cucim = cucim
            self.cucim_available = True
            logger.info("✓ cucim imported successfully")
        except ImportError as e:
            logger.warning(f"⚠ cucim not available: {e}")
            self.cucim = None
        
        # Import cupy
        try:
            import cupy as cp
            self.cupy = cp
            self.cupy_available = True
            logger.info("✓ cupy imported successfully")
        except ImportError as e:
            logger.warning(f"⚠ cupy not available: {e}")
            self.cupy = None
        
        # Import cudf with CUDA version handling
        try:
            import cudf
            self.cudf = cudf
            self.cudf_available = True
            logger.info("✓ cudf imported successfully")
        except RuntimeError as e:
            logger.warning(f"⚠ cudf failed due to CUDA version mismatch: {e}")
            import pandas as pd
            self.cudf = pd
            logger.info("✓ Using pandas as cudf fallback")
        except ImportError as e:
            logger.warning(f"⚠ cudf not available: {e}")
            import pandas as pd
            self.cudf = pd
            logger.info("✓ Using pandas as cudf fallback")
        
        # Import RAPIDS libraries
        try:
            from cuml import PCA, UMAP
            from cuml.neighbors import NearestNeighbors
            self.cuml_pca = PCA
            self.cuml_umap = UMAP
            self.cuml_neighbors = NearestNeighbors
            self.cuml_available = True
            logger.info("✓ cuml imported successfully")
        except ImportError as e:
            logger.warning(f"⚠ cuml not available: {e}")
            from sklearn.decomposition import PCA
            from sklearn.neighbors import NearestNeighbors
            self.cuml_pca = PCA
            self.cuml_umap = None  # UMAP not in sklearn
            self.cuml_neighbors = NearestNeighbors
            logger.info("✓ Using scikit-learn as cuml fallback")
        
        try:
            import cugraph
            self.cugraph = cugraph
            self.cugraph_available = True
            logger.info("✓ cugraph imported successfully")
        except ImportError as e:
            logger.warning(f"⚠ cugraph not available: {e}")
            import networkx as nx
            self.cugraph = nx
            logger.info("✓ Using networkx as cugraph fallback")
        
        # Import MONAI
        try:
            from monai.transforms import (
                Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
                RandFlipd, RandRotate90d, RandZoomd, ToTensord
            )
            from monai.networks.nets import DenseNet121
            from monai.data import CacheDataset, DataLoader
            from monai.losses import DiceLoss
            from monai.metrics import DiceMetric
            from monai.apps import download_and_extract
            
            self.monai_transforms = {
                'Compose': Compose,
                'LoadImaged': LoadImaged,
                'EnsureChannelFirstd': EnsureChannelFirstd,
                'ScaleIntensityd': ScaleIntensityd,
                'RandFlipd': RandFlipd,
                'RandRotate90d': RandRotate90d,
                'RandZoomd': RandZoomd,
                'ToTensord': ToTensord
            }
            self.monai_networks = {'DenseNet121': DenseNet121}
            self.monai_data = {'CacheDataset': CacheDataset, 'DataLoader': DataLoader}
            self.monai_losses = {'DiceLoss': DiceLoss}
            self.monai_metrics = {'DiceMetric': DiceMetric}
            self.monai_apps = {'download_and_extract': download_and_extract}
            self.monai_available = True
            logger.info("✓ MONAI imported successfully")
        except ImportError as e:
            logger.warning(f"⚠ MONAI not available: {e}")
            self.monai_available = False
        
        # Import NVIDIA DALI
        try:
            from nvidia.dali.pipeline import pipeline_def
            import nvidia.dali.fn as fn
            import nvidia.dali.types as types
            from nvidia.dali.plugin.pytorch import DALIGenericIterator
            
            self.dali_pipeline_def = pipeline_def
            self.dali_fn = fn
            self.dali_types = types
            self.dali_iterator = DALIGenericIterator
            self.dali_available = True
            logger.info("✓ NVIDIA DALI imported successfully")
        except ImportError as e:
            logger.warning(f"⚠ NVIDIA DALI not available: {e}")
            self.dali_available = False
        
        # Import PyTorch
        try:
            import torch
            from torch.optim import Adam
            self.torch = torch
            self.Adam = Adam
            logger.info("✓ PyTorch imported successfully")
        except ImportError as e:
            logger.error(f"✗ PyTorch not available: {e}")
            raise ImportError("PyTorch is required but not available")
        
        # Import Weights & Biases
        try:
            import wandb
            self.wandb = wandb
            self.wandb_available = True
            logger.info("✓ Weights & Biases imported successfully")
        except ImportError as e:
            logger.warning(f"⚠ Weights & Biases not available: {e}")
            self.wandb_available = False
        
        # Import standard libraries
        import os
        import glob
        self.os = os
        self.glob = glob

# Global imports instance
gpu_imports = GPUImports() 