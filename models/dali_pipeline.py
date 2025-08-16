"""
NVIDIA DALI data pipeline for GPU-accelerated data loading
"""
import logging
from typing import Optional, Any

from config import config
from utils.imports import gpu_imports

logger = logging.getLogger(__name__)

class DALIPipeline:
    """NVIDIA DALI data pipeline for medical imaging"""
    
    def __init__(self):
        self.batch_size = config.batch_size
        self.device_id = config.device_id
        self.data_dir = config.data_dir
        
    def create_pipeline(self) -> Optional[Any]:
        """Create DALI pipeline for data loading"""
        try:
            if not gpu_imports.dali_available:
                logger.warning("⚠ NVIDIA DALI not available")
                return None
            
            logger.info("Creating DALI pipeline...")
            
            @gpu_imports.dali_pipeline_def
            def dali_pipeline(file_root):
                jpegs, labels = gpu_imports.dali_fn.readers.file(
                    file_root=file_root, 
                    random_shuffle=True, 
                    name="Reader"
                )
                images = gpu_imports.dali_fn.decoders.image(jpegs, device="mixed")
                images = gpu_imports.dali_fn.resize(images, resize_x=config.image_size, resize_y=config.image_size)
                images = gpu_imports.dali_fn.flip(images, horizontal=1)
                images = gpu_imports.dali_fn.rotate(images, angle=gpu_imports.dali_fn.random.uniform(range=(-10.0, 10.0)))
                images = gpu_imports.dali_fn.brightness_contrast(images, brightness=gpu_imports.dali_fn.random.uniform(0.9, 1.1))
                images = gpu_imports.dali_fn.crop_mirror_normalize(
                    images,
                    dtype=gpu_imports.dali_types.FLOAT,
                    output_layout="CHW",
                    mean=[0.5 * 255],
                    std=[0.25 * 255]
                )
                return images, labels
            
            pipe = dali_pipeline(
                file_root=f"{self.data_dir}/MedNIST", 
                batch_size=self.batch_size, 
                num_threads=4, 
                device_id=self.device_id
            )
            pipe.build()
            
            logger.info("✓ DALI pipeline created successfully")
            return pipe
            
        except Exception as e:
            logger.error(f"✗ DALI pipeline creation failed: {e}")
            return None
    
    def create_iterator(self, pipe: Any) -> Optional[Any]:
        """Create DALI iterator for PyTorch integration"""
        try:
            if pipe is None:
                logger.warning("⚠ No DALI pipeline available")
                return None
            
            logger.info("Creating DALI iterator...")
            
            dali_iterator = gpu_imports.dali_iterator(
                [pipe],
                output_map=["data", "label"],
                reader_name="Reader",
                auto_reset=True
            )
            
            logger.info("✓ DALI iterator created successfully")
            return dali_iterator
            
        except Exception as e:
            logger.error(f"✗ DALI iterator creation failed: {e}")
            return None
    
    def get_data_loader(self) -> Optional[Any]:
        """Get complete DALI data loader"""
        try:
            pipe = self.create_pipeline()
            if pipe is None:
                return None
            
            iterator = self.create_iterator(pipe)
            return iterator
            
        except Exception as e:
            logger.error(f"✗ DALI data loader creation failed: {e}")
            return None 