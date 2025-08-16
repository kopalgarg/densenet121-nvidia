"""
Training module with MONAI model and GPU acceleration
"""
import logging
from typing import Optional, Dict, Any, List
import numpy as np

from config import config
from utils.imports import gpu_imports

logger = logging.getLogger(__name__)

class Trainer:
    """Training module with MONAI model and GPU acceleration"""
    
    def __init__(self):
        self.device = gpu_imports.torch.device(config.device)
        self.num_epochs = config.num_epochs
        self.learning_rate = config.learning_rate
        self.num_classes = config.num_classes
        
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.metrics = {}
        
    def create_model(self) -> bool:
        """Create MONAI DenseNet121 model"""
        try:
            if not gpu_imports.monai_available:
                logger.error("MONAI not available for model creation")
                return False
            
            logger.info("Creating DenseNet121 model...")
            
            self.model = gpu_imports.monai_networks['DenseNet121'](
                spatial_dims=2, 
                in_channels=1, 
                out_channels=self.num_classes
            ).to(self.device)
            
            logger.info("✓ Model created successfully")
            return True
            
        except Exception as e:
            logger.error(f"✗ Model creation failed: {e}")
            return False
    
    def create_optimizer(self) -> bool:
        """Create optimizer"""
        try:
            if self.model is None:
                logger.error("Model not created yet")
                return False
            
            logger.info("Creating optimizer...")
            
            self.optimizer = gpu_imports.Adam(self.model.parameters(), lr=self.learning_rate)
            
            logger.info("✓ Optimizer created successfully")
            return True
            
        except Exception as e:
            logger.error(f"✗ Optimizer creation failed: {e}")
            return False
    
    def create_loss_function(self) -> bool:
        """Create loss function"""
        try:
            if not gpu_imports.monai_available:
                logger.error("MONAI not available for loss function")
                return False
            
            logger.info("Creating DiceLoss...")
            
            self.loss_fn = gpu_imports.monai_losses['DiceLoss'](
                to_onehot_y=True, 
                softmax=True
            )
            
            logger.info("✓ Loss function created successfully")
            return True
            
        except Exception as e:
            logger.error(f"✗ Loss function creation failed: {e}")
            return False
    
    def create_metrics(self) -> bool:
        """Create evaluation metrics"""
        try:
            if not gpu_imports.monai_available:
                logger.warning("MONAI not available for metrics")
                return False
            
            logger.info("Creating evaluation metrics...")
            
            self.metrics = {
                'dice': gpu_imports.monai_metrics['DiceMetric'](include_background=False)
            }
            
            logger.info("✓ Metrics created successfully")
            return True
            
        except Exception as e:
            logger.error(f"✗ Metrics creation failed: {e}")
            return False
    
    def train_epoch(self, data_loader: Any) -> Dict[str, float]:
        """Train for one epoch"""
        try:
            if self.model is None or self.optimizer is None or self.loss_fn is None:
                logger.error("Model, optimizer, or loss function not initialized")
                return {}
            
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            logger.info("Starting training epoch...")
            
            for batch in data_loader:
                try:
                    data = batch[0]["data"].to(self.device)
                    labels = batch[0]["label"].squeeze().long().to(self.device)
                    
                    self.optimizer.zero_grad()
                    out = self.model(data)
                    
                    # Convert labels to one-hot format
                    one_hot_labels = gpu_imports.torch.nn.functional.one_hot(
                        labels, self.num_classes
                    ).permute(0, 3, 1, 2).float()
                    
                    loss = self.loss_fn(out, one_hot_labels)
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.warning(f"⚠ Batch training failed: {e}")
                    continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"✓ Epoch completed. Average loss: {avg_loss:.4f}")
            
            return {'loss': avg_loss}
            
        except Exception as e:
            logger.error(f"✗ Training epoch failed: {e}")
            return {}
    
    def validate(self, data_loader: Any) -> Dict[str, float]:
        """Validate model"""
        try:
            if self.model is None or self.loss_fn is None:
                logger.error("Model or loss function not initialized")
                return {}
            
            self.model.eval()
            total_loss = 0.0
            num_batches = 0
            
            logger.info("Starting validation...")
            
            with gpu_imports.torch.no_grad():
                for batch in data_loader:
                    try:
                        data = batch[0]["data"].to(self.device)
                        labels = batch[0]["label"].squeeze().long().to(self.device)
                        
                        out = self.model(data)
                        
                        # Convert labels to one-hot format
                        one_hot_labels = gpu_imports.torch.nn.functional.one_hot(
                            labels, self.num_classes
                        ).permute(0, 3, 1, 2).float()
                        
                        loss = self.loss_fn(out, one_hot_labels)
                        total_loss += loss.item()
                        num_batches += 1
                        
                    except Exception as e:
                        logger.warning(f"⚠ Batch validation failed: {e}")
                        continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"✓ Validation completed. Average loss: {avg_loss:.4f}")
            
            return {'val_loss': avg_loss}
            
        except Exception as e:
            logger.error(f"✗ Validation failed: {e}")
            return {}
    
    def train(self, train_loader: Any, val_loader: Optional[Any] = None) -> Dict[str, List[float]]:
        """Complete training loop"""
        try:
            logger.info(f"Starting training for {self.num_epochs} epochs...")
            
            # Initialize components
            if not all([
                self.create_model(),
                self.create_optimizer(),
                self.create_loss_function(),
                self.create_metrics()
            ]):
                logger.error("Failed to initialize training components")
                return {}
            
            # Training history
            history = {
                'train_loss': [],
                'val_loss': []
            }
            
            # Training loop
            for epoch in range(self.num_epochs):
                logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
                
                # Train
                train_metrics = self.train_epoch(train_loader)
                if train_metrics:
                    history['train_loss'].append(train_metrics['loss'])
                
                # Validate
                if val_loader is not None:
                    val_metrics = self.validate(val_loader)
                    if val_metrics:
                        history['val_loss'].append(val_metrics['val_loss'])
                
                # Log to wandb if available
                if gpu_imports.wandb_available:
                    try:
                        gpu_imports.wandb.log({
                            'epoch': epoch,
                            **train_metrics,
                            **val_metrics
                        })
                    except Exception as e:
                        logger.warning(f"⚠ WandB logging failed: {e}")
            
            logger.info("✓ Training completed successfully")
            return history
            
        except Exception as e:
            logger.error(f"✗ Training failed: {e}")
            return {}
    
    def save_model(self, path: str) -> bool:
        """Save trained model"""
        try:
            if self.model is None:
                logger.error("No model to save")
                return False
            
            gpu_imports.torch.save(self.model.state_dict(), path)
            logger.info(f"✓ Model saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Model saving failed: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load trained model"""
        try:
            if self.model is None:
                logger.error("Model not created yet")
                return False
            
            self.model.load_state_dict(gpu_imports.torch.load(path, map_location=self.device))
            logger.info(f"✓ Model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Model loading failed: {e}")
            return False 