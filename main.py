"""
Main execution script for Medical Imaging Pipeline
"""
import logging
import os
import sys
from typing import Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from utils.imports import gpu_imports
from data.data_manager import DataManager
from analysis.eda import EDA
from models.dali_pipeline import DALIPipeline
from models.trainer import Trainer

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main execution function"""
    try:
        logger.info("🚀 Starting Medical Imaging Pipeline")
        
        # Initialize Weights & Biases if available
        if gpu_imports.wandb_available and config.wandb_enabled:
            try:
                gpu_imports.wandb.init(project=config.wandb_project)
                logger.info("✓ Weights & Biases initialized")
            except Exception as e:
                logger.warning(f"⚠ WandB initialization failed: {e}")
        
        # Step 1: Data Management
        logger.info("📊 Step 1: Data Management")
        data_manager = DataManager()
        
        # Download dataset
        if not data_manager.download_dataset():
            logger.error("Failed to download dataset")
            return False
        
        # Prepare dataframe
        df, train_df, val_df = data_manager.prepare_dataframe()
        logger.info("✓ Data preparation completed")
        
        # Step 2: Exploratory Data Analysis
        logger.info("🔍 Step 2: Exploratory Data Analysis")
        eda = EDA()
        
        # Run EDA on subset of data
        eda_results = eda.run_full_eda(train_df, subset_size=100)
        if eda_results:
            logger.info(f"✓ EDA completed: {eda_results}")
        else:
            logger.warning("⚠ EDA failed, continuing...")
        
        # Step 3: Data Pipeline
        logger.info("🔄 Step 3: Data Pipeline")
        dali_pipeline = DALIPipeline()
        
        # Create data loaders
        train_loader = dali_pipeline.get_data_loader()
        if train_loader is None:
            logger.error("Failed to create data loader")
            return False
        
        logger.info("✓ Data pipeline created")
        
        # Step 4: Model Training
        logger.info("🎯 Step 4: Model Training")
        trainer = Trainer()
        
        # Train model
        history = trainer.train(train_loader)
        if history:
            logger.info(f"✓ Training completed: {history}")
        else:
            logger.error("Training failed")
            return False
        
        # Step 5: Save Model
        logger.info("💾 Step 5: Save Model")
        model_path = "models/densenet121_mednist.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if trainer.save_model(model_path):
            logger.info("✓ Model saved successfully")
        else:
            logger.error("Failed to save model")
            return False
        
        logger.info("🎉 Pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Pipeline failed: {e}")
        return False

def run_eda_only():
    """Run only the EDA pipeline"""
    try:
        logger.info("🔍 Running EDA Pipeline Only")
        
        # Data Management
        data_manager = DataManager()
        if not data_manager.download_dataset():
            return False
        
        df, train_df, val_df = data_manager.prepare_dataframe()
        
        # EDA
        eda = EDA()
        results = eda.run_full_eda(train_df, subset_size=200)
        
        if results:
            logger.info(f"✓ EDA Results: {results}")
            return True
        else:
            logger.error("EDA failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ EDA failed: {e}")
        return False

def run_training_only():
    """Run only the training pipeline"""
    try:
        logger.info("🎯 Running Training Pipeline Only")
        
        # Data Management
        data_manager = DataManager()
        if not data_manager.download_dataset():
            return False
        
        # Data Pipeline
        dali_pipeline = DALIPipeline()
        train_loader = dali_pipeline.get_data_loader()
        if train_loader is None:
            return False
        
        # Training
        trainer = Trainer()
        history = trainer.train(train_loader)
        
        if history:
            logger.info(f"✓ Training completed: {history}")
            return True
        else:
            logger.error("Training failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical Imaging Pipeline")
    parser.add_argument("--mode", choices=["full", "eda", "training"], 
                       default="full", help="Pipeline mode")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        success = main()
    elif args.mode == "eda":
        success = run_eda_only()
    elif args.mode == "training":
        success = run_training_only()
    else:
        logger.error("Invalid mode")
        success = False
    
    sys.exit(0 if success else 1) 