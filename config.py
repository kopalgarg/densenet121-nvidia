#!/usr/bin/env python3
"""
⚙️ Configuration Settings for Medical Imaging Pipeline Analysis
"""

import os

# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================

# Output directories
RESULTS_DIR = "comprehensive_gpu_cpu_analysis_results"
GPU_RESULTS_DIR = os.path.join(RESULTS_DIR, "gpu_results")
CPU_RESULTS_DIR = os.path.join(RESULTS_DIR, "cpu_results")
COMPARISON_RESULTS_DIR = os.path.join(RESULTS_DIR, "comparison_results")

# Analysis parameters
ANALYSIS_DURATION = 100  # seconds for time-series analysis
SAMPLING_RATE = 0.1      # seconds between samples
BATCH_SIZE = 32          # batch size for processing
IMAGE_SIZE = (256, 256)  # standard image dimensions

# GPU settings
GPU_DEVICE = "Tesla T4 GPU"
GPU_MEMORY = 15360       # MB
GPU_CORES = 2560
GPU_TENSOR_CORES = 320

# CPU settings
CPU_CORES = 8
CPU_THREADS = 16
CPU_MEMORY = 16          # GB

# Visualization settings
FIGURE_DPI = 300
FIGURE_SIZE = (16, 12)
COLOR_MAP = 'viridis'
STYLE = 'seaborn'

# Medical imaging settings
MEDICAL_MODALITIES = ['CT', 'MRI', 'X-Ray', 'Ultrasound']
PLANAR_VIEWS = ['Axial', 'Coronal', 'Sagittal']
AUGMENTATION_TYPES = [
    'Rotation', 'Flip', 'Noise', 'Brightness', 
    'Contrast', 'Blur', 'Zoom', 'Elastic'
]

# Performance thresholds
GPU_UTILIZATION_THRESHOLD = 0.8
MEMORY_EFFICIENCY_THRESHOLD = 0.7
POWER_EFFICIENCY_THRESHOLD = 0.6

# =============================================================================
# ENVIRONMENT SETTINGS
# =============================================================================

# Force CPU usage if needed
FORCE_CPU = os.environ.get('FORCE_CPU', 'false').lower() == 'true'

# GPU visibility
CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

# Output verbosity
VERBOSE = os.environ.get('VERBOSE', 'true').lower() == 'true'

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate configuration settings"""
    global FORCE_CPU
    
    # Create output directories
    for directory in [RESULTS_DIR, GPU_RESULTS_DIR, CPU_RESULTS_DIR, COMPARISON_RESULTS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Validate GPU settings
    if not FORCE_CPU:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ GPU detected: {torch.cuda.get_device_name()}")
            else:
                print("⚠️  GPU not available, falling back to CPU")
                FORCE_CPU = True
        except ImportError:
            print("⚠️  PyTorch not available, falling back to CPU")
            FORCE_CPU = True
    
    print(f"✅ Configuration validated. Force CPU: {FORCE_CPU}")

if __name__ == "__main__":
    validate_config() 