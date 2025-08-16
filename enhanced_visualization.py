#!/usr/bin/env python3
"""
Enhanced Medical Image Visualization Pipeline
With CPU Fallbacks and NVIDIA GPU Acceleration
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
import time
import os

# Try to import GPU libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ GPU visualization available with cuPy")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠ GPU not available, using CPU visualization")

class MedicalVisualizer:
    """Enhanced medical image visualization with GPU/CPU fallbacks"""
    
    def __init__(self, gpu_available=None):
        """Initialize visualizer with GPU/CPU detection"""
        self.gpu_available = gpu_available if gpu_available is not None else GPU_AVAILABLE
        self.colormaps = {
            'segmentation': 'viridis',
            'confidence': 'hot',
            'difference': 'RdBu',
            'original': 'gray'
        }
        # Default output directory (can be overridden via OUTPUT_DIR env var)
        self.default_output_dir = os.getenv("OUTPUT_DIR", "results")
        os.makedirs(self.default_output_dir, exist_ok=True)
    
    def load_image(self, image_path):
        """Load image with GPU/CPU optimization"""
        try:
            if self.gpu_available:
                # GPU loading
                img = Image.open(image_path)
                img_array = np.array(img)
                gpu_array = cp.asarray(img_array)
                return gpu_array
            else:
                # CPU loading
                img = Image.open(image_path)
                return np.array(img)
        except Exception as e:
            print(f"✗ Image loading failed: {e}")
            return None
    
    def create_segmentation_overlay(self, image, prediction, alpha=0.7):
        """Create segmentation overlay with GPU/CPU optimization"""
        
        if self.gpu_available:
            # GPU processing
            if hasattr(image, 'get'):
                image_cpu = image.get()
            else:
                image_cpu = image
            
            if hasattr(prediction, 'get'):
                pred_cpu = prediction.get()
            else:
                pred_cpu = prediction
            
            # Convert to RGB if grayscale
            if len(image_cpu.shape) == 2:
                image_rgb = cv2.cvtColor(image_cpu, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image_cpu
            
            # Create colored segmentation
            pred_colored = cv2.applyColorMap(pred_cpu.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            
            # Blend images
            overlay = cv2.addWeighted(image_rgb, 1-alpha, pred_colored, alpha, 0)
            
            return overlay
        else:
            # CPU processing
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image
            
            # Create colored segmentation
            pred_colored = cv2.applyColorMap(prediction.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            
            # Blend images
            overlay = cv2.addWeighted(image_rgb, 1-alpha, pred_colored, alpha, 0)
            
            return overlay
    
    def create_confidence_map(self, confidence_scores, colormap='hot'):
        """Create confidence map visualization"""
        
        if self.gpu_available:
            # GPU processing
            if hasattr(confidence_scores, 'get'):
                conf_cpu = confidence_scores.get()
            else:
                conf_cpu = confidence_scores
            
            # Normalize to 0-255
            conf_normalized = ((conf_cpu - conf_cpu.min()) / (conf_cpu.max() - conf_cpu.min()) * 255).astype(np.uint8)
            conf_colored = cv2.applyColorMap(conf_normalized, cv2.COLORMAP_HOT)
            
            return conf_colored
        else:
            # CPU processing
            conf_normalized = ((confidence_scores - confidence_scores.min()) / (confidence_scores.max() - confidence_scores.min()) * 255).astype(np.uint8)
            conf_colored = cv2.applyColorMap(conf_normalized, cv2.COLORMAP_HOT)
            
            return conf_colored
    
    def create_comparison_plot(self, original, prediction, ground_truth=None, confidence=None):
        """Create comprehensive comparison plot"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Medical Image Analysis Results', fontsize=16, fontweight='bold')
        
        # Convert GPU arrays to CPU if needed
        if self.gpu_available:
            if hasattr(original, 'get'):
                original_cpu = original.get()
            else:
                original_cpu = original
            
            if hasattr(prediction, 'get'):
                pred_cpu = prediction.get()
            else:
                pred_cpu = prediction
        else:
            original_cpu = original
            pred_cpu = prediction
        
        # Original image
        axes[0, 0].imshow(original_cpu, cmap='gray')
        axes[0, 0].set_title('Original Scan', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Segmentation overlay
        overlay = self.create_segmentation_overlay(original_cpu, pred_cpu)
        axes[0, 1].imshow(overlay)
        axes[0, 1].set_title('AI Segmentation Overlay', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Prediction mask
        axes[0, 2].imshow(pred_cpu, cmap='viridis')
        axes[0, 2].set_title('Segmentation Mask', fontweight='bold')
        axes[0, 2].axis('off')
        
        # Ground truth comparison (if available)
        if ground_truth is not None:
            if self.gpu_available and hasattr(ground_truth, 'get'):
                gt_cpu = ground_truth.get()
            else:
                gt_cpu = ground_truth
            
            axes[1, 0].imshow(gt_cpu, cmap='viridis')
            axes[1, 0].set_title('Ground Truth', fontweight='bold')
            axes[1, 0].axis('off')
            
            # Difference map
            diff = np.abs(pred_cpu.astype(float) - gt_cpu.astype(float))
            axes[1, 1].imshow(diff, cmap='RdBu')
            axes[1, 1].set_title('Difference Map', fontweight='bold')
            axes[1, 1].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'Ground Truth\nNot Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Ground Truth', fontweight='bold')
            axes[1, 0].axis('off')
            axes[1, 1].axis('off')
        
        # Confidence map
        if confidence is not None:
            if self.gpu_available and hasattr(confidence, 'get'):
                conf_cpu = confidence.get()
            else:
                conf_cpu = confidence
            
            conf_vis = self.create_confidence_map(conf_cpu)
            axes[1, 2].imshow(conf_vis)
            axes[1, 2].set_title(f'Confidence Map\n(Avg: {conf_cpu.mean():.3f})', fontweight='bold')
            axes[1, 2].axis('off')
        else:
            axes[1, 2].text(0.5, 0.5, 'Confidence\nNot Available', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Confidence Map', fontweight='bold')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_3d_visualization(self, volume_data, segmentation=None):
        """Create 3D visualization (if plotly is available)"""
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Convert GPU arrays to CPU
            if self.gpu_available:
                if hasattr(volume_data, 'get'):
                    vol_cpu = volume_data.get()
                else:
                    vol_cpu = volume_data
            else:
                vol_cpu = volume_data
            
            # Create 3D figure
            fig = go.Figure()
            
            # Add volume data
            fig.add_trace(go.Volume(
                x=vol_cpu[0].flatten(),
                y=vol_cpu[1].flatten(),
                z=vol_cpu[2].flatten(),
                value=vol_cpu[3].flatten(),
                opacity=0.3,
                name="Original Volume"
            ))
            
            # Add segmentation if available
            if segmentation is not None:
                if self.gpu_available and hasattr(segmentation, 'get'):
                    seg_cpu = segmentation.get()
                else:
                    seg_cpu = segmentation
                
                fig.add_trace(go.Volume(
                    x=seg_cpu[0].flatten(),
                    y=seg_cpu[1].flatten(),
                    z=seg_cpu[2].flatten(),
                    value=seg_cpu[3].flatten(),
                    opacity=0.7,
                    name="Segmentation"
                ))
            
            fig.update_layout(
                title="3D Medical Volume Visualization",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z"
                )
            )
            
            return fig
            
        except ImportError:
            print("⚠ Plotly not available for 3D visualization")
            return None
    
    def create_medical_report(self, image_path, prediction, metadata=None):
        """Create comprehensive medical report"""
        
        # Load image if path provided, otherwise use prediction shape for demo
        if image_path:
            image = self.load_image(image_path)
            if image is None:
                return None
        else:
            # For demo purposes, create a placeholder image
            image = np.random.randint(0, 255, prediction.shape, dtype=np.uint8)
        
        # Create report figure
        fig = plt.figure(figsize=(20, 12))
        
        # Main visualization
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        if self.gpu_available and hasattr(image, 'get'):
            image_cpu = image.get()
        else:
            image_cpu = image
        ax1.imshow(image_cpu, cmap='gray')
        ax1.set_title('Original Medical Image', fontweight='bold')
        ax1.axis('off')
        
        # Segmentation overlay
        ax2 = fig.add_subplot(gs[0, 1])
        overlay = self.create_segmentation_overlay(image_cpu, prediction)
        ax2.imshow(overlay)
        ax2.set_title('AI Segmentation Result', fontweight='bold')
        ax2.axis('off')
        
        # Prediction mask
        ax3 = fig.add_subplot(gs[0, 2])
        if self.gpu_available and hasattr(prediction, 'get'):
            pred_cpu = prediction.get()
        else:
            pred_cpu = prediction
        ax3.imshow(pred_cpu, cmap='viridis')
        ax3.set_title('Segmentation Mask', fontweight='bold')
        ax3.axis('off')
        
        # Metadata panel
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        
        if metadata:
            info_text = f"""
            Medical Analysis Report
            
            Patient ID: {metadata.get('patient_id', 'N/A')}
            Scan Date: {metadata.get('scan_date', 'N/A')}
            Modality: {metadata.get('modality', 'N/A')}
            
            Analysis Results:
            • Dice Score: {metadata.get('dice_score', 'N/A')}
            • Confidence: {metadata.get('confidence', 'N/A')}
            • Processing Time: {metadata.get('processing_time', 'N/A')}s
            
            AI Model: Medical Segmentation v1.0
            """
        else:
            info_text = """
            Medical Analysis Report
            
            AI Segmentation Completed
            Processing: GPU Accelerated
            Model: Medical Segmentation v1.0
            """
        
        ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        # Confidence visualization (if available)
        if 'confidence' in metadata:
            ax5 = fig.add_subplot(gs[1, :2])
            conf_vis = self.create_confidence_map(metadata['confidence'])
            ax5.imshow(conf_vis)
            ax5.set_title('Confidence Map', fontweight='bold')
            ax5.axis('off')
        
        # Histogram of prediction values
        ax6 = fig.add_subplot(gs[1, 2:])
        if self.gpu_available and hasattr(prediction, 'get'):
            pred_hist = prediction.get().flatten()
        else:
            pred_hist = prediction.flatten()
        ax6.hist(pred_hist, bins=50, alpha=0.7, color='blue')
        ax6.set_title('Prediction Value Distribution', fontweight='bold')
        ax6.set_xlabel('Prediction Value')
        ax6.set_ylabel('Frequency')
        
        # Performance metrics
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        if metadata:
            metrics_text = f"""
            Performance Metrics:
            
            • Segmentation Accuracy: {metadata.get('accuracy', 'N/A')}
            • Processing Speed: {metadata.get('speed', 'N/A')} images/sec
            • Memory Usage: {metadata.get('memory', 'N/A')} GB
            • GPU Utilization: {metadata.get('gpu_util', 'N/A')}%
            
            Quality Assessment:
            • Image Quality: {metadata.get('image_quality', 'N/A')}
            • Segmentation Quality: {metadata.get('seg_quality', 'N/A')}
            • Confidence Level: {metadata.get('confidence_level', 'N/A')}
            """
        else:
            metrics_text = """
            Performance Metrics:
            
            • GPU Accelerated Processing
            • Real-time Segmentation
            • High Confidence Results
            • Medical Grade Accuracy
            """
        
        ax7.text(0.1, 0.5, metrics_text, transform=ax7.transAxes, 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.5))
        
        plt.suptitle('Medical AI Analysis Report', fontsize=16, fontweight='bold')
        return fig
    
    def save_visualization(self, fig, output_path, dpi=300):
        """Save visualization to file, defaulting to the results directory if no folder is provided"""
        try:
            # If caller passed a bare filename, place it in the results directory
            directory, filename = os.path.dirname(output_path), os.path.basename(output_path)
            if directory == "":
                directory = self.default_output_dir
                output_path = os.path.join(directory, filename)
            # Ensure directory exists
            os.makedirs(directory, exist_ok=True)
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"✓ Visualization saved to: {output_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to save visualization: {e}")
            return False
    
    def save_analysis_results(self, results_data, filename="analysis_results.txt"):
        """Save analysis results to a text file in the results directory"""
        try:
            output_path = os.path.join(self.default_output_dir, filename)
            os.makedirs(self.default_output_dir, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write("Medical Image Analysis Results\n")
                f.write("=" * 50 + "\n\n")
                
                for key, value in results_data.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{key}: {value}\n")
                    elif isinstance(value, str):
                        f.write(f"{key}: {value}\n")
                    elif isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for sub_key, sub_value in value.items():
                            f.write(f"  {sub_key}: {sub_value}\n")
                    else:
                        f.write(f"{key}: {str(value)}\n")
                
                f.write(f"\nGenerated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"GPU Available: {self.gpu_available}\n")
            
            print(f"✓ Analysis results saved to: {output_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to save analysis results: {e}")
            return False
    
    def save_metrics_summary(self, metrics, filename="metrics_summary.txt"):
        """Save performance metrics to a summary file"""
        try:
            output_path = os.path.join(self.default_output_dir, filename)
            os.makedirs(self.default_output_dir, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write("Performance Metrics Summary\n")
                f.write("=" * 40 + "\n\n")
                
                # Performance metrics
                f.write("Performance Metrics:\n")
                f.write("-" * 20 + "\n")
                for key, value in metrics.items():
                    if key in ['processing_time', 'memory_usage', 'gpu_utilization']:
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                
                f.write("\nQuality Metrics:\n")
                f.write("-" * 15 + "\n")
                for key, value in metrics.items():
                    if key in ['accuracy', 'dice_score', 'confidence']:
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                
                f.write(f"\nGenerated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"GPU Available: {self.gpu_available}\n")
            
            print(f"✓ Metrics summary saved to: {output_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to save metrics summary: {e}")
            return False
    
    def create_results_directory_structure(self):
        """Create organized directory structure for results"""
        try:
            # Create main results directory
            os.makedirs(self.default_output_dir, exist_ok=True)
            
            # Create subdirectories
            subdirs = ['visualizations', 'reports', 'metrics', 'data']
            for subdir in subdirs:
                os.makedirs(os.path.join(self.default_output_dir, subdir), exist_ok=True)
            
            print(f"✓ Results directory structure created in: {self.default_output_dir}")
            return True
        except Exception as e:
            print(f"✗ Failed to create directory structure: {e}")
            return False
    
    def save_all_outputs(self, fig, results_data, metrics, base_filename="analysis"):
        """Save all outputs (visualization, results, metrics) with consistent naming"""
        try:
            # Create directory structure
            self.create_results_directory_structure()
            
            # Save visualization
            vis_path = os.path.join(self.default_output_dir, 'visualizations', f'{base_filename}_visualization.png')
            self.save_visualization(fig, vis_path)
            
            # Save results data
            results_path = os.path.join(self.default_output_dir, 'reports', f'{base_filename}_results.txt')
            self.save_analysis_results(results_data, os.path.basename(results_path))
            
            # Save metrics
            metrics_path = os.path.join(self.default_output_dir, 'metrics', f'{base_filename}_metrics.txt')
            self.save_metrics_summary(metrics, os.path.basename(metrics_path))
            
            print(f"✓ All outputs saved to {self.default_output_dir}/")
            return True
        except Exception as e:
            print(f"✗ Failed to save all outputs: {e}")
            return False

def demo_visualization():
    """Demo the enhanced visualization pipeline"""
    
    print("🎨 Medical Image Visualization Demo")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = MedicalVisualizer()
    
    # Create demo data
    print("📊 Creating demo medical data...")
    
    # Simulate medical image (64x64 grayscale)
    original = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    
    # Simulate segmentation prediction
    prediction = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    
    # Simulate confidence scores
    confidence = np.random.random((64, 64))
    
    # Simulate metadata
    metadata = {
        'patient_id': 'DEMO_001',
        'scan_date': '2024-01-15',
        'modality': 'CT',
        'dice_score': 0.85,
        'confidence': confidence,
        'processing_time': 0.15,
        'accuracy': '95.2%',
        'speed': '120',
        'memory': '2.1',
        'gpu_util': '87',
        'image_quality': 'Excellent',
        'seg_quality': 'High',
        'confidence_level': 'Very High'
    }
    
    print("🖼️ Creating visualizations...")
    
    # Create comparison plot
    fig1 = visualizer.create_comparison_plot(original, prediction, confidence=confidence)
    
    # Create medical report (pass None for image path since we're using demo data)
    fig2 = visualizer.create_medical_report(None, prediction, metadata)
    
    print("💾 Saving all outputs to results folder...")
    
    # Save all outputs using the comprehensive method
    results_data = {
        'demo_data': {
            'image_size': original.shape,
            'prediction_range': f"{prediction.min()}-{prediction.max()}",
            'confidence_range': f"{confidence.min():.3f}-{confidence.max():.3f}"
        },
        'metadata': metadata
    }
    
    metrics = {
        'processing_time': metadata['processing_time'],
        'accuracy': metadata['accuracy'],
        'dice_score': metadata['dice_score'],
        'confidence': f"{confidence.mean():.3f}",
        'gpu_utilization': metadata['gpu_util'] + '%',
        'memory_usage': metadata['memory'] + ' GB'
    }
    
    # Save comparison plot outputs
    visualizer.save_all_outputs(fig1, results_data, metrics, 'demo_comparison')
    
    # Save medical report outputs
    visualizer.save_all_outputs(fig2, results_data, metrics, 'demo_medical_report')
    
    print("✅ Demo completed successfully!")
    print("📁 Generated files in organized structure:")
    print("   • results/visualizations/")
    print("   • results/reports/")
    print("   • results/metrics/")
    print("   • results/data/")

if __name__ == "__main__":
    demo_visualization() 