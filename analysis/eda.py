"""
Exploratory Data Analysis module with GPU acceleration
"""
import logging
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

from config import config
from utils.imports import gpu_imports

logger = logging.getLogger(__name__)

class EDA:
    """Exploratory Data Analysis with GPU acceleration"""
    
    def __init__(self):
        self.pca_components = config.pca_components
        self.umap_components = config.umap_components
        self.knn_neighbors = config.knn_neighbors
        
    def load_images_gpu(self, paths: List[str], subset_size: Optional[int] = None) -> np.ndarray:
        """Load images using GPU-accelerated cuCIM"""
        try:
            if not gpu_imports.cucim_available:
                raise ImportError("cuCIM not available for GPU image loading")
            
            if subset_size:
                paths = paths[:subset_size]
            
            logger.info(f"Loading {len(paths)} images with cuCIM...")
            
            # Load images with cuCIM
            def load_image_gpu(path):
                img = gpu_imports.cucim.CuImage(path)
                return gpu_imports.cupy.asarray(img.read_region((0, 0), (config.image_size, config.image_size)))
            
            features = gpu_imports.cupy.stack([load_image_gpu(p) for p in paths])
            features = features.reshape(features.shape[0], -1)
            
            logger.info(f"✓ Loaded {features.shape[0]} images with {features.shape[1]} features each")
            return features
            
        except Exception as e:
            logger.error(f"✗ GPU image loading failed: {e}")
            raise
    
    def run_pca(self, features: np.ndarray) -> np.ndarray:
        """Run PCA dimensionality reduction"""
        try:
            logger.info("Running PCA...")
            
            if gpu_imports.cuml_available:
                # Use GPU-accelerated PCA
                pca = gpu_imports.cuml_pca(n_components=self.pca_components)
                pca_features = pca.fit_transform(features)
            else:
                # Use CPU PCA with conversion
                if gpu_imports.cupy_available:
                    features_cpu = gpu_imports.cupy.asnumpy(features)
                else:
                    features_cpu = features
                
                pca = gpu_imports.cuml_pca(n_components=self.pca_components)
                pca_features = pca.fit_transform(features_cpu)
            
            logger.info(f"✓ PCA features shape: {pca_features.shape}")
            return pca_features
            
        except Exception as e:
            logger.error(f"✗ PCA failed: {e}")
            raise
    
    def run_umap(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Run UMAP dimensionality reduction if available"""
        try:
            if gpu_imports.cuml_available and gpu_imports.cuml_umap is not None:
                logger.info("Running UMAP...")
                umap = gpu_imports.cuml_umap(n_components=self.umap_components)
                embedding = umap.fit_transform(features)
                logger.info(f"✓ UMAP embedding shape: {embedding.shape}")
                return embedding
            else:
                logger.warning("⚠ UMAP not available, skipping...")
                return None
                
        except Exception as e:
            logger.error(f"✗ UMAP failed: {e}")
            return None
    
    def build_knn_graph(self, features: np.ndarray) -> Tuple[List[int], List[int]]:
        """Build k-NN graph from features"""
        try:
            logger.info("Building k-NN graph...")
            
            if gpu_imports.cuml_available:
                # Use GPU-accelerated k-NN
                nn = gpu_imports.cuml_neighbors(n_neighbors=self.knn_neighbors)
                nn.fit(features)
                distances, indices = nn.kneighbors(features)
            else:
                # Use CPU k-NN
                if gpu_imports.cupy_available:
                    features_cpu = gpu_imports.cupy.asnumpy(features)
                else:
                    features_cpu = features
                
                nn = gpu_imports.cuml_neighbors(n_neighbors=self.knn_neighbors)
                nn.fit(features_cpu)
                distances, indices = nn.kneighbors(features_cpu)
            
            # Build edges
            src = []
            dst = []
            for i in range(indices.shape[0]):
                for j in range(1, self.knn_neighbors):  # Skip self-connection
                    src.append(i)
                    dst.append(int(indices[i, j]))
            
            logger.info(f"✓ Created {len(src)} edges")
            return src, dst
            
        except Exception as e:
            logger.error(f"✗ k-NN graph construction failed: {e}")
            raise
    
    def analyze_graph(self, src: List[int], dst: List[int]) -> Dict[str, Any]:
        """Analyze the constructed graph"""
        try:
            logger.info("Analyzing graph...")
            
            if gpu_imports.cugraph_available:
                # Use GPU-accelerated graph analysis
                edges_df = gpu_imports.cudf.DataFrame({"src": src, "dst": dst})
                G = gpu_imports.cugraph.Graph()
                G.from_cudf_edgelist(edges_df, source="src", destination="dst")
                
                # Louvain community detection
                parts, score = gpu_imports.cugraph.louvain(G)
                
                analysis = {
                    'num_vertices': G.number_of_vertices(),
                    'num_edges': G.number_of_edges(),
                    'communities': len(parts),
                    'modularity_score': score,
                    'graph_type': 'GPU (cuGraph)'
                }
                
            else:
                # Use CPU graph analysis
                G = gpu_imports.cugraph.Graph()  # This is networkx
                for s, d in zip(src, dst):
                    G.add_edge(s, d)
                
                # Basic graph metrics
                components = list(gpu_imports.cugraph.connected_components(G))
                
                analysis = {
                    'num_vertices': G.number_of_nodes(),
                    'num_edges': G.number_of_edges(),
                    'num_components': len(components),
                    'largest_component': len(max(components, key=len)) if components else 0,
                    'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
                    'graph_type': 'CPU (NetworkX)'
                }
            
            logger.info(f"✓ Graph analysis completed: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"✗ Graph analysis failed: {e}")
            raise
    
    def run_full_eda(self, train_df: object, subset_size: int = 100) -> Dict[str, Any]:
        """Run complete EDA pipeline"""
        try:
            logger.info("Starting EDA pipeline...")
            
            # Get image paths
            from data.data_manager import DataManager
            dm = DataManager()
            paths = dm.get_image_paths(train_df, subset_size)
            
            # Load images
            features = self.load_images_gpu(paths, subset_size)
            
            # Run PCA
            pca_features = self.run_pca(features)
            
            # Run UMAP (if available)
            umap_embedding = self.run_umap(pca_features)
            
            # Build k-NN graph
            src, dst = self.build_knn_graph(pca_features)
            
            # Analyze graph
            graph_analysis = self.analyze_graph(src, dst)
            
            # Compile results
            results = {
                'num_images': len(paths),
                'feature_dim': features.shape[1],
                'pca_components': pca_features.shape[1],
                'umap_available': umap_embedding is not None,
                'graph_analysis': graph_analysis
            }
            
            logger.info("✓ EDA pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"✗ EDA pipeline failed: {e}")
            raise 