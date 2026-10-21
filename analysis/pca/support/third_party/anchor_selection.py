"""
Anchor Selection for MLIP Embeddings

Select representative anchor embeddings from flat atomic embeddings using
various clustering and sampling strategies.

Supports:
- DIRECT sampling (via maml package)
- K-means clustering
- Farthest point sampling
- Random sampling with stratification

Usage:
    from anchor_selection import AnchorSelector

    selector = AnchorSelector(method="direct", n_anchors=100)
    result = selector.fit_transform(flat_embeddings)

    anchors = result.anchor_embeddings
    indices = result.anchor_indices
"""

import pickle
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal

import numpy as np
from tqdm import tqdm


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AnchorResult:
    """Container for anchor selection results."""
    anchor_embeddings: np.ndarray  # (n_anchors, embed_dim)
    anchor_indices: np.ndarray  # (n_anchors,) indices into original flat array
    n_anchors: int
    n_total: int
    embed_dim: int
    method: str

    # Optional: cluster assignments for all points
    cluster_labels: Optional[np.ndarray] = None  # (n_total,)
    cluster_centers: Optional[np.ndarray] = None  # (n_clusters, embed_dim)

    # Optional: PCA features if computed
    pca_features: Optional[np.ndarray] = None

    # Optional: material IDs for anchors (if provided)
    anchor_material_ids: Optional[List[str]] = None

    # Metadata
    params: Dict = field(default_factory=dict)


# =============================================================================
# Base Selector Class
# =============================================================================

class BaseAnchorSelector(ABC):
    """Abstract base class for anchor selection methods."""

    def __init__(self, n_anchors: int = 100):
        self.n_anchors = n_anchors

    @abstractmethod
    def fit_transform(
        self,
        embeddings: np.ndarray,
        material_ids: Optional[List[str]] = None,
    ) -> AnchorResult:
        """
        Select anchors from embeddings.

        Args:
            embeddings: Flat embeddings array (n_samples, embed_dim)
            material_ids: Optional material IDs per sample

        Returns:
            AnchorResult with selected anchors
        """
        pass


# =============================================================================
# DIRECT Sampler (via MAML)
# =============================================================================

class DIRECTSelector(BaseAnchorSelector):
    """
    DIRECT sampling using MAML package.

    Uses Birch clustering followed by representative selection from each cluster.

    Args:
        n_anchors: Number of anchors to select (= number of clusters)
        threshold_init: Initial threshold for Birch clustering
        k_per_cluster: Number of samples to select from each cluster
    """

    def __init__(
        self,
        n_anchors: int = 100,
        threshold_init: float = 0.3,
        k_per_cluster: int = 1,
    ):
        super().__init__(n_anchors)
        self.threshold_init = threshold_init
        self.k_per_cluster = k_per_cluster

    def fit_transform(
        self,
        embeddings: np.ndarray,
        material_ids: Optional[List[str]] = None,
    ) -> AnchorResult:
        """Select anchors using DIRECT sampling."""
        try:
            from maml.sampling.direct import (
                BirchClustering,
                DIRECTSampler,
                SelectKFromClusters,
            )
        except ImportError:
            raise ImportError(
                "MAML package required for DIRECT sampling. "
                "Install with: pip install maml"
            )

        embeddings = np.asarray(embeddings)
        n_total, embed_dim = embeddings.shape

        # Create DIRECT sampler
        sampler = DIRECTSampler(
            structure_encoder=None,
            clustering=BirchClustering(n=self.n_anchors, threshold_init=self.threshold_init),
            select_k_from_clusters=SelectKFromClusters(k=self.k_per_cluster),
        )

        # Run selection
        result = sampler.fit_transform(embeddings)

        selected_indices = np.array(result["selected_indexes"])
        anchor_embeddings = embeddings[selected_indices]

        # Get material IDs for anchors if provided
        anchor_mids = None
        if material_ids is not None:
            anchor_mids = [material_ids[i] for i in selected_indices]

        return AnchorResult(
            anchor_embeddings=anchor_embeddings,
            anchor_indices=selected_indices,
            n_anchors=len(selected_indices),
            n_total=n_total,
            embed_dim=embed_dim,
            method="direct",
            pca_features=result.get("PCAfeatures"),
            anchor_material_ids=anchor_mids,
            params={
                "n_anchors": self.n_anchors,
                "threshold_init": self.threshold_init,
                "k_per_cluster": self.k_per_cluster,
            },
        )


# =============================================================================
# K-Means Selector
# =============================================================================

class KMeansSelector(BaseAnchorSelector):
    """
    K-Means clustering based anchor selection.

    Clusters embeddings and selects the point closest to each cluster center.

    Args:
        n_anchors: Number of anchors (= number of clusters)
        random_state: Random seed for reproducibility
        n_init: Number of K-means initializations
    """

    def __init__(
        self,
        n_anchors: int = 100,
        random_state: int = 42,
        n_init: int = 10,
    ):
        super().__init__(n_anchors)
        self.random_state = random_state
        self.n_init = n_init

    def fit_transform(
        self,
        embeddings: np.ndarray,
        material_ids: Optional[List[str]] = None,
    ) -> AnchorResult:
        """Select anchors using K-means clustering."""
        from sklearn.cluster import KMeans

        embeddings = np.asarray(embeddings)
        n_total, embed_dim = embeddings.shape

        # Fit K-means
        kmeans = KMeans(
            n_clusters=self.n_anchors,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.cluster_centers_

        # Select point closest to each center
        selected_indices = []
        for i in range(self.n_anchors):
            cluster_mask = labels == i
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            cluster_points = embeddings[cluster_indices]
            distances = np.linalg.norm(cluster_points - centers[i], axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(closest_idx)

        selected_indices = np.array(selected_indices)
        anchor_embeddings = embeddings[selected_indices]

        anchor_mids = None
        if material_ids is not None:
            anchor_mids = [material_ids[i] for i in selected_indices]

        return AnchorResult(
            anchor_embeddings=anchor_embeddings,
            anchor_indices=selected_indices,
            n_anchors=len(selected_indices),
            n_total=n_total,
            embed_dim=embed_dim,
            method="kmeans",
            cluster_labels=labels,
            cluster_centers=centers,
            anchor_material_ids=anchor_mids,
            params={
                "n_anchors": self.n_anchors,
                "random_state": self.random_state,
                "n_init": self.n_init,
            },
        )


# =============================================================================
# Farthest Point Sampling
# =============================================================================

class FarthestPointSelector(BaseAnchorSelector):
    """
    Farthest Point Sampling (FPS) for anchor selection.

    Iteratively selects points that are farthest from already selected points.
    Provides good coverage of the embedding space.

    Args:
        n_anchors: Number of anchors to select
        random_state: Random seed for initial point selection
        batch_size: Batch size for distance computation (memory efficiency)
    """

    def __init__(
        self,
        n_anchors: int = 100,
        random_state: int = 42,
        batch_size: int = 10000,
    ):
        super().__init__(n_anchors)
        self.random_state = random_state
        self.batch_size = batch_size

    def fit_transform(
        self,
        embeddings: np.ndarray,
        material_ids: Optional[List[str]] = None,
    ) -> AnchorResult:
        """Select anchors using farthest point sampling."""
        embeddings = np.asarray(embeddings)
        n_total, embed_dim = embeddings.shape

        np.random.seed(self.random_state)

        # Initialize with random point
        selected_indices = [np.random.randint(n_total)]
        min_distances = np.full(n_total, np.inf)

        for _ in tqdm(range(self.n_anchors - 1), desc="FPS selection"):
            # Update min distances to selected set
            last_selected = embeddings[selected_indices[-1]]

            # Batch distance computation for memory efficiency
            for start in range(0, n_total, self.batch_size):
                end = min(start + self.batch_size, n_total)
                batch = embeddings[start:end]
                distances = np.linalg.norm(batch - last_selected, axis=1)
                min_distances[start:end] = np.minimum(
                    min_distances[start:end], distances
                )

            # Select point with maximum minimum distance
            # Exclude already selected points
            mask = np.ones(n_total, dtype=bool)
            mask[selected_indices] = False
            masked_distances = np.where(mask, min_distances, -np.inf)
            next_idx = np.argmax(masked_distances)
            selected_indices.append(next_idx)

        selected_indices = np.array(selected_indices)
        anchor_embeddings = embeddings[selected_indices]

        anchor_mids = None
        if material_ids is not None:
            anchor_mids = [material_ids[i] for i in selected_indices]

        return AnchorResult(
            anchor_embeddings=anchor_embeddings,
            anchor_indices=selected_indices,
            n_anchors=len(selected_indices),
            n_total=n_total,
            embed_dim=embed_dim,
            method="fps",
            anchor_material_ids=anchor_mids,
            params={
                "n_anchors": self.n_anchors,
                "random_state": self.random_state,
            },
        )


# =============================================================================
# Random Stratified Selector
# =============================================================================

class StratifiedRandomSelector(BaseAnchorSelector):
    """
    Stratified random sampling based on density estimation.

    Divides the space into density-based strata and samples proportionally
    or uniformly from each stratum.

    Args:
        n_anchors: Number of anchors to select
        n_strata: Number of strata (using K-means for stratification)
        sampling: "uniform" (equal from each stratum) or "proportional"
        random_state: Random seed
    """

    def __init__(
        self,
        n_anchors: int = 100,
        n_strata: int = 20,
        sampling: Literal["uniform", "proportional"] = "uniform",
        random_state: int = 42,
    ):
        super().__init__(n_anchors)
        self.n_strata = n_strata
        self.sampling = sampling
        self.random_state = random_state

    def fit_transform(
        self,
        embeddings: np.ndarray,
        material_ids: Optional[List[str]] = None,
    ) -> AnchorResult:
        """Select anchors using stratified random sampling."""
        from sklearn.cluster import KMeans

        embeddings = np.asarray(embeddings)
        n_total, embed_dim = embeddings.shape

        np.random.seed(self.random_state)

        # Create strata using K-means
        kmeans = KMeans(
            n_clusters=self.n_strata,
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(embeddings)

        # Determine samples per stratum
        if self.sampling == "uniform":
            base_samples = self.n_anchors // self.n_strata
            extra = self.n_anchors % self.n_strata
            samples_per_stratum = [
                base_samples + (1 if i < extra else 0)
                for i in range(self.n_strata)
            ]
        else:  # proportional
            stratum_sizes = np.bincount(labels, minlength=self.n_strata)
            proportions = stratum_sizes / n_total
            samples_per_stratum = np.round(proportions * self.n_anchors).astype(int)
            # Adjust to match exactly n_anchors
            diff = self.n_anchors - samples_per_stratum.sum()
            for i in range(abs(diff)):
                if diff > 0:
                    samples_per_stratum[i % self.n_strata] += 1
                else:
                    idx = np.argmax(samples_per_stratum)
                    samples_per_stratum[idx] -= 1

        # Sample from each stratum
        selected_indices = []
        for stratum_id, n_samples in enumerate(samples_per_stratum):
            stratum_indices = np.where(labels == stratum_id)[0]
            if len(stratum_indices) == 0:
                continue
            n_to_select = min(n_samples, len(stratum_indices))
            selected = np.random.choice(stratum_indices, n_to_select, replace=False)
            selected_indices.extend(selected)

        selected_indices = np.array(selected_indices)
        anchor_embeddings = embeddings[selected_indices]

        anchor_mids = None
        if material_ids is not None:
            anchor_mids = [material_ids[i] for i in selected_indices]

        return AnchorResult(
            anchor_embeddings=anchor_embeddings,
            anchor_indices=selected_indices,
            n_anchors=len(selected_indices),
            n_total=n_total,
            embed_dim=embed_dim,
            method="stratified_random",
            cluster_labels=labels,
            anchor_material_ids=anchor_mids,
            params={
                "n_anchors": self.n_anchors,
                "n_strata": self.n_strata,
                "sampling": self.sampling,
                "random_state": self.random_state,
            },
        )


# =============================================================================
# Unified Interface
# =============================================================================

class AnchorSelector:
    """
    Unified interface for anchor selection methods.

    Args:
        method: Selection method
            - "direct": DIRECT sampling with Birch clustering (requires maml)
            - "kmeans": K-means clustering
            - "fps": Farthest point sampling
            - "stratified": Stratified random sampling
        n_anchors: Number of anchors to select
        **kwargs: Method-specific parameters

    Example:
        >>> selector = AnchorSelector(method="direct", n_anchors=100)
        >>> result = selector.fit_transform(flat_embeddings)
        >>> anchors = result.anchor_embeddings
        >>> indices = result.anchor_indices
    """

    METHODS = {
        "direct": DIRECTSelector,
        "kmeans": KMeansSelector,
        "fps": FarthestPointSelector,
        "farthest_point": FarthestPointSelector,
        "stratified": StratifiedRandomSelector,
        "stratified_random": StratifiedRandomSelector,
    }

    def __init__(
        self,
        method: Literal["direct", "kmeans", "fps", "stratified"] = "direct",
        n_anchors: int = 100,
        **kwargs
    ):
        self.method = method.lower()
        self.n_anchors = n_anchors
        self.kwargs = kwargs

        if self.method not in self.METHODS:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Supported: {list(self.METHODS.keys())}"
            )

        self.selector = self.METHODS[self.method](n_anchors=n_anchors, **kwargs)

    def fit_transform(
        self,
        embeddings: np.ndarray,
        material_ids: Optional[List[str]] = None,
    ) -> AnchorResult:
        """
        Select anchors from embeddings.

        Args:
            embeddings: Flat embeddings (n_samples, embed_dim)
            material_ids: Optional material IDs per sample

        Returns:
            AnchorResult with selected anchors
        """
        return self.selector.fit_transform(embeddings, material_ids)

    def save_anchors(
        self,
        result: AnchorResult,
        output_path: str,
        save_format: str = "pkl",
    ):
        """
        Save anchor selection results.

        Args:
            result: AnchorResult to save
            output_path: Output file path (without extension)
            save_format: "pkl" for pickle, "npy" for numpy, "both"
        """
        output_path = str(output_path)
        if output_path.endswith('.pkl') or output_path.endswith('.npy'):
            output_path = output_path.rsplit('.', 1)[0]

        if save_format in ["pkl", "both"]:
            # Save full result as pickle
            with open(f"{output_path}_anchors.pkl", 'wb') as f:
                pickle.dump(result.anchor_embeddings, f)
            with open(f"{output_path}_indices.pkl", 'wb') as f:
                pickle.dump(result.anchor_indices, f)
            with open(f"{output_path}_result.pkl", 'wb') as f:
                pickle.dump(result, f)
            print(f"Saved anchors to {output_path}_anchors.pkl")

        if save_format in ["npy", "both"]:
            np.save(f"{output_path}_anchors.npy", result.anchor_embeddings)
            np.save(f"{output_path}_indices.npy", result.anchor_indices)
            print(f"Saved anchors to {output_path}_anchors.npy")

    @staticmethod
    def load_anchors(path: str) -> np.ndarray:
        """Load anchor embeddings from file."""
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                return pickle.load(f)
        elif path.endswith('.npy'):
            return np.load(path)
        else:
            raise ValueError(f"Unknown file format: {path}")


# =============================================================================
# Convenience Functions
# =============================================================================

def select_anchors_direct(
    embeddings: np.ndarray,
    n_anchors: int = 100,
    threshold_init: float = 0.3,
    k_per_cluster: int = 1,
    material_ids: Optional[List[str]] = None,
) -> AnchorResult:
    """
    Select anchors using DIRECT sampling.

    This is the method you're currently using with MAML.

    Args:
        embeddings: Flat embeddings (n_samples, embed_dim)
        n_anchors: Number of anchors (= number of Birch clusters)
        threshold_init: Initial threshold for Birch clustering
        k_per_cluster: Samples per cluster
        material_ids: Optional material IDs

    Returns:
        AnchorResult with selected anchors
    """
    selector = AnchorSelector(
        method="direct",
        n_anchors=n_anchors,
        threshold_init=threshold_init,
        k_per_cluster=k_per_cluster,
    )
    return selector.fit_transform(embeddings, material_ids)


def select_anchors_kmeans(
    embeddings: np.ndarray,
    n_anchors: int = 100,
    random_state: int = 42,
    material_ids: Optional[List[str]] = None,
) -> AnchorResult:
    """Select anchors using K-means clustering."""
    selector = AnchorSelector(
        method="kmeans",
        n_anchors=n_anchors,
        random_state=random_state,
    )
    return selector.fit_transform(embeddings, material_ids)


def select_anchors_fps(
    embeddings: np.ndarray,
    n_anchors: int = 100,
    random_state: int = 42,
    material_ids: Optional[List[str]] = None,
) -> AnchorResult:
    """Select anchors using farthest point sampling."""
    selector = AnchorSelector(
        method="fps",
        n_anchors=n_anchors,
        random_state=random_state,
    )
    return selector.fit_transform(embeddings, material_ids)


# =============================================================================
# Visualization
# =============================================================================

def plot_anchor_selection(
    embeddings: np.ndarray,
    result: AnchorResult,
    max_points: int = 5000,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
):
    """
    Visualize anchor selection results using PCA projection.

    Args:
        embeddings: Original embeddings
        result: AnchorResult from selection
        max_points: Maximum points to plot (for performance)
        figsize: Figure size
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # PCA projection
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        anchors_2d = pca.transform(result.anchor_embeddings)
    else:
        embeddings_2d = embeddings
        anchors_2d = result.anchor_embeddings

    # Subsample for visualization
    n_total = len(embeddings_2d)
    if n_total > max_points:
        idx = np.random.choice(n_total, max_points, replace=False)
        embeddings_2d_sub = embeddings_2d[idx]
    else:
        embeddings_2d_sub = embeddings_2d

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: All points with anchors highlighted
    ax1 = axes[0]
    ax1.scatter(embeddings_2d_sub[:, 0], embeddings_2d_sub[:, 1],
               c='lightgray', alpha=0.3, s=5, label='All points')
    ax1.scatter(anchors_2d[:, 0], anchors_2d[:, 1],
               c='red', alpha=0.8, s=50, marker='*', label='Anchors')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title(f'Anchor Selection ({result.method})\n'
                 f'{result.n_anchors} anchors from {result.n_total} points')
    ax1.legend()

    # Plot 2: Cluster distribution if available
    ax2 = axes[1]
    if result.cluster_labels is not None:
        scatter = ax2.scatter(embeddings_2d_sub[:, 0], embeddings_2d_sub[:, 1],
                             c=result.cluster_labels[idx] if n_total > max_points
                             else result.cluster_labels,
                             cmap='tab20', alpha=0.5, s=5)
        ax2.scatter(anchors_2d[:, 0], anchors_2d[:, 1],
                   c='red', alpha=0.8, s=50, marker='*', edgecolors='black')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('Cluster Assignments')
        plt.colorbar(scatter, ax=ax2, label='Cluster')
    else:
        # Show anchor density
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(embeddings_2d.T)
            density = kde(embeddings_2d_sub.T)
            scatter = ax2.scatter(embeddings_2d_sub[:, 0], embeddings_2d_sub[:, 1],
                                 c=density, cmap='viridis', alpha=0.5, s=5)
            ax2.scatter(anchors_2d[:, 0], anchors_2d[:, 1],
                       c='red', alpha=0.8, s=50, marker='*', edgecolors='white')
            ax2.set_xlabel('PC1')
            ax2.set_ylabel('PC2')
            ax2.set_title('Density with Anchors')
            plt.colorbar(scatter, ax=ax2, label='Density')
        except Exception:
            ax2.text(0.5, 0.5, 'Density estimation failed',
                    transform=ax2.transAxes, ha='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """Example demonstrating anchor selection methods."""
    print("=" * 80)
    print("Anchor Selection Examples")
    print("=" * 80)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 10000
    embed_dim = 128

    # Simulated atom embeddings (mixture of clusters)
    centers = np.random.randn(5, embed_dim) * 2
    embeddings = np.vstack([
        centers[i] + np.random.randn(n_samples // 5, embed_dim) * 0.5
        for i in range(5)
    ])
    np.random.shuffle(embeddings)

    material_ids = [f"mat_{i // 20}" for i in range(n_samples)]

    print(f"\nInput: {embeddings.shape[0]} atoms, {embed_dim}-dim embeddings")

    # =========================================================================
    # Method 1: DIRECT (your current approach)
    # =========================================================================
    print("\n--- Method 1: DIRECT Sampling ---")
    print("""
    # Using MAML's DIRECT sampler (your current approach)
    result = select_anchors_direct(
        embeddings=flat_embeddings,
        n_anchors=100,
        threshold_init=0.3,
        k_per_cluster=1,
    )
    """)

    # =========================================================================
    # Method 2: K-means
    # =========================================================================
    print("\n--- Method 2: K-Means ---")
    result_kmeans = select_anchors_kmeans(embeddings, n_anchors=100)
    print(f"Selected {result_kmeans.n_anchors} anchors using K-means")
    print(f"Anchor shape: {result_kmeans.anchor_embeddings.shape}")

    # =========================================================================
    # Method 3: Farthest Point Sampling
    # =========================================================================
    print("\n--- Method 3: Farthest Point Sampling ---")
    result_fps = select_anchors_fps(embeddings, n_anchors=100)
    print(f"Selected {result_fps.n_anchors} anchors using FPS")

    # =========================================================================
    # Unified Interface
    # =========================================================================
    print("\n--- Unified Interface ---")
    print("""
    from anchor_selection import AnchorSelector

    # DIRECT (requires maml)
    selector = AnchorSelector(method="direct", n_anchors=100)

    # K-means
    selector = AnchorSelector(method="kmeans", n_anchors=100)

    # Farthest point sampling
    selector = AnchorSelector(method="fps", n_anchors=100)

    # Stratified random
    selector = AnchorSelector(method="stratified", n_anchors=100, n_strata=20)

    result = selector.fit_transform(flat_embeddings, material_ids)
    selector.save_anchors(result, "output/anchors")
    """)

    # =========================================================================
    # Full Pipeline Example
    # =========================================================================
    print("\n--- Full Pipeline with Embedding Extraction ---")
    print("""
    from mlip_embedding_extractor import MLIPEmbeddingExtractor
    from anchor_selection import AnchorSelector

    # 1. Extract embeddings
    extractor = MLIPEmbeddingExtractor("mace", "/path/to/model")
    results = extractor.extract_from_json("structures.json")

    # 2. Get flat embeddings
    flat_emb = results.flat_embeddings
    flat_ids = results.flat_material_ids

    # 3. Select anchors
    selector = AnchorSelector(method="direct", n_anchors=100)
    anchor_result = selector.fit_transform(flat_emb, flat_ids)

    # 4. Save anchors
    selector.save_anchors(anchor_result, "mace_anchors_100")

    # Access results
    anchors = anchor_result.anchor_embeddings  # (100, embed_dim)
    indices = anchor_result.anchor_indices     # indices into flat_emb
    """)

    return result_kmeans, result_fps


if __name__ == "__main__":
    example_usage()
