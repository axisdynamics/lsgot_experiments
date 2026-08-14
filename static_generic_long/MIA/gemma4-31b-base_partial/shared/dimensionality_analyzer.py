"""
Estimación de dimensión fractal (intrínseca) de nubes de puntos.

Métodos disponibles:
- 'correlation': Grassberger-Procaccia (log-log slope de función de correlación)
- 'mle': Maximum Likelihood Estimation (Levina & Bickel, 2004)

LSGOT predice: dim(VEX) < dim(vanilla), con reducción teórica 40-80%.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from typing import List, Dict, Tuple


class FractalDimensionEstimator:
    """
    Estima dimensión fractal intrínseca de una nube de puntos.
    """

    def __init__(self, method: str = 'mle', n_neighbors: int = 10, n_scales: int = 20):
        self.method = method
        self.n_neighbors = n_neighbors
        self.n_scales = n_scales

    def estimate(self, points: np.ndarray) -> float:
        """
        Estima dimensión fractal.

        Args:
            points: Matriz (n_samples, n_features)

        Returns:
            Dimensión fractal estimada (float)
        """
        if len(points) < max(self.n_neighbors + 2, 5):
            return 0.0

        if self.method == 'correlation':
            return self._correlation_dimension(points)
        elif self.method == 'mle':
            return self._mle_dimension(points)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _correlation_dimension(self, points: np.ndarray) -> float:
        """Método Grassberger-Procaccia: log(C(r)) ~ D * log(r)."""
        n = len(points)
        sample_size = min(300, n)
        idx = np.random.choice(n, sample_size, replace=False)
        sampled = points[idx]

        dists = euclidean_distances(sampled)
        flat_dists = dists[~np.eye(sample_size, dtype=bool)]
        flat_dists = flat_dists[flat_dists > 0]

        if len(flat_dists) == 0:
            return 0.0

        log_r_min = np.log(np.min(flat_dists) + 1e-10)
        log_r_max = np.log(np.max(flat_dists))
        if log_r_max <= log_r_min:
            return 0.0

        log_r = np.linspace(log_r_min, log_r_max, self.n_scales)
        log_C = []
        for r in np.exp(log_r):
            C = np.mean(flat_dists < r)
            log_C.append(np.log(C) if C > 0 else None)

        valid_pairs = [(log_r[i], log_C[i]) for i in range(len(log_C))
                       if log_C[i] is not None]
        if len(valid_pairs) < 2:
            return 0.0

        lr, lc = zip(*valid_pairs)
        from scipy import stats
        slope, _, _, _, _ = stats.linregress(lr, lc)
        return max(0.0, float(slope))

    def _mle_dimension(self, points: np.ndarray) -> float:
        """Estimador MLE (Levina & Bickel, 2004)."""
        n, d = points.shape
        k = min(self.n_neighbors, n - 2)
        if k < 2:
            return 0.0

        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nn.fit(points)
        distances, _ = nn.kneighbors(points)
        distances = distances[:, 1:]  # excluir distancia a sí mismo

        # Evitar log(0)
        distances = np.maximum(distances, 1e-10)
        log_dist = np.log(distances)

        # dim_i = (k-1) / sum(log(r_k/r_i)) para i < k
        r_k = distances[:, -1:]
        log_ratios = np.log(r_k) - log_dist[:, :-1]
        log_ratios = np.maximum(log_ratios, 1e-10)
        dim_i = (k - 1) / np.sum(log_ratios, axis=1)
        dim_i = np.clip(dim_i, 0, d)

        return float(np.mean(dim_i))


class DimensionalityComparator:
    """Compara dimensionalidad intrínseca entre grupos experimentales."""

    def __init__(self, estimator: FractalDimensionEstimator):
        self.estimator = estimator

    def compare_groups(self, matrices_a: List[np.ndarray],
                       matrices_b: List[np.ndarray],
                       n_bootstrap: int = 100) -> Dict:
        """
        Compara dimensión fractal entre dos grupos con bootstrap CI.
        """
        dims_a = [self.estimator.estimate(m) for m in matrices_a if len(m) > 5]
        dims_b = [self.estimator.estimate(m) for m in matrices_b if len(m) > 5]

        if not dims_a or not dims_b:
            return {
                'dim_a': {'mean': 0.0, 'std': 0.0, 'values': dims_a},
                'dim_b': {'mean': 0.0, 'std': 0.0, 'values': dims_b},
                'mean_difference': 0.0,
                'reduction_percent': 0.0,
                'ci_lower': 0.0, 'ci_upper': 0.0
            }

        rng = np.random.RandomState(42)
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            sa = rng.choice(dims_a, len(dims_a), replace=True)
            sb = rng.choice(dims_b, len(dims_b), replace=True)
            bootstrap_diffs.append(float(np.mean(sa) - np.mean(sb)))

        mean_a = float(np.mean(dims_a))
        mean_b = float(np.mean(dims_b))
        reduction = ((mean_b - mean_a) / mean_b * 100) if mean_b > 0 else 0.0

        return {
            'dim_a': {'mean': mean_a, 'std': float(np.std(dims_a)), 'values': dims_a},
            'dim_b': {'mean': mean_b, 'std': float(np.std(dims_b)), 'values': dims_b},
            'mean_difference': mean_a - mean_b,
            'reduction_percent': reduction,
            'ci_lower': float(np.percentile(bootstrap_diffs, 2.5)),
            'ci_upper': float(np.percentile(bootstrap_diffs, 97.5)),
            'bootstrap_diffs': bootstrap_diffs
        }

    def compare_groups_from_values(self, dims_a: List[float],
                                   dims_b: List[float],
                                   n_bootstrap: int = 100) -> Dict:
        """
        Igual que compare_groups pero recibe dimensiones ya calculadas (listas de floats).
        Usar cuando las dimensiones se calcularon inline durante la extracción.
        """
        dims_a = [d for d in dims_a if d is not None and np.isfinite(d)]
        dims_b = [d for d in dims_b if d is not None and np.isfinite(d)]

        if not dims_a or not dims_b:
            return {
                'dim_a': {'mean': 0.0, 'std': 0.0, 'values': dims_a},
                'dim_b': {'mean': 0.0, 'std': 0.0, 'values': dims_b},
                'mean_difference': 0.0, 'reduction_percent': 0.0,
                'ci_lower': 0.0, 'ci_upper': 0.0, 'bootstrap_diffs': []
            }

        rng = np.random.RandomState(42)
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            sa = rng.choice(dims_a, len(dims_a), replace=True)
            sb = rng.choice(dims_b, len(dims_b), replace=True)
            bootstrap_diffs.append(float(np.mean(sa) - np.mean(sb)))

        mean_a = float(np.mean(dims_a))
        mean_b = float(np.mean(dims_b))
        reduction = ((mean_b - mean_a) / mean_b * 100) if mean_b > 0 else 0.0

        return {
            'dim_a': {'mean': mean_a, 'std': float(np.std(dims_a)), 'values': dims_a},
            'dim_b': {'mean': mean_b, 'std': float(np.std(dims_b)), 'values': dims_b},
            'mean_difference': mean_a - mean_b,
            'reduction_percent': reduction,
            'ci_lower': float(np.percentile(bootstrap_diffs, 2.5)),
            'ci_upper': float(np.percentile(bootstrap_diffs, 97.5)),
            'bootstrap_diffs': bootstrap_diffs
        }
