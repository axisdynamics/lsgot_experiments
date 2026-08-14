"""
Pruebas estadísticas para validación geométrica LSGOT.

Métricas:
- Wasserstein Distance (Earth Mover's Distance) via POT
- Permutation test (distribución nula por permutación)
- Mann-Whitney U (no paramétrico)
- Cohen's d (tamaño del efecto)
- Bootstrap CI
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy import stats


class GeometricStatisticalTests:
    """Batería de pruebas estadísticas para comparación geométrica."""

    def __init__(self, n_permutations: int = 1000, random_seed: int = 42):
        self.n_permutations = n_permutations
        self.rng = np.random.RandomState(random_seed)

    def wasserstein_distance(self, samples_a: List[float],
                              samples_b: List[float]) -> float:
        """Calcula Wasserstein-1 (Earth Mover's Distance) entre dos muestras 1D.
        Usa scipy (O(n log n), sin matriz de distancias) para muestras grandes."""
        from scipy.stats import wasserstein_distance as sci_wd
        return float(sci_wd(samples_a, samples_b))

    def permutation_test(self, samples_a: List[float],
                          samples_b: List[float],
                          metric: str = 'mean_difference') -> Dict:
        """
        Prueba de permutación bilateral.

        Args:
            samples_a, samples_b: Muestras a comparar
            metric: 'mean_difference' o 'wasserstein'

        Returns:
            Dict con observed_statistic, p_value, significant
        """
        combined = np.concatenate([samples_a, samples_b])
        n_a = len(samples_a)

        if metric == 'mean_difference':
            observed = float(np.mean(samples_a) - np.mean(samples_b))
            def compute_stat(perm_a, perm_b):
                return float(np.mean(perm_a) - np.mean(perm_b))
        elif metric == 'wasserstein':
            observed = self.wasserstein_distance(samples_a, samples_b)
            def compute_stat(perm_a, perm_b):
                return self.wasserstein_distance(perm_a.tolist(), perm_b.tolist())
        else:
            raise ValueError(f"Unknown metric: {metric}")

        null_distribution = []
        for _ in range(self.n_permutations):
            permuted = self.rng.permutation(combined)
            null_stat = compute_stat(permuted[:n_a], permuted[n_a:])
            null_distribution.append(null_stat)

        null_arr = np.array(null_distribution)
        if observed >= 0:
            p_value = float(np.mean(null_arr >= observed))
        else:
            p_value = float(np.mean(null_arr <= observed))

        return {
            'observed_statistic': observed,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'n_permutations': self.n_permutations
        }

    def mann_whitney_u(self, samples_a: List[float],
                        samples_b: List[float]) -> Dict:
        """Mann-Whitney U test (no paramétrico)."""
        if not samples_a or not samples_b:
            return {'statistic': 0.0, 'p_value': 1.0, 'significant': False}
        result = stats.mannwhitneyu(samples_a, samples_b, alternative='two-sided')
        return {
            'statistic': float(result.statistic),
            'p_value': float(result.pvalue),
            'significant': bool(result.pvalue < 0.05)
        }

    def cohens_d(self, samples_a: List[float], samples_b: List[float]) -> float:
        """Tamaño del efecto de Cohen."""
        if len(samples_a) < 2 or len(samples_b) < 2:
            return 0.0
        std_a = np.std(samples_a, ddof=1)
        std_b = np.std(samples_b, ddof=1)
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        if pooled_std == 0:
            return 0.0
        return float((np.mean(samples_a) - np.mean(samples_b)) / pooled_std)

    def bootstrap_ci(self, samples: List[float], confidence: float = 0.95,
                      n_boot: int = 1000) -> Tuple[float, float]:
        """Intervalo de confianza bootstrap para la media."""
        boot_means = [
            float(np.mean(self.rng.choice(samples, size=len(samples), replace=True)))
            for _ in range(n_boot)
        ]
        alpha = (1 - confidence) / 2
        return (float(np.percentile(boot_means, alpha * 100)),
                float(np.percentile(boot_means, (1 - alpha) * 100)))

    def full_comparison(self, samples_a: List[float],
                         samples_b: List[float],
                         label_a: str = "A",
                         label_b: str = "B") -> Dict:
        """
        Ejecuta batería completa de tests entre dos muestras.
        """
        if not samples_a or not samples_b:
            return {'error': 'Empty samples'}

        w1 = self.wasserstein_distance(samples_a, samples_b)
        perm = self.permutation_test(samples_a, samples_b, 'mean_difference')
        mw = self.mann_whitney_u(samples_a, samples_b)
        d = self.cohens_d(samples_a, samples_b)
        ci_a = self.bootstrap_ci(samples_a)
        ci_b = self.bootstrap_ci(samples_b)

        return {
            f'mean_{label_a}': float(np.mean(samples_a)),
            f'mean_{label_b}': float(np.mean(samples_b)),
            f'std_{label_a}': float(np.std(samples_a)),
            f'std_{label_b}': float(np.std(samples_b)),
            f'ci95_{label_a}': ci_a,
            f'ci95_{label_b}': ci_b,
            'wasserstein_w1': w1,
            'permutation_test': perm,
            'mann_whitney_u': mw,
            'cohens_d': d,
            'effect_size_interpretation': self._interpret_d(d)
        }

    def _interpret_d(self, d: float) -> str:
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
