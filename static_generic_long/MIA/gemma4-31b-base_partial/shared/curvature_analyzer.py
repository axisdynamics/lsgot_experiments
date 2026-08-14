"""
Cálculo de curvatura de Forman-Ricci en grafos de trayectoria.

Fórmula: κ_F(e_uv) = w(e_uv) - Σ_{f⊃e} w(f)/|f| + Σ_{g⊃e} w(g)/|g|

Donde:
- e_uv: arista entre nodos u y v
- f: triángulos (ciclos de orden 3) que contienen e_uv
- g: cuadriláteros (ciclos de orden 4) que contienen e_uv

κ > 0: región convergente (esférica) — alta coherencia geométrica local
κ < 0: región divergente (hiperbólica) — baja coherencia, mayor dispersión local
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple


class FormanRicciAnalyzer:
    """Calcula curvatura de Forman-Ricci para aristas en un grafo."""

    def __init__(self, include_4cycles: bool = True):
        self.include_4cycles = include_4cycles

    def compute_edge_curvature(self, G: nx.Graph, u: int, v: int) -> float:
        """Calcula curvatura Forman-Ricci para la arista (u, v)."""
        if not G.has_edge(u, v):
            return 0.0

        w_e = G[u][v].get('weight', 1.0)

        # Término triangular
        term_triangles = 0.0
        for w in set(G.neighbors(u)) & set(G.neighbors(v)):
            w_triangle = (
                G[u][v].get('weight', 1.0) +
                G[u][w].get('weight', 1.0) +
                G[v][w].get('weight', 1.0)
            ) / 3.0
            term_triangles += w_triangle / 3.0

        # Término cuadrilateral
        term_quads = 0.0
        if self.include_4cycles:
            neighbors_u = set(G.neighbors(u)) - {v}
            neighbors_v = set(G.neighbors(v)) - {u}
            seen_cycles = set()
            for x in neighbors_u:
                for y in neighbors_v:
                    if x == y:
                        continue
                    if G.has_edge(x, y):
                        cycle_key = tuple(sorted([u, v, x, y]))
                        if cycle_key not in seen_cycles:
                            seen_cycles.add(cycle_key)
                            edges = [(u, v), (u, x), (x, y), (v, y)]
                            weights = [G[a][b].get('weight', 1.0)
                                       for a, b in edges if G.has_edge(a, b)]
                            if weights:
                                term_quads += np.mean(weights) / 4.0

        return w_e - term_triangles + term_quads

    def compute_all_curvatures(self, G: nx.Graph) -> List[float]:
        """Calcula curvatura para todas las aristas del grafo."""
        return [self.compute_edge_curvature(G, u, v) for u, v in G.edges()]

    def compute_curvature_distribution(self, G: nx.Graph) -> Dict:
        """Calcula distribución estadística de curvaturas."""
        curvatures = self.compute_all_curvatures(G)
        if not curvatures:
            return {'mean': 0.0, 'std': 0.0, 'values': []}

        arr = np.array(curvatures)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'median': float(np.median(arr)),
            'q25': float(np.percentile(arr, 25)),
            'q75': float(np.percentile(arr, 75)),
            'positive_ratio': float(np.mean(arr > 0)),
            'negative_ratio': float(np.mean(arr < 0)),
            'values': curvatures
        }

    def compare_groups(self, graphs_a: List[nx.Graph],
                       graphs_b: List[nx.Graph]) -> Dict:
        """Compara distribuciones de curvatura entre dos grupos."""
        curvatures_a = []
        curvatures_b = []

        for G in graphs_a:
            curvatures_a.extend(self.compute_all_curvatures(G))
        for G in graphs_b:
            curvatures_b.extend(self.compute_all_curvatures(G))

        if not curvatures_a or not curvatures_b:
            return {'curvatures_a': curvatures_a, 'curvatures_b': curvatures_b,
                    'mean_a': 0.0, 'mean_b': 0.0, 'mean_difference': 0.0, 'cohens_d': 0.0}

        mean_a = float(np.mean(curvatures_a))
        mean_b = float(np.mean(curvatures_b))
        std_a = float(np.std(curvatures_a))
        std_b = float(np.std(curvatures_b))
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

        return {
            'curvatures_a': curvatures_a,
            'curvatures_b': curvatures_b,
            'mean_a': mean_a,
            'std_a': std_a,
            'mean_b': mean_b,
            'std_b': std_b,
            'mean_difference': mean_a - mean_b,
            'cohens_d': cohens_d,
            'n_a': len(curvatures_a),
            'n_b': len(curvatures_b)
        }
