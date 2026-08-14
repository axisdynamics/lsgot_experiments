"""
Construcción de grafos de trayectoria a partir de embeddings.

Cada grafo tiene:
- Aristas temporales: conectan pasos consecutivos de la trayectoria
- Aristas semánticas: conectan embeddings cercanos en espacio coseno (k-NN)
"""

import numpy as np
import networkx as nx
from typing import List
from sklearn.neighbors import NearestNeighbors


class TrajectoryGraphBuilder:
    """
    Construye grafo mixto (temporal + semántico) de una trayectoria.
    """

    def __init__(self, k_nn: int = 5, temporal_weight: float = 1.0):
        self.k_nn = k_nn
        self.temporal_weight = temporal_weight

    def build(self, embeddings: List[np.ndarray]) -> nx.Graph:
        """
        Construye grafo a partir de secuencia de embeddings.

        Args:
            embeddings: Lista de vectores numpy

        Returns:
            Grafo nx.Graph con nodos indexados por posición temporal
        """
        G = nx.Graph()
        n = len(embeddings)

        if n < 2:
            if n == 1:
                G.add_node(0, embedding=embeddings[0], position=0)
            return G

        emb_matrix = np.vstack(embeddings)

        # Aristas temporales
        for i in range(n - 1):
            cos_dist = 1.0 - self._cosine_similarity(embeddings[i], embeddings[i + 1])
            weight = self.temporal_weight / (1.0 + cos_dist)
            G.add_edge(i, i + 1, weight=weight, edge_type='temporal')

        # Aristas semánticas (k-NN)
        k_actual = min(self.k_nn + 1, n)
        nn = NearestNeighbors(n_neighbors=k_actual, metric='cosine', algorithm='brute')
        nn.fit(emb_matrix)
        distances, indices = nn.kneighbors(emb_matrix)

        for i in range(n):
            for j_idx, j in enumerate(indices[i]):
                if i == j or i >= j:
                    continue
                cos_dist = distances[i, j_idx]
                weight = 1.0 / (1.0 + cos_dist)
                if G.has_edge(i, j):
                    # Actualizar peso si el semántico es mayor
                    if G[i][j]['weight'] < weight:
                        G[i][j]['weight'] = weight
                        G[i][j]['edge_type'] = 'semantic'
                else:
                    G.add_edge(i, j, weight=weight, edge_type='semantic')

        # Solo almacenar posición — los embeddings se mantienen en el dict de trayectorias
        for i in range(n):
            G.nodes[i]['position'] = i

        return G

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
