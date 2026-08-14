#!/usr/bin/env python3
"""
Análisis geométrico parcial — Gemma4-31B-base
Corre E1 (axis vs generic_long) y E2 (generic_long vs generic_short)
con los embeddings disponibles (falta vanilla, pod caído a mitad de extracción).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "shared"))

import numpy as np
import networkx as nx
from curvature_analyzer import FormanRicciAnalyzer
from graph_builder import TrajectoryGraphBuilder
from dimensionality_analyzer import DimensionalityComparator, FractalDimensionEstimator
from statistical_tests import GeometricStatisticalTests

RESULTS = Path(__file__).parent / "results_fase1"


def load_group(name):
    d = np.load(RESULTS / f"{name}_embeddings.npz", allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)
    lengths = d["lengths"].astype(int)
    return emb, lengths


def build_graphs(emb, lengths, builder):
    graphs = []
    for i, L in enumerate(lengths):
        traj = emb[i, :L]
        G = builder.build(list(traj))
        graphs.append(G)
    return graphs


def main():
    builder = TrajectoryGraphBuilder(k_nn=5)
    analyzer = FormanRicciAnalyzer()
    stat = GeometricStatisticalTests(n_permutations=1000)
    dim_est = FractalDimensionEstimator(method="mle")
    dim_comp = DimensionalityComparator(dim_est)

    groups = {}
    for name in ["axis", "generic_long", "generic_short"]:
        emb, lengths = load_group(name)
        groups[name] = (emb, lengths)
        print(f"{name}: {emb.shape[0]} trayectorias, longitud media={lengths.mean():.1f}, dim={emb.shape[2]}")

    print("\nConstruyendo grafos k-NN...")
    graphs = {}
    dims = {}
    for name, (emb, lengths) in groups.items():
        graphs[name] = build_graphs(emb, lengths, builder)
        dims[name] = []
        for i, L in enumerate(lengths):
            if L > 5:
                try:
                    dims[name].append(float(dim_est.estimate(emb[i, :L])))
                except Exception:
                    pass

    def compare(name_a, name_b):
        print(f"\n{'='*60}\n{name_a} vs {name_b}\n{'='*60}")
        comp = analyzer.compare_groups(graphs[name_a], graphs[name_b])
        w1 = stat.wasserstein_distance(comp["curvatures_a"], comp["curvatures_b"])
        perm = stat.permutation_test(comp["curvatures_a"], comp["curvatures_b"])
        print(f"  κ̄_{name_a} = {comp['mean_a']:.4f} ± {comp['std_a']:.4f}")
        print(f"  κ̄_{name_b} = {comp['mean_b']:.4f} ± {comp['std_b']:.4f}")
        print(f"  Δκ = {comp['mean_difference']:+.4f} | d = {comp['cohens_d']:+.4f}")
        print(f"  W₁ = {w1:.4f} | p = {perm['p_value']:.6f}")

        dim_r = dim_comp.compare_groups_from_values(dims.get(name_a, []), dims.get(name_b, []))
        print(f"  dim_{name_a} = {dim_r['dim_a']['mean']:.2f} | dim_{name_b} = {dim_r['dim_b']['mean']:.2f} | reducción = {dim_r['reduction_percent']:+.1f}%")
        return comp, w1, perm

    print("\n" + "#"*60)
    print("# E1 — axis vs generic_long (identidad, longitud emparejada)")
    print("#"*60)
    compare("axis", "generic_long")

    print("\n" + "#"*60)
    print("# E2 — generic_long vs generic_short (efecto puro de longitud)")
    print("#"*60)
    compare("generic_long", "generic_short")

    print("\n" + "#"*60)
    print("# BONUS — axis vs generic_short")
    print("#"*60)
    compare("axis", "generic_short")

    print("\n[NOTA] vanilla no disponible — pod cayó a mitad de extracción (10/20).")
    print("E3 (axis vs vanilla) pendiente hasta re-ejecutar ese grupo.")


if __name__ == "__main__":
    main()
