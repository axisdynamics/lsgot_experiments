"""
SIA Geometric Validation — Gemma 4 31B-it
=========================================
Experimento base SIA (~12K chars VEX estructurado) en modelo 31B dense.

Hipótesis de escalado: SIA necesita ≥70B para integrar estructura VEX completa.
31B es el primer punto de validación intermedia (8B → 31B → 72B).

Uso:
  python run_exp.py --subset --save-embeddings   # 20 prompts prioritarios
  python run_exp.py --layer-analysis             # análisis por capas
  python run_exp.py --token hf_xxx               # HF token
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

# ─── CONFIGURACIÓN Gemma 4 31B-it ────────────────────────────────────────────

MODEL_CONFIG = {
    "hf_id":       "google/gemma-4-31B-it",
    "nv_path":     "/workspace/models/gemma-4-31B-it",
    "results_dir": "results/sia",
    "cache_dir":   "cache_sia",
    "hidden_dim":  5376,
    "n_layers":    60,
    "min_vram_gb": 65,
}

DEFAULT_LAYERS = [15, 30, 40, 50, 60]

# ─── Resto del pipeline (idéntico a MIA, solo cambian paths) ────────────────

import argparse
import os
import json
import gc
import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PRIORITY_SUBSET = [1, 3, 6, 10, 14, 21, 23, 27, 31, 39, 41, 45, 51, 59, 61, 65, 71, 79, 91, 98]

# 4 condiciones: axis, generic_long (token-matched), generic_short, vanilla
GROUPS_CONFIG = {
    "axis":           {"system_prompt_path": "prompts/axis.dna"},
    "generic_long":   {"system_prompt_path": "prompts/generic_long.txt"},
    "generic_short":  {"system_prompt_path": "prompts/generic_short.txt"},
    "vanilla":        {"system_prompt": "You are a helpful assistant."},
}


def get_model_path(model_cfg, hf_token=None):
    import glob as _glob
    nv_path = model_cfg["nv_path"]
    if Path(nv_path).exists() and any(Path(nv_path).iterdir()):
        print(f"Modelo en Network Volume: {nv_path}")
        return nv_path

    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    hf_id = model_cfg["hf_id"]
    cache_name = hf_id.replace("/", "--")
    pattern = f"{hf_cache}/models--{cache_name}/snapshots/*/config.json"
    matches = _glob.glob(pattern)
    if matches:
        snapshot = str(Path(matches[0]).parent)
        print(f"Modelo en HF cache: {snapshot}")
        return snapshot

    print(f"Descargando {hf_id}... (62 GB, ~15-20 min)")
    from huggingface_hub import snapshot_download
    kwargs = {"repo_id": hf_id, "ignore_patterns": ["*.gguf", "*.ggml"]}
    if hf_token:
        kwargs["token"] = hf_token
    if Path("/workspace/models").exists():
        kwargs["local_dir"] = nv_path
    return snapshot_download(**kwargs)


def get_system_prompt(config):
    if "system_prompt_path" in config:
        path = Path(__file__).parent / config["system_prompt_path"]
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return config.get("system_prompt", "You are a helpful assistant.")


def phase_extract(groups, prompts, results_dir, cache_dir, max_new_tokens, hf_token,
                  save_embeddings=False):
    from hidden_state_extractor import HiddenStateExtractor
    from graph_builder import TrajectoryGraphBuilder
    from dimensionality_analyzer import FractalDimensionEstimator
    from trajectory_metrics import compute_trajectory_metrics
    from dataclasses import asdict
    from tqdm import tqdm

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("FASE 1 — Extracción de Hidden States (SIA)")
    print("="*60)
    print(f"GPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  {p.name} | {p.total_memory/1e9:.1f} GB VRAM")
    print(f"Grupos: {groups} | Preguntas: {len(prompts)} | max_tokens: {max_new_tokens}")

    model_path = get_model_path(MODEL_CONFIG, hf_token)
    extractor = HiddenStateExtractor(
        model_path, layer_idx=-1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_dir=cache_dir, max_new_tokens=max_new_tokens,
        min_vram_gb=MODEL_CONFIG.get("min_vram_gb", 65)
    )
    extractor.load_model()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dim_est = FractalDimensionEstimator(method="mle")
    builder = TrajectoryGraphBuilder(k_nn=5)

    graphs_json, dims_json, traj_metrics_json = {}, {}, {}

    for group in groups:
        print(f"\n{'─'*40}\nGrupo: {group}")
        sys_prompt = get_system_prompt(GROUPS_CONFIG[group])
        edge_lists, group_dims, group_metrics = [], [], []
        emb_matrices, responses_out, n_tokens_list = [], [], []

        for p in tqdm(prompts, desc=group):
            text = p.get("text", p.get("question", ""))
            traj = extractor.extract(text, sys_prompt, group=group)

            G = builder.build(traj.embeddings)
            edges = [(int(u), int(v), float(d["weight"])) for u, v, d in G.edges(data=True)]
            edge_lists.append(edges)

            emb_arr = traj.embedding_matrix
            if traj.num_steps > 5:
                try:
                    group_dims.append(float(dim_est.estimate(emb_arr)))
                except Exception:
                    pass
            try:
                m = compute_trajectory_metrics(emb_arr)
                group_metrics.append(asdict(m))
            except Exception as e:
                print(f"  [metrics] {e}")

            if save_embeddings:
                emb_matrices.append(emb_arr.astype(np.float16))

            n_tokens_list.append(traj.num_steps)
            responses_out.append({
                "id": p["id"], "prompt": traj.prompt_text,
                "response": traj.response_text, "n_tokens": traj.num_steps
            })
            del traj, G, edges, emb_arr
            gc.collect()

        print(f"  {len(edge_lists)} trayectorias | tokens/traj: {np.mean(n_tokens_list):.1f}")

        graphs_json[group] = edge_lists
        dims_json[group] = group_dims
        traj_metrics_json[group] = group_metrics

        with open(results_dir / f"{group}_responses.json", "w", encoding="utf-8") as f:
            json.dump(responses_out, f, ensure_ascii=False, indent=2)

        if save_embeddings and emb_matrices:
            max_T = max(m.shape[0] for m in emb_matrices)
            D = emb_matrices[0].shape[1]
            padded = np.zeros((len(emb_matrices), max_T, D), dtype=np.float16)
            lengths = []
            for i, m in enumerate(emb_matrices):
                T = m.shape[0]
                padded[i, :T, :] = m
                lengths.append(T)
            npz_path = results_dir / f"{group}_embeddings.npz"
            np.savez_compressed(npz_path, embeddings=padded, lengths=np.array(lengths))
            print(f"  Embeddings: {npz_path.name} ({npz_path.stat().st_size/1e6:.1f} MB)")
            del padded, emb_matrices
        else:
            del emb_matrices
        del edge_lists, group_dims, responses_out
        gc.collect()

    with open(results_dir / "_graphs.json", "w") as f:
        json.dump(graphs_json, f)
    with open(results_dir / "_dims.json", "w") as f:
        json.dump(dims_json, f)
    with open(results_dir / "_traj_metrics.json", "w") as f:
        json.dump(traj_metrics_json, f)

    print(f"\nFase 1 completa → {results_dir}/")
    extractor._model = None
    extractor._tokenizer = None
    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def phase_analyze(groups, prompts, results_dir):
    import networkx as nx
    from curvature_analyzer import FormanRicciAnalyzer
    from dimensionality_analyzer import DimensionalityComparator, FractalDimensionEstimator
    from statistical_tests import GeometricStatisticalTests
    from trajectory_metrics import _agg, print_metrics_summary

    results_dir = Path(results_dir)
    print("\n" + "="*60)
    print("FASE 2 — Análisis Geométrico (SIA)")
    print("="*60)

    with open(results_dir / "_graphs.json") as f:
        graphs_json = json.load(f)
    with open(results_dir / "_dims.json") as f:
        dims_json = json.load(f)

    traj_metrics_agg = {}
    traj_metrics_path = results_dir / "_traj_metrics.json"
    if traj_metrics_path.exists():
        with open(traj_metrics_path) as f:
            raw = json.load(f)
        fields = ["recurrence_rate", "mean_velocity", "velocity_std",
                  "velocity_trend", "velocity_centroid_corr",
                  "entropy", "prompt_response_cosine"]
        for group, metrics_list in raw.items():
            traj_metrics_agg[group] = {
                fld: _agg([m[fld] for m in metrics_list if fld in m]) for fld in fields
            }

    graphs = {}
    for group, edge_lists in graphs_json.items():
        group_graphs = []
        for edges in edge_lists:
            G = nx.Graph()
            G.add_weighted_edges_from(edges)
            group_graphs.append(G)
        graphs[group] = group_graphs

    del graphs_json
    gc.collect()

    analyzer = FormanRicciAnalyzer()
    stat = GeometricStatisticalTests(n_permutations=1000)
    dim_comp = DimensionalityComparator(FractalDimensionEstimator(method="mle"))

    results = {
        "experiment": {"model": MODEL_CONFIG["hf_id"], "n_prompts": len(prompts),
                       "groups": groups, "hidden_dim": MODEL_CONFIG["hidden_dim"],
                       "n_layers": MODEL_CONFIG["n_layers"]},
        "comparisons": {},
        "trajectory_metrics": traj_metrics_agg,
    }

    available = list(graphs.keys())
    BASELINE = next((b for b in ("vanilla",) if b in available),
                    available[-1])
    compare_pairs = [(g, BASELINE) for g in available if g != BASELINE]
    # E1 (primary): axis vs generic_long — longitud emparejada
    if "axis" in graphs and "generic_long" in graphs:
        compare_pairs.append(("axis", "generic_long"))

    for name_a, name_b in compare_pairs:
        label = f"{name_a}_vs_{name_b}"
        print(f"\n{'─'*40}\nComparación: {name_a} vs {name_b}")
        comp = analyzer.compare_groups(graphs[name_a], graphs[name_b])
        w1 = stat.wasserstein_distance(comp["curvatures_a"], comp["curvatures_b"])
        perm = stat.permutation_test(comp["curvatures_a"], comp["curvatures_b"])
        mw = stat.mann_whitney_u(comp["curvatures_a"], comp["curvatures_b"])
        print(f"  Δκ = {comp['mean_difference']:.4f} | d = {comp['cohens_d']:.4f}")
        print(f"  W₁ = {w1:.4f} | p = {perm['p_value']:.8f}")

        dim_r = dim_comp.compare_groups_from_values(
            dims_json.get(name_a, []), dims_json.get(name_b, []))
        print(f"  Reducción dim = {dim_r['reduction_percent']:.2f}%")

        results["comparisons"][label] = {
            "group_a": name_a, "group_b": name_b,
            "curvature": {**comp, "wasserstein_distance": w1,
                          "permutation_test": perm, "mann_whitney": mw},
            "dimensionality": dim_r,
        }

    if traj_metrics_agg:
        print("\n" + "="*60)
        print("MÉTRICAS DE TRAYECTORIA (SIA)")
        print("="*60)
        print_metrics_summary(traj_metrics_agg)

    def to_json(obj):
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, dict): return {k: to_json(v) for k, v in obj.items()}
        if isinstance(obj, list): return [to_json(i) for i in obj]
        return obj

    with open(results_dir / "results.json", "w") as f:
        json.dump(to_json(results), f, indent=2)

    print("\n" + "="*60)
    print("RESUMEN SIA 31B")
    print("="*60)
    for label, cmp in results["comparisons"].items():
        curv = cmp["curvature"]
        dim = cmp["dimensionality"]
        perm_r = curv.get("permutation_test", {})
        red = dim.get("reduction_percent", 0)
        h1 = bool(perm_r.get("significant") and abs(curv.get("mean_difference", 0)) > 0.1)
        h2 = bool(abs(red) > 18)
        h3 = bool(perm_r.get("p_value", 1) < 0.001)
        print(f"  [{label}] H₁={'✓' if h1 else '—'} H₂={'✓' if h2 else '—'} H₃={'✓' if h3 else '—'}")

    print(f"\nResultados → {results_dir}/results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"SIA on {MODEL_CONFIG['hf_id']}")
    parser.add_argument("--subset", action="store_true")
    parser.add_argument("--groups", nargs="+", default=None, choices=list(GROUPS_CONFIG.keys()))
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    groups = args.groups or list(GROUPS_CONFIG.keys())
    results_dir = MODEL_CONFIG["results_dir"]
    cache_dir = MODEL_CONFIG["cache_dir"]

    print(f"\nSIA — {MODEL_CONFIG['hf_id']}")
    print(f"Grupos: {groups} | Results: {results_dir}/")

    if args.force and not args.analyze_only:
        import shutil
        if Path(cache_dir).exists():
            shutil.rmtree(cache_dir)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    with open("../data/prompts.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if args.subset:
        prompts = [p for p in prompts if p["id"] in PRIORITY_SUBSET]

    if args.analyze_only:
        phase_analyze(groups, prompts, results_dir)
    elif args.extract_only:
        phase_extract(groups, prompts, results_dir, cache_dir,
                      args.max_tokens, args.token, save_embeddings=args.save_embeddings)
    else:
        phase_extract(groups, prompts, results_dir, cache_dir,
                      args.max_tokens, args.token, save_embeddings=args.save_embeddings)
        import subprocess
        print("\nIniciando Fase 2...")
        cmd = [sys.executable, __file__, "--analyze-only", "--groups"] + groups
        if args.subset:
            cmd.append("--subset")
        sys.exit(subprocess.run(cmd, check=True).returncode)
