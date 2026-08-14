"""
H4_rev — Perturbation Experiment (MIA + SIA, Gemma 4 31B-it)
=============================================================
Inyecta ruido gaussiano en hidden states de capas intermedias durante la generación
y mide si el grupo axis recupera su firma geométrica.

Corre MIA y SIA usando el MISMO modelo (google/gemma-4-31B-it) — una sola descarga.

Discriminante:
  Resiliencia post-perturbación dependiente de contenido → τ_axis < τ_vanilla, SampEn delta > 0 (exploración activa)
  Resonancia pura                                         → τ_axis ≈ τ_vanilla

Uso:
  python run_perturbation.py --exp mia --layer L30    # MIA en capa 50%
  python run_perturbation.py --exp sia --layer L30    # SIA en capa 50%
  python run_perturbation.py --exp both --layer L30   # ambos secuencialmente
  python run_perturbation.py --exp mia --token hf_xxx

Requisito previo:
  python ../mia/run_exp.py --subset --save-embeddings
  python ../sia/run_exp.py --subset --save-embeddings
  → actualizar mean_velocity en este script con los valores reales

Resultados en:
  results/perturbation_{exp}_{layer}_{sigma}/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from perturbation_extractor import PerturbationExtractor, compute_sigma
from recovery_analyzer import compute_recovery, aggregate_recovery, compare_groups

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


# ═══════════════════════════════════════════════════════════════════════════════
# Gemma 4 31B-it — Modelo base (compartido MIA + SIA)
# ═══════════════════════════════════════════════════════════════════════════════
#
# hidden_dim=5376, n_layers=60. BF16 ~62 GB + activaciones → A100 80GB / H100.
# mean_velocity: ACTUALIZAR tras correr run_exp.py base (ver _traj_metrics.json)

MODEL_BASE = {
    "hf_id":            "google/gemma-4-31B-it",
    "nv_path":          "/workspace/models/gemma-4-31B-it",
    "hidden_dim":       5376,
    "n_layers":         60,
    "min_vram_gb":      65,
    "max_input_tokens": 9500,
}

# Capas: L30=50% (umbral topológico), L40=67% (pico de concentración geométrica)
LAYER_SETS = {
    "L30":     [29],
    "L40":     [39],
    "L30_L40": [29, 39],
}

# ═══════════════════════════════════════════════════════════════════════════════
# Experimento MIA (~5K chars)
# ═══════════════════════════════════════════════════════════════════════════════

MIA_CONFIG = {
    "mean_velocity": 220.0,  # PLACEHOLDER — actualizar tras run_exp.py MIA
    "groups": {
        "axis":           {"system_prompt_path": "../mia/prompts/axis.dna"},
        "generic_long":   {"system_prompt_path": "../mia/prompts/generic_long.txt"},
        "generic_short":  {"system_prompt_path": "../mia/prompts/generic_short.txt"},
        "vanilla":        {"system_prompt": "You are a helpful assistant."},
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# Experimento SIA (~12K chars VEX)
# ═══════════════════════════════════════════════════════════════════════════════

SIA_CONFIG = {
    "mean_velocity": 240.0,  # PLACEHOLDER — actualizar tras run_exp.py SIA
    "groups": {
        "axis":           {"system_prompt_path": "../sia/prompts/axis.dna"},
        "generic_long":   {"system_prompt_path": "../sia/prompts/generic_long.txt"},
        "generic_short":  {"system_prompt_path": "../sia/prompts/generic_short.txt"},
        "vanilla":        {"system_prompt": "You are a helpful assistant."},
    },
}

# Timesteps de inyección
T_INJ_VALUES = [50, 128, 200]

# 20 prompts prioritarios
PRIORITY_SUBSET = [1, 3, 6, 10, 14, 21, 23, 27, 31, 39,
                   41, 45, 51, 59, 61, 65, 71, 79, 91, 98]

MAX_NEW_TOKENS = 256


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_prompts(prompts_file: str, subset=None):
    with open(prompts_file) as f:
        data = json.load(f)
    prompts = []
    for item in data:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict):
            q = item.get("question") or item.get("text") or item.get("prompt") or ""
            prompts.append(q)
    if subset:
        prompts = [prompts[i - 1] for i in subset if 0 < i <= len(prompts)]
    return prompts


def load_system_prompt(cfg: Dict, base_dir: Path) -> str:
    if "system_prompt" in cfg:
        return cfg["system_prompt"]
    path = base_dir / cfg["system_prompt_path"]
    if not path.exists():
        path = Path(__file__).parent.parent / cfg["system_prompt_path"].lstrip("../")
    with open(path) as f:
        return f.read().strip()


def resolve_model_path(hf_token=None):
    nv = MODEL_BASE["nv_path"]
    if os.path.isdir(nv):
        print(f"  Modelo en Network Volume: {nv}")
        return nv
    print(f"  Usando HF hub: {MODEL_BASE['hf_id']}")
    return MODEL_BASE["hf_id"]


# ── Core ─────────────────────────────────────────────────────────────────────

def run_experiment(exp_name: str, exp_config: dict, args):
    """Ejecuta H4_rev completo para un experimento (MIA o SIA)."""
    groups_config = exp_config["groups"]
    mean_velocity = exp_config["mean_velocity"]
    base_dir = Path(__file__).parent.parent

    layer_key = args.layer
    if layer_key not in LAYER_SETS:
        raise ValueError(f"--layer '{layer_key}' no válido. Opciones: {list(LAYER_SETS.keys())}")
    layer_indices = LAYER_SETS[layer_key]

    sigma_multipliers = {"small": 0.5, "medium": 1.0, "large": 2.0}
    sigma_val = compute_sigma(mean_velocity, MODEL_BASE["hidden_dim"],
                              k=sigma_multipliers[args.sigma])

    results_dir = (
        Path(__file__).parent / f"results/perturbation_{exp_name}_{layer_key}_{args.sigma}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(__file__).parent / f"cache_{exp_name}_{layer_key}_{args.sigma}"

    print("\n" + "=" * 70)
    print(f"H4_rev — {exp_name.upper()} — Gemma 4 31B-it")
    print(f"  Capas   : {layer_key} → índices {layer_indices}")
    print(f"  Sigma   : {args.sigma} → {sigma_val:.4f} per-dim")
    print(f"  t_inj   : {T_INJ_VALUES}")
    print(f"  Grupos  : {list(groups_config.keys())}")
    print(f"  mean_vel: {mean_velocity:.1f} (placeholder={mean_velocity in (220.0, 240.0)})")
    print("=" * 70 + "\n")

    # Prompts
    if args.prompts:
        prompts_file = Path(args.prompts)
    else:
        candidates = [
            base_dir / "data/prompts.json",
            Path("/workspace/data/prompts.json"),
        ]
        prompts_file = next((p for p in candidates if p.exists()), candidates[0])
    if not prompts_file.exists():
        raise FileNotFoundError(f"prompts.json no encontrado. Usa --prompts.")
    prompts = load_prompts(str(prompts_file), subset=PRIORITY_SUBSET)
    print(f"  {len(prompts)} prompts cargados\n")

    # System prompts
    system_prompts = {g: load_system_prompt(cfg, base_dir)
                      for g, cfg in groups_config.items()}
    for g, sp in system_prompts.items():
        print(f"  [{g}] {len(sp)} chars")
    print()

    # Extractor
    if args.force and cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print("  Caché eliminado (--force)\n")

    model_path = resolve_model_path(args.token)
    extractor = PerturbationExtractor(
        model_path=model_path,
        max_new_tokens=MAX_NEW_TOKENS,
        layer_idx=-1,
        max_input_tokens=MODEL_BASE["max_input_tokens"],
        cache_dir=str(cache_dir),
        device="cuda",
        min_vram_gb=MODEL_BASE["min_vram_gb"],
        hf_token=args.token,
    )
    n_layers, hidden_dim = extractor.load_model()
    print(f"\n  hidden_dim={hidden_dim} (config={MODEL_BASE['hidden_dim']}) | "
          f"capas={n_layers}\n")

    # Loop principal
    details: Dict = {g: {t: [] for t in T_INJ_VALUES} for g in groups_config}
    total_runs = len(prompts) * len(groups_config) * (1 + len(T_INJ_VALUES))
    run_count = 0
    t_start = time.time()

    for pi, prompt in enumerate(prompts):
        print(f"\n[Prompt {pi+1}/{len(prompts)}] {prompt[:70]}...")

        for group, sp in system_prompts.items():
            t0 = time.time()
            baseline_traj = extractor.extract_baseline(prompt, sp, group=group)
            run_count += 1
            elapsed = time.time() - t0
            cached = " (cached)" if elapsed < 0.5 else ""
            print(f"    {group:20s} baseline  T={baseline_traj.n_steps:3d}  "
                  f"{elapsed:.1f}s{cached}")

            for t_inj in T_INJ_VALUES:
                if t_inj >= baseline_traj.n_steps - 5:
                    details[group][t_inj].append(
                        {"status": "skipped_short_trajectory",
                         "t_inj": t_inj, "T_baseline": baseline_traj.n_steps})
                    print(f"    {group:20s} t_inj={t_inj:3d}  SKIP")
                    continue

                t0 = time.time()
                perturb_traj = extractor.extract_perturbed(
                    prompt, sp, group=group, t_inj=t_inj,
                    sigma=sigma_val, layer_indices=layer_indices)
                run_count += 1
                elapsed = time.time() - t0
                cached = " (cached)" if elapsed < 0.5 else ""

                rec = compute_recovery(baseline_traj.embeddings,
                                       perturb_traj.embeddings, t_inj=t_inj)
                details[group][t_inj].append(rec)

                tau_str = (f"τ={rec['tau_tokens']}"
                           if rec.get("tau_tokens") is not None else "τ=∞")
                print(f"    {group:20s} t_inj={t_inj:3d}  "
                      f"T={perturb_traj.n_steps:3d}  "
                      f"disp={rec.get('displacement_l2', 0):.1f}  "
                      f"{tau_str}  gap={rec.get('recovery_gap', 0):.3f}  "
                      f"{elapsed:.1f}s{cached}")

        done_frac = run_count / total_runs
        elapsed_total = time.time() - t_start
        if done_frac > 0:
            eta = elapsed_total / done_frac * (1 - done_frac)
            print(f"  → {run_count}/{total_runs} | ETA: {eta/60:.1f} min")

    # Agregación
    print("\n" + "=" * 70)
    print(f"ANÁLISIS — {exp_name.upper()}")
    print("=" * 70)

    aggregated = {}
    for group in groups_config:
        aggregated[group] = {}
        for t_inj in T_INJ_VALUES:
            aggregated[group][str(t_inj)] = aggregate_recovery(details[group][t_inj])

    comparison_by_t = {}
    for t_inj in T_INJ_VALUES:
        ga = {g: aggregated[g][str(t_inj)] for g in groups_config}
        comparison_by_t[str(t_inj)] = compare_groups(ga)

    # Tablas de contraste
    def _f(d, k):
        v = d.get(k, {}); return v.get("mean") if isinstance(v, dict) else v
    def _fmt(v):
        return f"{v:.1f}" if v is not None else "N/A"

    # E1: axis vs generic_long (misma longitud → efecto identidad puro)
    print(f"\n── E1: axis vs generic_long (misma longitud, distinto contenido) ──")
    print(f"{'t_inj':>6}  {'τ_axis':>8}  {'τ_glong':>8}  {'Δτ':>10}  "
          f"{'recov_ax':>8}  {'recov_gl':>8}  {'SampEnΔ_ax':>10}  {'SampEnΔ_gl':>10}")
    print("─" * 78)
    for t_inj in T_INJ_VALUES:
        comp = comparison_by_t[str(t_inj)]
        tau = comp.get("tau_tokens", {})
        rec = comp.get("recovery_rate", {})
        sen = comp.get("sampen_delta", {})
        print(f"{t_inj:>6}  {_fmt(_f(tau,'axis')):>8}  {_fmt(_f(tau,'generic_long')):>8}  "
              f"{_fmt(_f(tau,'delta_axis_minus_generic_long')):>10}  "
              f"{_fmt(_f(rec,'axis')):>8}  {_fmt(_f(rec,'generic_long')):>8}  "
              f"{_fmt(_f(sen,'axis')):>10}  {_fmt(_f(sen,'generic_long')):>10}")

    # E2: generic_long vs generic_short (efecto puro de longitud)
    print(f"\n── E2: generic_long vs generic_short (efecto puro de longitud) ──")
    print(f"{'t_inj':>6}  {'τ_glong':>8}  {'τ_gshort':>8}  {'Δτ':>10}  "
          f"{'recov_gl':>8}  {'recov_gs':>8}  {'SampEnΔ_gl':>10}  {'SampEnΔ_gs':>10}")
    print("─" * 78)
    for t_inj in T_INJ_VALUES:
        comp = comparison_by_t[str(t_inj)]
        tau = comp.get("tau_tokens", {})
        rec = comp.get("recovery_rate", {})
        sen = comp.get("sampen_delta", {})
        print(f"{t_inj:>6}  {_fmt(_f(tau,'generic_long')):>8}  {_fmt(_f(tau,'generic_short')):>8}  "
              f"{_fmt(_f(tau,'delta_generic_long_minus_generic_short')):>10}  "
              f"{_fmt(_f(rec,'generic_long')):>8}  {_fmt(_f(rec,'generic_short')):>8}  "
              f"{_fmt(_f(sen,'generic_long')):>10}  {_fmt(_f(sen,'generic_short')):>10}")

    # E3: axis vs vanilla (referencia histórica)
    print(f"\n── E3: axis vs vanilla (referencia) ──")
    print(f"{'t_inj':>6}  {'τ_axis':>8}  {'τ_van':>8}  {'Δτ':>10}  "
          f"{'recov_ax':>8}  {'recov_van':>8}  {'SampEnΔ_ax':>10}  {'SampEnΔ_van':>10}")
    print("─" * 74)
    for t_inj in T_INJ_VALUES:
        comp = comparison_by_t[str(t_inj)]
        tau = comp.get("tau_tokens", {})
        rec = comp.get("recovery_rate", {})
        sen = comp.get("sampen_delta", {})
        print(f"{t_inj:>6}  {_fmt(_f(tau,'axis')):>8}  {_fmt(_f(tau,'vanilla')):>8}  "
              f"{_fmt(_f(tau,'delta_axis_minus_vanilla')):>10}  "
              f"{_fmt(_f(rec,'axis')):>8}  {_fmt(_f(rec,'vanilla')):>8}  "
              f"{_fmt(_f(sen,'axis')):>10}  {_fmt(_f(sen,'vanilla')):>10}")

    # Guardar
    summary = {
        "experiment": f"H4_rev_{exp_name}",
        "model": MODEL_BASE["hf_id"],
        "layer_key": layer_key, "layer_indices": layer_indices,
        "sigma_key": args.sigma, "sigma_value": sigma_val,
        "t_inj_values": T_INJ_VALUES, "n_prompts": len(prompts),
        "groups": list(groups_config.keys()),
        "aggregated": aggregated, "comparison_by_t_inj": comparison_by_t,
    }
    details_out = {
        "experiment": f"H4_rev_{exp_name}",
        "layer_key": layer_key, "sigma_value": sigma_val,
        "details": {g: {str(t): lst for t, lst in inner.items()}
                    for g, inner in details.items()},
    }

    def _json_safe(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"No serializable: {type(obj)}")

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_safe)
    with open(results_dir / "details.json", "w") as f:
        json.dump(details_out, f, indent=2, default=_json_safe)

    print(f"\n  → {results_dir}/")
    print(f"  Tiempo: {(time.time()-t_start)/60:.1f} min")
    return summary


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="H4_rev — Gemma 4 31B-it (MIA + SIA)"
    )
    parser.add_argument("--exp", choices=["mia", "sia", "both"], default="both",
                        help="Experimento a ejecutar (default: both)")
    parser.add_argument("--layer", default="L30",
                        help="Capa: L30 (50% gate), L40 (67% pico geométrico), L30_L40")
    parser.add_argument("--sigma", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--token", default=None, help="HF token")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prompts", default=None)
    args = parser.parse_args()

    experiments = []
    if args.exp in ("mia", "both"):
        experiments.append(("mia", MIA_CONFIG))
    if args.exp in ("sia", "both"):
        experiments.append(("sia", SIA_CONFIG))

    for exp_name, exp_config in experiments:
        run_experiment(exp_name, exp_config, args)

    print("\n" + "=" * 70)
    print("FIN H4_rev — Gemma 4 31B-it")
    print("=" * 70)
