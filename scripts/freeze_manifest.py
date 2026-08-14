#!/usr/bin/env python3
"""
Exp. 0.5 — Higiene de reproducibilidad: congela artefactos con SHA-256.

Genera MANIFEST.sha256 con hashes de:
  1. Prompts (dataset + axis templates + controles + vanilla string)
  2. Hidden states extraídos (.npz embeddings)
  3. Resultados de análisis (results.json, _graphs/_dims/_traj_metrics, responses)
  4. Resultados H4_rev (summary/details)
  5. Código del pipeline (scripts del experimento)

Formato: <sha256>  <ruta-relativa-al-TGZ>

Uso:
  python freeze_manifest.py [--include-code]
"""

import hashlib
import json
import sys
from pathlib import Path

BASE = Path(__file__).parent
RESULTS = BASE / "results_local"

# Vanilla string — el cuarto "prompt de sistema"
VANILLA_TEXT = "You are a helpful assistant."

# Rutas REALES relativas a BASE (para que sha256sum -c funcione directo)
PROMPT_ARTIFACTS = [
    ("data/prompts.json",            BASE / "data/prompts.json"),
    ("mia/prompts/axis.dna",         BASE / "mia/prompts/axis.dna"),
    ("mia/prompts/generic_long.txt", BASE / "mia/prompts/generic_long.txt"),
    ("mia/prompts/generic_short.txt",BASE / "mia/prompts/generic_short.txt"),
    ("sia/prompts/axis.dna",         BASE / "sia/prompts/axis.dna"),
    ("sia/prompts/generic_long.txt", BASE / "sia/prompts/generic_long.txt"),
    ("sia/prompts/generic_short.txt",BASE / "sia/prompts/generic_short.txt"),
    ("prompts_vanilla.string",       None),  # texto inline — archivo hash-only
]

CODE_ARTIFACTS = [
    "mia/run_exp.py",
    "sia/run_exp.py",
    "perturbation/run_perturbation.py",
    "perturbation/perturbation_extractor.py",
    "perturbation/recovery_analyzer.py",
    "verify_tokens.py",
    "calibrate_sigma.py",
    "shared/hidden_state_extractor.py",
    "shared/graph_builder.py",
    "shared/curvature_analyzer.py",
    "shared/dimensionality_analyzer.py",
    "shared/statistical_tests.py",
    "shared/trajectory_metrics.py",
    "shared/analyze_dynamics.py",  # renombrado desde analyze_autopoiesis.py en este repo curado
    "shared/visualization.py",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    include_code = "--include-code" in sys.argv
    lines = []
    lines.append(f"# MANIFEST SHA-256 — Exp. 0.5 reproducibilidad")
    lines.append(f"# Generado: 2026-08-12 | Modelo: google/gemma-4-31B-it | GPU: RTX PRO 6000 (96GB)")
    lines.append(f"# Vanilla system prompt: '{VANILLA_TEXT}'")
    lines.append("")

    # ── 1. Prompts ──────────────────────────────────────────────────────────
    lines.append("# === 1. PROMPTS ===")
    for label, path in PROMPT_ARTIFACTS:
        if path is None:
            lines.append(f"{sha256_text(VANILLA_TEXT)}  {label}")
        elif path.exists():
            lines.append(f"{sha256_file(path)}  {label}")
        else:
            lines.append(f"MISSING  {label}")
    lines.append("")

    # ── 2. Hidden states (.npz) ─────────────────────────────────────────────
    lines.append("# === 2. HIDDEN STATES (embeddings .npz) ===")
    for exp in ["mia", "sia"]:
        npz_dir = RESULTS / exp
        for npz in sorted(npz_dir.glob("*_embeddings.npz")):
            rel = f"results_local/{exp}/{npz.name}"
            lines.append(f"{sha256_file(npz)}  {rel}")
    lines.append("")

    # ── 3. Resultados base ───────────────────────────────────────────────────
    lines.append("# === 3. RESULTADOS BASE (JSON) ===")
    for exp in ["mia", "sia"]:
        for jf in sorted((RESULTS / exp).glob("*.json")):
            rel = f"results_local/{exp}/{jf.name}"
            lines.append(f"{sha256_file(jf)}  {rel}")
    lines.append("")

    # ── 4. Resultados H4_rev ─────────────────────────────────────────────────
    lines.append("# === 4. RESULTADOS H4_REV ===")
    for pdir in sorted(RESULTS.glob("perturbation_*")):
        for jf in sorted(pdir.glob("*.json")):
            rel = f"results_local/{pdir.name}/{jf.name}"
            lines.append(f"{sha256_file(jf)}  {rel}")
    lines.append("")

    # ── 5. Código ────────────────────────────────────────────────────────────
    if include_code:
        lines.append("# === 5. CÓDIGO DEL PIPELINE ===")
        for rel in CODE_ARTIFACTS:
            p = BASE / rel
            if p.exists():
                lines.append(f"{sha256_file(p)}  {rel}")
        lines.append("")

    # ── Resumen de seeds (metadata) ──────────────────────────────────────────
    lines.append("# === SEEDS ===")
    seeds = {
        "permutation_test": 42,
        "mle_dimension": 42,
        "pca_random_state": 42,
        "decoding": "greedy (argmax, determinista)",
        "perturbation_noise": "NO SEMBRADO — por diseño; trayectorias congeladas vía este manifest",
        "correlation_dim": "no usado en este pipeline (solo mle)",
    }
    for k, v in seeds.items():
        lines.append(f"# {k}: {v}")

    out = BASE / "MANIFEST.sha256"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Manifest: {out} ({len(lines)} líneas, {len(lines)-6} artefactos)")
    print(f"Verificar con: sha256sum -c <(grep -v '^#' MANIFEST.sha256)")


if __name__ == "__main__":
    main()
