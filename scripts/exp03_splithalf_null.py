#!/usr/bin/env python3
"""
Exp. 0.3 (GUIA_EXPERIMENTAL_v3.md) — piso de ruido split-half, aplicado a
W1_angular y W1_euclidean (los dos componentes de r, Exp. 0.2 E2) para
poder normalizarlos antes de tomar su cociente.

Adaptacion respecto del texto literal de la guia: la guia describe dividir
las trayectorias de UNA condicion en dos mitades al azar (n/2 vs n/2). Eso
no es directamente aplicable aqui sin sesgo: W1 entre dos muestras finitas
de la MISMA distribucion depende del tamaño de muestra (mas puntos ->
match mas denso -> W1 tipicamente menor), y el numero que r() compara
(W1(axis, vanilla)) usa n=20 vs n=20. Un null de 10 vs 10 subestimaria el
verdadero piso de ruido a n=20 y sesgaria "cuantos multiplos del piso es
lo observado" hacia arriba.

En su lugar: null de permutacion con tamaño de muestra IGUAL al de la
comparacion real. Se agrupan axis U vanilla (40 puntos), se sortean 1000
particiones aleatorias en dos subconjuntos de 20 (sin reposicion, sin
respetar la etiqueta real de grupo) y se calcula W1_euclidean/W1_angular
entre cada par de subconjuntos. Bajo la hipotesis nula ("no hay diferencia
real entre axis y vanilla"), las etiquetas son intercambiables y esta es
exactamente la distribucion nula correcta al mismo n que el estadistico
observado -- mismo principio que un permutation test, aplicado a un
estadistico W1 en vez de a una diferencia de medias.

Uso:
    python3 exp03_splithalf_null.py --npz-dir <dir> --label base
"""
import argparse
import json
from pathlib import Path

import numpy as np

from exp02_baselines_norm_r import (
    SEED,
    angular_distance_matrix,
    euclidean_distance_matrix,
    load_v1,
    w1_exact_balanced,
)

N_NULL = 1000


def splithalf_null(pool: np.ndarray, cost_fn, rng: np.random.RandomState, n_null=N_NULL):
    n_total = pool.shape[0]
    n_half = n_total // 2
    vals = np.empty(n_null)
    idx = np.arange(n_total)
    for i in range(n_null):
        rng.shuffle(idx)
        A = pool[idx[:n_half]]
        B = pool[idx[n_half:2 * n_half]]
        vals[i] = w1_exact_balanced(A, B, cost_fn)
    return vals


def run(npz_dir: Path, label: str) -> dict:
    axis = load_v1(npz_dir, "axis")
    vanilla = load_v1(npz_dir, "vanilla")
    pool = np.vstack([axis, vanilla])  # 40 puntos, 20+20

    obs_eucl = w1_exact_balanced(axis, vanilla, euclidean_distance_matrix)
    obs_ang = w1_exact_balanced(axis, vanilla, angular_distance_matrix)

    null_eucl = splithalf_null(pool, euclidean_distance_matrix, np.random.RandomState(SEED))
    null_ang = splithalf_null(pool, angular_distance_matrix, np.random.RandomState(SEED + 1))

    def summarize(obs, null):
        median = float(np.median(null))
        pct = float((null < obs).mean() * 100)
        return {
            "observed": float(obs),
            "null_median": median,
            "null_mean": float(np.mean(null)),
            "null_std": float(np.std(null)),
            "percentile_of_null": pct,
            "multiple_of_null_median": float(obs / median) if median > 0 else None,
        }

    eucl_summary = summarize(obs_eucl, null_eucl)
    ang_summary = summarize(obs_ang, null_ang)

    r_raw = obs_ang / obs_eucl
    r_normalized = ang_summary["multiple_of_null_median"] / eucl_summary["multiple_of_null_median"]

    out = {
        "label": label,
        "n_axis": int(len(axis)),
        "n_vanilla": int(len(vanilla)),
        "n_null_splits": N_NULL,
        "w1_euclidean": eucl_summary,
        "w1_angular": ang_summary,
        "r_raw_unnormalized": r_raw,
        "r_normalized_by_own_null_median": r_normalized,
        "note": (
            "r_normalized = (W1_eucl_obs / mediana_null_eucl) dividido por "
            "(W1_ang_obs / mediana_null_ang) -- ambos numerador y "
            "denominador son adimensionales (multiplos de su propio piso de "
            "ruido), resuelve el desajuste de unidades radianes-vs-euclidiano "
            "crudo de exp02_baselines_norm_r.py."
        ),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True, type=str)
    ap.add_argument("--label", required=True, type=str)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    result = run(Path(args.npz_dir), args.label)
    out_path = Path(args.out) if args.out else Path(__file__).parent.parent / "reports" / f"exp03_{args.label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("\nsaved:", out_path)


if __name__ == "__main__":
    main()
