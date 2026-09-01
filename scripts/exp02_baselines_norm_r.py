#!/usr/bin/env python3
"""
Exp. 0.2 (GUIA_EXPERIMENTAL_v3.md) — baselines simples vs los dos
estadísticos primarios de la v3: Delta_norm (indice de norma, E1) y
r = W1_angular / W1_euclidean (diagnostico direccion vs magnitud, E2).

No reusa el pipeline de curvatura (Delta_kappa/W1 de Forman-Ricci) — esa
metrica ya fue auditada por separado (REPORTE_FASE0.md, CURVATURE_SELF_
AUDIT_REPORT.md, ambos en LSGOT_v4) y no es la que este experimento evalua.
Aqui se corren, sobre v1 (hidden state del primer paso de generacion,
embeddings[:, 0, :]), con el MISMO protocolo de permutacion a nivel de
trayectoria para que sea comparable manzana con manzana:

  - Delta_norm(M)  = mean||v1||_axis - mean||v1||_generic_long   (E1)
  - r(M)           = W1_angular(axis, vanilla) / W1_euclidean(axis, vanilla)  (E2)
  - Baselines, axis vs generic_long, sobre v1:
      * distancia entre centroides
      * MMD con kernel RBF (heuristica de mediana para bandwidth)
      * 1 - CKA lineal (mismos 20 prompts en ambas condiciones -> CKA
        pareado, no pooled)
      * AUC de probe lineal, CV agrupada por prompt (GroupKFold)
      * Mann-Whitney sobre ||v1|| (el propio Delta_norm, como baseline)

Todos los p-valores de deteccion (excepto Mann-Whitney, que es exacto) usan
permutation test de la diferencia de medias, n_permutations=1000, seed=42
-- misma convencion que el resto del corpus (REPRODUCIBILITY.md Sec.3).

r(M) no es un detector de separacion (eso ya lo prueban los baselines de
arriba) sino un diagnostico de DONDE vive la separacion ya establecida
-- se reporta con un bootstrap CI (resampleo pareado de prompts, B=2000,
seed=42), no con un p-valor de deteccion.

Uso:
    python3 exp02_baselines_norm_r.py --base-dir <axis-vanilla-glong npz dir> --label base
    python3 exp02_baselines_norm_r.py --base-dir <...> --label instruct
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score

SEED = 42
N_PERM = 1000
N_BOOT = 2000


def load_v1(npz_dir: Path, group: str) -> np.ndarray:
    d = np.load(npz_dir / f"{group}_embeddings.npz", allow_pickle=True)
    emb = d["embeddings"].astype(np.float64)  # (N, T, D)
    return emb[:, 0, :]  # v1: hidden state del primer paso de generacion


def perm_test_mean_diff(a: np.ndarray, b: np.ndarray, rng: np.random.RandomState,
                         n_perm: int = N_PERM):
    """Permutation test de dos colas sobre la diferencia de medias escalares."""
    obs = float(np.mean(a) - np.mean(b))
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = np.mean(pooled[:n_a]) - np.mean(pooled[n_a:])
        if abs(diff) >= abs(obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled_std = np.sqrt((sa**2 + sb**2) / 2)
    d = obs / pooled_std if pooled_std > 0 else 0.0
    return obs, p, d


def w1_exact_balanced(X: np.ndarray, Y: np.ndarray, cost_fn) -> float:
    """W1 exacto para dos nubes de puntos con el mismo n y pesos uniformes:
    se reduce al problema de asignacion lineal (Hungarian), resultado
    identico a resolver el LP de transporte optimo discreto balanceado."""
    assert X.shape[0] == Y.shape[0], "W1 exacto balanceado requiere n_X == n_Y"
    C = cost_fn(X, Y)
    row, col = linear_sum_assignment(C)
    return float(C[row, col].mean())


def angular_distance_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    cos = np.clip(Xn @ Yn.T, -1.0, 1.0)
    return np.arccos(cos)  # radianes, metrica propia en la esfera


def euclidean_distance_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return cdist(X, Y, metric="euclidean")


def compute_r(axis: np.ndarray, vanilla: np.ndarray) -> float:
    w1_eucl = w1_exact_balanced(axis, vanilla, euclidean_distance_matrix)
    w1_ang = w1_exact_balanced(axis, vanilla, angular_distance_matrix)
    return w1_ang / w1_eucl, w1_ang, w1_eucl


def bootstrap_r_ci(axis: np.ndarray, vanilla: np.ndarray, rng: np.random.RandomState,
                    n_boot: int = N_BOOT):
    n = axis.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)  # resampleo pareado (mismo prompt en axis/vanilla)
        r, _, _ = compute_r(axis[idx], vanilla[idx])
        vals.append(r)
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(np.std(vals))


def mmd_rbf(X: np.ndarray, Y: np.ndarray) -> float:
    """MMD^2 (estimador sesgado), kernel RBF, bandwidth = heuristica de mediana
    sobre la union de X e Y."""
    Z = np.vstack([X, Y])
    d2 = cdist(Z, Z, metric="sqeuclidean")
    med = np.median(d2[d2 > 0])
    gamma = 1.0 / (2 * med) if med > 0 else 1.0

    def k(A, B):
        return np.exp(-gamma * cdist(A, B, metric="sqeuclidean"))

    Kxx = k(X, X)
    Kyy = k(Y, Y)
    Kxy = k(X, Y)
    m, n = len(X), len(Y)
    mmd2 = Kxx.sum() / (m * m) + Kyy.sum() / (n * n) - 2 * Kxy.sum() / (m * n)
    return float(mmd2)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """CKA lineal pareada: X e Y evaluadas sobre el MISMO conjunto de
    prompts (misma fila = mismo prompt), como Kornblith et al. 2019."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(Xc.T @ Yc, "fro") ** 2
    hsic_xx = np.linalg.norm(Xc.T @ Xc, "fro") ** 2
    hsic_yy = np.linalg.norm(Yc.T @ Yc, "fro") ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 0 else 0.0


def probe_auc(axis: np.ndarray, vanilla: np.ndarray, rng_seed: int = SEED) -> float:
    """AUC de un probe lineal axis-vs-condicion, CV agrupada por prompt
    (GroupKFold: el mismo prompt nunca aparece en train Y test a la vez)."""
    X = np.vstack([axis, vanilla])
    y = np.concatenate([np.ones(len(axis)), np.zeros(len(vanilla))])
    groups = np.concatenate([np.arange(len(axis)), np.arange(len(vanilla))])
    n_splits = min(5, len(axis))
    gkf = GroupKFold(n_splits=n_splits)
    clf = LogisticRegression(max_iter=2000, random_state=rng_seed)
    scores = cross_val_score(clf, X, y, groups=groups, cv=gkf, scoring="roc_auc")
    return float(scores.mean())


def centroid_distance(X: np.ndarray, Y: np.ndarray) -> float:
    return float(np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0)))


def run(npz_dir: Path, label: str) -> dict:
    rng = np.random.RandomState(SEED)

    axis = load_v1(npz_dir, "axis")
    glong = load_v1(npz_dir, "generic_long")
    vanilla = load_v1(npz_dir, "vanilla")

    norm_axis = np.linalg.norm(axis, axis=1)
    norm_glong = np.linalg.norm(glong, axis=1)

    # --- E1: Delta_norm(axis vs generic_long) ---
    delta_norm, p_norm, d_norm = perm_test_mean_diff(norm_axis, norm_glong, rng)
    u_stat, p_mwu = mannwhitneyu(norm_axis, norm_glong, alternative="two-sided")

    # --- E2: r(axis vs vanilla) ---
    r_val, w1_ang, w1_eucl = compute_r(axis, vanilla)
    r_lo, r_hi, r_std = bootstrap_r_ci(axis, vanilla, np.random.RandomState(SEED))

    # --- Baselines, axis vs generic_long, sobre v1 ---
    # La distancia de centroides y el MMD son escalares únicos por par de
    # grupos, no una distribución por prompt -> se testean vía permutación
    # de las ETIQUETAS de grupo (reasignar filas a axis/generic_long al
    # azar), no vía perm_test_mean_diff (que es para estadísticas por
    # muestra como ||v1||).
    def centroid_perm_test(X, Y, rng, n_perm=N_PERM):
        obs = centroid_distance(X, Y)
        pooled = np.vstack([X, Y])
        n_x = len(X)
        idx = np.arange(len(pooled))
        count = 0
        for _ in range(n_perm):
            rng.shuffle(idx)
            Xp, Yp = pooled[idx[:n_x]], pooled[idx[n_x:]]
            if centroid_distance(Xp, Yp) >= obs:
                count += 1
        p = (count + 1) / (n_perm + 1)
        return obs, p

    def mmd_perm_test(X, Y, rng, n_perm=N_PERM):
        obs = mmd_rbf(X, Y)
        pooled = np.vstack([X, Y])
        n_x = len(X)
        idx = np.arange(len(pooled))
        count = 0
        for _ in range(n_perm):
            rng.shuffle(idx)
            Xp, Yp = pooled[idx[:n_x]], pooled[idx[n_x:]]
            if mmd_rbf(Xp, Yp) >= obs:
                count += 1
        p = (count + 1) / (n_perm + 1)
        return obs, p

    cdist_obs, p_cdist2 = centroid_perm_test(axis, glong, np.random.RandomState(SEED))
    mmd_obs, p_mmd = mmd_perm_test(axis, glong, np.random.RandomState(SEED))
    cka_val = linear_cka(axis, glong)
    auc_val = probe_auc(axis, glong)

    out = {
        "label": label,
        "n_axis": int(len(axis)),
        "n_generic_long": int(len(glong)),
        "n_vanilla": int(len(vanilla)),
        "E1_delta_norm": {
            "delta_norm": delta_norm,
            "perm_p": p_norm,
            "cohens_d": d_norm,
            "mannwhitney_U": float(u_stat),
            "mannwhitney_p": float(p_mwu),
            "mean_norm_axis": float(norm_axis.mean()),
            "mean_norm_generic_long": float(norm_glong.mean()),
        },
        "E2_r_direction_vs_magnitude": {
            "r": r_val,
            "r_bootstrap_95ci": [r_lo, r_hi],
            "r_bootstrap_std": r_std,
            "w1_angular_rad": w1_ang,
            "w1_euclidean": w1_eucl,
            "note": "axis vs vanilla, no axis vs generic_long -- coincide con la definicion E2 de la guia",
        },
        "baselines_axis_vs_generic_long": {
            "centroid_distance": {"value": cdist_obs, "perm_p": p_cdist2},
            "mmd_rbf": {"value": mmd_obs, "perm_p": p_mmd},
            "linear_cka": {"value": cka_val, "note": "pareado, mismos 20 prompts en ambas condiciones"},
            "linear_probe_auc": {"value": auc_val, "cv": "GroupKFold por prompt"},
            "mannwhitney_norm": {"U": float(u_stat), "p": float(p_mwu)},
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True, type=str)
    ap.add_argument("--label", required=True, type=str)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    result = run(Path(args.npz_dir), args.label)
    out_path = Path(args.out) if args.out else Path(__file__).parent.parent / "reports" / f"exp02_{args.label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("\nsaved:", out_path)


if __name__ == "__main__":
    main()
