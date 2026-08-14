"""
Recovery Analyzer — H4_rev

Computa métricas de recuperación comparando trayectorias baseline vs perturbadas.

Pregunta central: ¿El grupo axis recupera su firma geométrica más rápido que
vanilla/generic tras una perturbación en hidden states?

  - Resiliencia post-perturbación dependiente de contenido → τ_axis << τ_vanilla (recuperación activa)
  - Resonancia pura                                         → τ_axis ≈ τ_vanilla  (sin ventaja dinámica)

Métricas:
  displacement_l2    : ||h_perturbed[t_inj] - h_baseline[t_inj]||₂
                       (cuánto desplazó el ruido el estado en el punto de inyección)
  centroid_cos_post  : cos(centroid_post_perturbed, centroid_baseline_full)
                       (¿regresa la trayectoria al mismo "centro de gravedad"?)
  centroid_cos_base  : mismo cálculo para baseline (referencia)
  recovery_gap       : centroid_cos_base - centroid_cos_post (≈0 → recuperado)
  tau_tokens         : primer W tal que rolling_cos(W) ≥ τ_threshold × centroid_cos_base
  sampen_post        : SampEn de ||h_perturbed[t_inj:]||  (regularidad post-inyección)
  sampen_base        : SampEn de ||h_baseline[t_inj:]||   (referencia)
  sampen_delta       : sampen_post - sampen_base (>0 → perturbación aumentó entropía)
  w1_norms_post      : W₁ entre ||h_perturbed[t_inj:]|| y ||h_baseline[t_inj:]||
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ── Constantes ────────────────────────────────────────────────────────────────

TAU_THRESHOLD = 0.95      # cos debe llegar a ≥95% del baseline para considerar recuperado
TAU_MIN_WINDOW = 5        # ventana mínima en tokens para estabilizar el cálculo
TAU_STEP = 1              # granularidad de búsqueda de τ (tokens)


# ── Función principal ─────────────────────────────────────────────────────────

def compute_recovery(
    baseline: np.ndarray,    # (T_b, D)
    perturbed: np.ndarray,   # (T_p, D)
    t_inj: int,
    tau_threshold: float = TAU_THRESHOLD,
) -> Dict:
    """
    Computa métricas de recuperación para un par (baseline, perturbed).

    Parámetros
    ----------
    baseline   : trayectoria sin perturbación  (T_b, D)
    perturbed  : trayectoria con perturbación  (T_p, D)
    t_inj      : paso en que se inyectó el ruido
    tau_threshold : fracción del coseno baseline para declarar recuperación

    Retorna
    -------
    dict con todas las métricas descritas en el módulo docstring.
    """
    T_b = len(baseline)
    T_p = len(perturbed)

    # Validar que hay suficientes tokens post-inyección
    post_b = baseline[t_inj:]  # puede ser vacío si t_inj >= T_b
    post_p = perturbed[t_inj:] if t_inj < T_p else np.zeros((0, baseline.shape[1]))

    min_post = min(len(post_b), len(post_p))

    if min_post < TAU_MIN_WINDOW:
        return _empty_result(t_inj, "insufficient_post_tokens")

    # ── Centroide de referencia (trayectoria baseline completa) ───────────────
    centroid_full = baseline.mean(axis=0)                # (D,)
    norm_centroid = np.linalg.norm(centroid_full) + 1e-8

    # ── Desplazamiento en t_inj ───────────────────────────────────────────────
    if t_inj < T_b and t_inj < T_p:
        displacement_l2 = float(
            np.linalg.norm(perturbed[t_inj] - baseline[t_inj])
        )
    else:
        displacement_l2 = None

    # ── Cosenos de centroides post-inyección ──────────────────────────────────
    centroid_post_p = post_p[:min_post].mean(axis=0)
    centroid_post_b = post_b[:min_post].mean(axis=0)

    cos_post_p = float(
        np.dot(centroid_post_p, centroid_full)
        / (np.linalg.norm(centroid_post_p) + 1e-8)
        / norm_centroid
    )
    cos_post_b = float(
        np.dot(centroid_post_b, centroid_full)
        / (np.linalg.norm(centroid_post_b) + 1e-8)
        / norm_centroid
    )

    recovery_gap = cos_post_b - cos_post_p   # ≈0 → recuperado

    # ── Búsqueda de τ (ventana rolling de cosenos) ───────────────────────────
    tau_tokens = _find_tau(
        post_p, centroid_full, cos_post_b, tau_threshold
    )

    # ── SampEn post-inyección ─────────────────────────────────────────────────
    norms_p = np.linalg.norm(post_p[:min_post], axis=1)
    norms_b = np.linalg.norm(post_b[:min_post], axis=1)
    sampen_post = _sample_entropy(norms_p)
    sampen_base = _sample_entropy(norms_b)
    sampen_delta = (sampen_post - sampen_base
                    if sampen_post is not None and sampen_base is not None
                    else None)

    # ── W₁ entre distribuciones de normas post-inyección ─────────────────────
    w1_norms_post = _wasserstein1(norms_p, norms_b)

    return {
        "t_inj": t_inj,
        "T_baseline": T_b,
        "T_perturbed": T_p,
        "post_tokens_used": min_post,
        "displacement_l2": displacement_l2,
        "centroid_cos_post_perturbed": cos_post_p,
        "centroid_cos_post_baseline": cos_post_b,
        "recovery_gap": float(recovery_gap),      # ≈0 → recuperado; >0 → no recuperado
        "tau_tokens": tau_tokens,                  # None si no recuperó
        "sampen_post": sampen_post,
        "sampen_base": sampen_base,
        "sampen_delta": sampen_delta,             # >0 → perturbación aumentó entropía
        "w1_norms_post": w1_norms_post,
        "status": "ok",
    }


def _find_tau(
    post_perturbed: np.ndarray,   # (S, D) — segmento post-inyección perturbed
    centroid_full: np.ndarray,    # (D,) — centroide de referencia
    cos_reference: float,         # coseno baseline como referencia
    threshold: float,
) -> Optional[int]:
    """
    Busca el menor W tal que cos(mean(post_perturbed[:W]), centroid_full)
    ≥ threshold × cos_reference.

    Retorna W en tokens, o None si no se alcanza en S tokens.
    """
    if cos_reference <= 0 or len(post_perturbed) < TAU_MIN_WINDOW:
        return None

    target = threshold * cos_reference
    norm_c = np.linalg.norm(centroid_full) + 1e-8
    cumulative = np.zeros_like(centroid_full)

    for w in range(1, len(post_perturbed) + 1):
        cumulative += post_perturbed[w - 1]
        centroid_w = cumulative / w
        cos_w = (np.dot(centroid_w, centroid_full)
                 / (np.linalg.norm(centroid_w) + 1e-8)
                 / norm_c)
        if cos_w >= target and w >= TAU_MIN_WINDOW:
            return w

    return None   # no recuperó dentro del horizonte


# ── Agregación cross-prompts ──────────────────────────────────────────────────

def aggregate_recovery(results: List[Dict]) -> Dict:
    """
    Agrega métricas de recuperación de múltiples prompts para un grupo.

    Retorna estadísticas (mean, std, n) por métrica + tasa de recuperación.
    """
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return {"n": 0, "status": "no_results"}

    def _stats(key: str) -> Dict:
        vals = [r[key] for r in ok if r.get(key) is not None]
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "n":    len(vals),
        }

    taus = [r["tau_tokens"] for r in ok if r["tau_tokens"] is not None]
    recovery_rate = len(taus) / len(ok) if ok else 0.0

    return {
        "n_total": len(results),
        "n_ok": len(ok),
        "recovery_rate": float(recovery_rate),       # fracción que recuperó antes de EOS
        "tau_tokens": _stats("tau_tokens"),
        "displacement_l2": _stats("displacement_l2"),
        "recovery_gap": _stats("recovery_gap"),
        "centroid_cos_post_perturbed": _stats("centroid_cos_post_perturbed"),
        "centroid_cos_post_baseline": _stats("centroid_cos_post_baseline"),
        "sampen_delta": _stats("sampen_delta"),
        "sampen_post": _stats("sampen_post"),
        "sampen_base": _stats("sampen_base"),
        "w1_norms_post": _stats("w1_norms_post"),
    }


def compare_groups(group_agg: Dict[str, Dict]) -> Dict:
    """
    Compara métricas de recuperación entre grupos.
    Diseño 4 condiciones: axis, generic_long, generic_short, vanilla.

    Deltas clave:
      E1: axis vs generic_long (misma longitud → efecto identidad puro)
      E2: generic_long vs generic_short (efecto puro de longitud)
      E3: axis vs vanilla (referencia histórica)
    """
    comparison = {}

    keys_to_compare = [
        "tau_tokens", "recovery_rate", "recovery_gap",
        "sampen_delta", "w1_norms_post", "displacement_l2",
    ]

    for key in keys_to_compare:
        row: Dict = {}
        for gname, gagg in group_agg.items():
            if key == "recovery_rate":
                row[gname] = gagg.get("recovery_rate")
            else:
                v = gagg.get(key, {})
                row[gname] = v.get("mean") if isinstance(v, dict) else v

        # E1: axis vs generic_long (identity effect, length-controlled)
        ax = row.get("axis")
        gl = row.get("generic_long")
        if ax is not None and gl is not None:
            row["delta_axis_minus_generic_long"] = ax - gl
        else:
            row["delta_axis_minus_generic_long"] = None

        # E2: generic_long vs generic_short (pure length effect)
        gs = row.get("generic_short")
        if gl is not None and gs is not None:
            row["delta_generic_long_minus_generic_short"] = gl - gs
        else:
            row["delta_generic_long_minus_generic_short"] = None

        # E3: axis vs vanilla (historical reference)
        van = row.get("vanilla")
        if ax is not None and van is not None:
            row["delta_axis_minus_vanilla"] = ax - van
        else:
            row["delta_axis_minus_vanilla"] = None

        comparison[key] = row

    return comparison


# ── Métricas internas ─────────────────────────────────────────────────────────

def _sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> Optional[float]:
    """SampEn sobre serie 1D (normas de hidden states)."""
    if x is None or len(x) < m + 2:
        return None
    std = x.std()
    if std < 1e-10:
        return 0.0
    x = (x - x.mean()) / std

    N = len(x)
    B, A = 0, 0
    for i in range(N - m - 1):
        for j in range(i + 1, N - m):
            if max(abs(x[i + k] - x[j + k]) for k in range(m)) < r:
                B += 1
                if max(abs(x[i + k] - x[j + k]) for k in range(m + 1)) < r:
                    A += 1
    if B == 0:
        return 10.0
    return float(-np.log((A + 1e-8) / (B + 1e-8)))


def _wasserstein1(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """W₁ entre dos distribuciones 1D (normas) vía diferencia de CDF empíricas."""
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return None
    # Interpolar al mismo tamaño
    n = max(len(a), len(b))
    a_s = np.sort(a)
    b_s = np.sort(b)
    a_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a_s)), a_s)
    b_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(b_s)), b_s)
    return float(np.abs(a_interp - b_interp).mean())


def _empty_result(t_inj: int, reason: str) -> Dict:
    return {
        "t_inj": t_inj,
        "status": reason,
        "tau_tokens": None,
        "displacement_l2": None,
        "recovery_gap": None,
        "centroid_cos_post_perturbed": None,
        "centroid_cos_post_baseline": None,
        "sampen_post": None,
        "sampen_base": None,
        "sampen_delta": None,
        "w1_norms_post": None,
    }
