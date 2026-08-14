"""
Análisis de Dinámica de Trayectoria — DeepSeek hidden states
==============================================================

Indicadores geométricos de persistencia y consistencia de trayectoria,
comparando:
  - axis          : DeepSeek + MIA DNA (~5K chars)
  - generic_asst  : DeepSeek + generic_assistant_small.txt (~5K chars)
  - vanilla       : DeepSeek + "You are a helpful assistant."

Cada indicador mide una propiedad geométrica distinta de la trayectoria,
sin presuponer dinámica de atractor:
  (a) consistencia entre distintos prompts (similitud inter-trayectoria)
  (b) persistencia del estado inicial durante la generación (alineación intra-trayectoria)
  (c) regularidad temporal (autocorrelación, entropía muestral)
  (d) separabilidad como cluster geométrico (silhouette)
"""

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import json
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = "results/gemma4_4b"
GROUPS = ["axis", "generic_assistant", "vanilla"]
COLORS = {"axis": "🟦", "generic_assistant": "🟩", "vanilla": "🟥"}

# ── Carga ────────────────────────────────────────────────────────────────────

def load_group(name):
    d = np.load(f"{RESULTS_DIR}/{name}_embeddings.npz", allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)   # (N, T, D)
    lengths = d["lengths"].astype(int)          # (N,)
    return emb, lengths


def masked_mean(emb, length):
    """Promedio sobre tokens válidos (hasta length)."""
    return emb[:length].mean(axis=0)


# ── Métrica 1: Prompt-Response Alignment (PRA) ───────────────────────────────

def compute_pra(emb, lengths):
    """
    cosine(emb[0], mean(emb[1:length])) por trayectoria.
    Mide si el estado inicial persiste durante la generación.
    """
    pras = []
    for i, L in enumerate(lengths):
        if L < 2:
            continue
        v0 = emb[i, 0]
        centroid = emb[i, 1:L].mean(axis=0)
        norm0 = np.linalg.norm(v0)
        normc = np.linalg.norm(centroid)
        if norm0 < 1e-8 or normc < 1e-8:
            continue
        pras.append(float(np.dot(v0, centroid) / (norm0 * normc)))
    return np.array(pras)


# ── Métrica 2: Intra-trajectory cosine decay ─────────────────────────────────

def cosine_vs_origin(emb, lengths, n_points=20):
    """
    Para cada trayectoria, mide cos(emb[t], emb[0]) a lo largo del tiempo.
    Si el estado inicial persiste, la curva tiene una asíntota alta.
    Devuelve array (N, n_points) interpolado.
    """
    curves = []
    for i, L in enumerate(lengths):
        if L < 4:
            continue
        v0 = emb[i, 0]
        norm0 = np.linalg.norm(v0)
        if norm0 < 1e-8:
            continue
        ts = np.linspace(1, L - 1, n_points, dtype=int)
        row = []
        for t in ts:
            vt = emb[i, t]
            normt = np.linalg.norm(vt)
            if normt < 1e-8:
                row.append(0.0)
            else:
                row.append(float(np.dot(v0, vt) / (norm0 * normt)))
        curves.append(row)
    return np.array(curves)  # (N, n_points)


# ── Métrica 3: Inter-trajectory centroid consistency ─────────────────────────

def centroid_per_trajectory(emb, lengths):
    """Centroide de cada trayectoria (tokens 1..L)."""
    centroids = []
    for i, L in enumerate(lengths):
        if L < 2:
            continue
        centroids.append(emb[i, 1:L].mean(axis=0))
    return np.vstack(centroids)  # (N, D)


def pairwise_cosine_mean(mat):
    """Similitud coseno media entre todos los pares de filas."""
    # normalizar filas
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat_n = mat / (norms + 1e-10)
    N = len(mat_n)
    total = 0.0
    count = 0
    # batch para evitar OOM
    BATCH = 50
    for i in range(0, N, BATCH):
        chunk = mat_n[i:i+BATCH]
        sims = chunk @ mat_n.T         # (batch, N)
        # excluir diagonal y triangulo inferior para no duplicar
        for bi, gi in enumerate(range(i, i + len(chunk))):
            sims[bi, :gi+1] = 0.0     # zero diagonal + lower
        total += sims.sum()
        count += max(0, N * len(chunk) - (gi + 1) * len(chunk) // 2)
    # recount exact
    count = N * (N - 1) // 2
    pairs = []
    for i in range(N):
        sims_i = mat_n[i] @ mat_n[i+1:].T
        pairs.extend(sims_i.tolist())
    return float(np.mean(pairs)), float(np.std(pairs))


# ── Métrica 4: Centroid drift across token position ──────────────────────────

def centroid_drift(emb, lengths, n_points=20):
    """
    Centroide del ensemble en posición t — ||centroid(t) - centroid(0)||.
    Si el centroide permanece estable, la deriva es baja.
    """
    # Encontrar longitud mínima razonable
    min_L = int(np.percentile(lengths[lengths > 4], 10))
    min_L = max(min_L, 5)
    ts = np.linspace(0, min_L - 1, n_points, dtype=int)

    # Filtrar trayectorias con L >= min_L
    mask = lengths >= min_L
    sub = emb[mask]   # (M, T, D)
    if len(sub) < 5:
        return ts, np.zeros(n_points), np.zeros(n_points)

    centroids_t = []
    for t in ts:
        c = sub[:, t, :].mean(axis=0)
        centroids_t.append(c)
    centroids_t = np.vstack(centroids_t)   # (n_points, D)

    c0 = centroids_t[0]
    drifts = np.linalg.norm(centroids_t - c0, axis=1)
    return ts, drifts, sub


# ── Métrica 5: Silhouette axis vs generic_assistant ──────────────────────────

def silhouette_groups(centroids_dict):
    """
    Silhouette score de los centroides de trayectoria en el espacio PCA-50.
    axis vs generic_assistant — ¿forman clusters distintos?
    """
    groups = [k for k in ["axis", "generic_assistant", "vanilla"] if k in centroids_dict]
    all_vecs = np.vstack([centroids_dict[g] for g in groups])
    labels   = np.concatenate([[i]*len(centroids_dict[g]) for i, g in enumerate(groups)])

    # PCA-50 para reducir ruido
    n_comp = min(50, all_vecs.shape[0]-1, all_vecs.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    reduced = pca.fit_transform(all_vecs)
    var_exp = pca.explained_variance_ratio_.sum()

    sil = silhouette_score(reduced, labels, metric="cosine")
    return sil, var_exp


# ── Métrica 6: Variance explained by 1st PC ──────────────────────────────────

def first_pc_variance(emb, lengths):
    """
    ¿Qué fracción de varianza captura el 1er componente principal?
    Mayor = mayor concentración dimensional (menos dimensiones efectivas).
    """
    centroids = centroid_per_trajectory(emb, lengths)
    n_comp = min(10, centroids.shape[0]-1, centroids.shape[1])
    if n_comp < 2:
        return 0.0
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(centroids)
    return float(pca.explained_variance_ratio_[0])


# ── Métrica 7: Temporal self-similarity (autocorrelation) ────────────────────

def temporal_self_similarity(emb, lengths):
    """
    Autocorrelación de lag-1 de la serie ||v_t||.
    Alta autocorrelación → el estado persiste en el tiempo.
    """
    acorrs = []
    for i, L in enumerate(lengths):
        if L < 4:
            continue
        norms = np.linalg.norm(emb[i, :L], axis=1)  # (L,)
        if np.std(norms) < 1e-8:
            continue
        # Pearson lag-1
        x = norms[:-1] - norms[:-1].mean()
        y = norms[1:] - norms[1:].mean()
        denom = np.std(norms[:-1]) * np.std(norms[1:]) * (L - 1)
        if denom < 1e-10:
            continue
        acorrs.append(float(np.sum(x * y) / denom))
    return np.array(acorrs)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  ANÁLISIS DE DINÁMICA DE TRAYECTORIA — Gemma-4-E4B-it (8B real, 2560D, 42 capas)")
    print("="*70)

    data = {}
    for g in GROUPS:
        emb, lengths = load_group(g)
        data[g] = (emb, lengths)
        print(f"\n{COLORS[g]} {g}: {emb.shape[0]} trayectorias, "
              f"longitud media = {lengths.mean():.1f} tokens, "
              f"dim = {emb.shape[2]}")

    # ── PRA ──────────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("1. PROMPT-RESPONSE ALIGNMENT  (cos(v₀, centroide_respuesta))")
    print("   Mayor = condicionamiento inicial persiste durante generación")
    print("─"*70)
    pra_results = {}
    for g in GROUPS:
        emb, lengths = data[g]
        pra = compute_pra(emb, lengths)
        pra_results[g] = pra
        print(f"  {COLORS[g]} {g:<22}  mean={pra.mean():.4f}  std={pra.std():.4f}  "
              f"min={pra.min():.4f}  max={pra.max():.4f}")

    axis_pra = pra_results["axis"]
    gen_pra  = pra_results["generic_assistant"]
    delta_pra = axis_pra.mean() - gen_pra.mean()
    print(f"\n  Δ(axis - generic):  {delta_pra:+.4f}")
    if delta_pra > 0.01:
        print("  ✓ axis mantiene mejor alineación inicial → ADN persiste más")
    elif delta_pra < -0.01:
        print("  ✗ generic_assistant mantiene mejor alineación (inesperado)")
    else:
        print("  ~ Diferencia no significativa en PRA")

    # ── Intra-trajectory cosine decay ────────────────────────────────────────
    print("\n" + "─"*70)
    print("2. CURVA DE SIMILITUD CON EL ESTADO INICIAL (cos(v_t, v_0))")
    print("   Asíntota más alta = mayor retención de memoria del estado inicial")
    print("─"*70)
    n_points = 10
    for g in GROUPS:
        emb, lengths = data[g]
        curves = cosine_vs_origin(emb, lengths, n_points=n_points)
        if len(curves) == 0:
            continue
        mean_curve = curves.mean(axis=0)
        print(f"  {COLORS[g]} {g:<22}  "
              f"t=0: {mean_curve[0]:.3f}  "
              f"t=mid: {mean_curve[n_points//2]:.3f}  "
              f"t=end: {mean_curve[-1]:.3f}  "
              f"decay={mean_curve[0]-mean_curve[-1]:.3f}")

    # ── Inter-trajectory cosine (cross-prompt consistency) ───────────────────
    print("\n" + "─"*70)
    print("3. CONSISTENCIA ENTRE TRAYECTORIAS  (cos entre centroides de distintos prompts)")
    print("   Mayor = el ADN fuerza al modelo al mismo región ℌ para cualquier pregunta")
    print("─"*70)
    centroids_dict = {}
    for g in GROUPS:
        emb, lengths = data[g]
        c = centroid_per_trajectory(emb, lengths)
        centroids_dict[g] = c
        mu, sigma = pairwise_cosine_mean(c)
        print(f"  {COLORS[g]} {g:<22}  cos_inter_mean={mu:.4f}  std={sigma:.4f}")

    # ── Silhouette ────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("4. SILHOUETTE SCORE  (separabilidad en PCA-50)")
    print("   Mayor = clusters más distintos e irrepetibles")
    print("─"*70)
    sil, var = silhouette_groups(centroids_dict)
    print(f"  Silhouette (axis vs generic vs vanilla) = {sil:.4f}")
    print(f"  Varianza explicada por PCA-50: {var:.1%}")

    # axis vs generic_assistant only
    sub_dict = {k: centroids_dict[k] for k in ["axis", "generic_assistant"]}
    sil2, var2 = silhouette_groups(sub_dict)
    print(f"  Silhouette (axis vs generic_only)        = {sil2:.4f}")

    # ── 1er PC variance ───────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("5. VARIANZA DEL 1er PC  (concentración dimensional)")
    print("   Mayor = espacio efectivo más comprimido (mayor concentración dimensional)")
    print("─"*70)
    for g in GROUPS:
        emb, lengths = data[g]
        v1 = first_pc_variance(emb, lengths)
        print(f"  {COLORS[g]} {g:<22}  PC1_var = {v1:.4f}")

    # ── Temporal self-similarity (autocorr lag-1) ────────────────────────────
    print("\n" + "─"*70)
    print("6. AUTOCORRELACIÓN LAG-1  (persistencia temporal del estado)")
    print("   Mayor = el estado cambia lentamente (mayor persistencia temporal)")
    print("─"*70)
    for g in GROUPS:
        emb, lengths = data[g]
        ac = temporal_self_similarity(emb, lengths)
        print(f"  {COLORS[g]} {g:<22}  acorr_mean={ac.mean():.4f}  std={ac.std():.4f}")

    # ── Centroid drift ────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("7. DERIVA DEL CENTROIDE TEMPORAL  (||centroide(t) - centroide(0)||)")
    print("   Menor deriva = el ensemble permanece en la misma región")
    print("─"*70)
    for g in GROUPS:
        emb, lengths = data[g]
        ts, drifts, _ = centroid_drift(emb, lengths, n_points=8)
        drift_str = "  ".join(f"t{ts[i]}:{drifts[i]:.1f}" for i in range(len(ts)))
        print(f"  {COLORS[g]} {g:<22}  {drift_str}")

    # ── Resumen ejecutivo ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  RESUMEN EJECUTIVO — Indicadores de Persistencia y Consistencia Geométrica")
    print("="*70)

    axis_pra_mean = pra_results["axis"].mean()
    gen_pra_mean  = pra_results["generic_assistant"].mean()
    van_pra_mean  = pra_results["vanilla"].mean()

    axis_inter,_ = pairwise_cosine_mean(centroids_dict["axis"])
    gen_inter,_  = pairwise_cosine_mean(centroids_dict["generic_assistant"])
    van_inter,_  = pairwise_cosine_mean(centroids_dict["vanilla"])

    print(f"\n  PRA    axis={axis_pra_mean:.4f}  generic={gen_pra_mean:.4f}  Δ={axis_pra_mean-gen_pra_mean:+.4f}")
    print(f"  Inter  axis={axis_inter:.4f}   generic={gen_inter:.4f}    Δ={axis_inter-gen_inter:+.4f}")
    print(f"  Sil(axis vs generic+vanilla) = {sil:.4f}")
    print(f"  Sil(axis vs generic)          = {sil2:.4f}")

    verdict = []
    if axis_pra_mean > gen_pra_mean + 0.005:
        verdict.append("✓ PRA: axis retiene mejor el estado inicial")
    if axis_inter > gen_inter + 0.005:
        verdict.append("✓ Inter-traj: axis forma un cluster más cohesivo")
    if sil2 > 0.3:
        verdict.append("✓ Silhouette > 0.3: axis y generic son geométricamente distinguibles")
    if sil2 > 0.5:
        verdict.append("✓✓ Silhouette > 0.5: separación FUERTE")

    if verdict:
        print("\n  INDICADORES GEOMÉTRICOS POSITIVOS DETECTADOS:")
        for v in verdict:
            print(f"    {v}")
    else:
        print("\n  No se detectan indicadores geométricos claros con estos umbrales.")

    print()


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
