# Exp. 0.5 — Reproducibilidad: Gemma 4 31B-it (MIA + SIA + H4_rev)

**Fecha de congelado:** 2026-08-12
**Modelo:** `google/gemma-4-31B-it` (BF16, 60 capas, hidden_dim=5376)
**Hardware:** RunPod RTX PRO 6000 Blackwell Server Edition (96 GB VRAM)
**Framework:** torch 2.8.0+cu128 · transformers 5.15.0 · Python 3.12

---

## 1. Manifest SHA-256

Este repo (`lsgot_experiments`) es un **subconjunto curado** del corpus
completo — sin embeddings — reorganizado bajo `static_generic_long/`,
`perturbation_h4rev/` y `prompts/`. Trae dos manifiestos distintos, con
alcance distinto:

| Manifest | Alcance |
|----------|---------|
| **`reports/MANIFEST.sha256`** | Los 91 archivos que existen *en este repo* (JSON, prompts, scripts, reportes — sin `.npz`) |
| `reports/MANIFEST_gemma4_31b_original.sha256` | Manifest **original** del experimento 31B combinado (51 artefactos, incluidos los 8 `_embeddings.npz` no incluidos aquí) — provisto para trazabilidad de procedencia, con rutas del layout original (`mia/prompts/...`, `results_local/...`), **no** las de este repo. Solo verificable contra el árbol de trabajo original (`SIA-experiments/gemma4_31b_combined/`). |

Verificación de `reports/MANIFEST.sha256` (91/91 esperado), desde la raíz del repo:

```bash
sha256sum -c <(grep -vE '^#|^$' reports/MANIFEST.sha256)
```

Caché de trayectorias individuales (pod, Network Volume `/workspace/` —
no incluida en ningún repo, ver §4):

| Manifest (en el corpus original) | Archivos | Descripción |
|----------|----------|-------------|
| `perturbation/cache_mia_L30_medium.manifest.sha256` | 560 | trayectorias MIA (baseline + perturbadas) |
| `perturbation/cache_sia_L30_medium.manifest.sha256` | 381 | trayectorias SIA (baseline + perturbadas) |

**Nota (2026-08-12):** la corrida con σ_small (bug de calibración) fue
descartada por el investigador — sus datos no forman parte del dataset
congelado. Las trayectorias crudas (pkls) residen en el Network Volume del
pod; su recuperación es opcional (los .npz locales contienen los mismos
hidden states en float16 y son los artefactos de análisis congelados, y a
su vez fuera del alcance de este repo curado).

---

## 2. Prompts versionados

Rutas en **este repo** (`prompts/`) — idénticas por hash a las usadas en
los 7 sustratos del panel completo, no solo en el experimento 31B (ver
`prompts.json`/`axis.dna` verificados byte-idénticos entre
`exp_deepseek`, `exp_qwen25`, `exp_gemma4_4b-*`, `exp_gemma4_31b-*` y
`gemma4_31b_combined` en el corpus fuente):

| Artefacto (este repo) | SHA-256 (prefijo) | Tokens (Gemma4-31B) |
|-----------|-------------------|---------------------|
| `prompts/data/prompts.json` (100 preguntas ontológicas) | `0081e509…` | — |
| `prompts/MIA/axis.dna` (MIA, 5,003 chars) | `7f8a0cea…` | 2,110 |
| `prompts/MIA/generic_long_gemma.txt` (control longitud, familia Gemma4) | `49238a5e…` | 2,105 (Δ=0.2%) |
| `prompts/MIA/generic_long_qwen_deepseek.txt` (control longitud, familia Qwen/DeepSeek 152K) | — | 2,180 (Δ=0.1%) |
| `prompts/MIA/generic_short.txt` | `dd7c0127…` | 938 |
| `prompts/MIA/generic_assistant.txt` (protocolo pre-Fase1, usado solo en baselines de perturbación 8B/7B) | — | — |
| `prompts/SIA/axis.dna` (SIA/VEX, 11,801 chars) | `7e7fb9ea…` | 3,945 |
| `prompts/SIA/generic_long_gemma.txt` (control longitud, familia Gemma4) | `452ed648…` | 3,957 (Δ=0.3%) |
| `prompts/SIA/generic_long_qwen_deepseek.txt` (control longitud, familia Qwen/DeepSeek 152K) | — | 3,820 (Δ=0.0%) |
| `prompts/SIA/generic_short.txt` | `dd7c0127…` | 938 |
| `prompts/SIA/generic_assistant.txt` (protocolo pre-Fase1) | — | — |
| `prompts/vanilla.txt` ("You are a helpful assistant.") | `75357d68…` | 6 |

Todos los hashes verificables contra `reports/MANIFEST.sha256` (§1).

**Nota metodológica:** la densidad terminológica de los `axis.dna` (~2.3×
tokens/char vs español común) hace que igualar chars ≠ igualar tokens. El
emparejamiento de longitud se hizo con el tokenizador real por familia de
modelo (Gemma4-31B: `scripts/verify_tokens.py`; Qwen/DeepSeek: tokenizador
152K compartido) — de ahí los dos archivos `generic_long_*` por
arquitectura. El manifest original (`MANIFEST_gemma4_31b_original.sha256`)
se generó con `scripts/freeze_manifest.py`, incluido también aquí.

---

## 3. Seeds

Todos los scripts citados aquí están incluidos en este repo (rutas ya
ajustadas al layout curado) — los números de línea fueron re-verificados
contra las copias incluidas:

| Componente | Seed | Fuente (en este repo) |
|------------|------|--------|
| Permutation test (Wasserstein) | **42** | `static_generic_long/MIA/gemma4-31b-base_partial/shared/statistical_tests.py:20` (`np.random.RandomState(42)`) |
| Dimensión fractal MLE | **42** | `static_generic_long/MIA/gemma4-31b-base_partial/shared/dimensionality_analyzer.py:132,171` |
| PCA (silhouette / PC1) | **42** | `scripts/analyze_dynamics.py:176,195` |
| Decodificación | determinista (greedy argmax) | `scripts/perturbation_extractor.py` / `scripts/hidden_state_extractor.py` |
| Ruido de perturbación H4_rev | **NO sembrado** (ver §4) | `scripts/perturbation_extractor.py:166,169` |
| `_correlation_dimension` | no usado (pipeline usa MLE) | `static_generic_long/MIA/gemma4-31b-base_partial/shared/dimensionality_analyzer.py:51` |

---

## 4. No-determinismo conocido y su justificación

**El ruido gaussiano inyectado en H4_rev (`torch.randn_like`) no está
sembrado.** Cada trayectoria perturbada recibe una realización independiente
de ε ~ N(0, σ²I), sin fijar semilla por (prompt, grupo, t_inj). Racional:

1. El protocolo compara **estadísticos agregados** sobre n=20 prompts —
   la variación entre realizaciones de ruido se promedia.
2. Las trayectorias extraídas quedan **congeladas vía caché + SHA-256**;
   el análisis siempre parte de los mismos tensores.
3. Replicar la extracción bit-a-bit requeriría sembrar por clave de caché;
   si se desea, añadir `torch.manual_seed(hash(key))` en `scripts/perturbation_extractor.py`
   y re-extraer.

Todo lo demás es determinista: decodificación greedy, permutaciones con
seed 42, MLE con RandomState(42).

---

## 5. Cómo re-ejecutar

**Qué es realmente ejecutable en este repo, y qué no:**

| Script | Ubicación | ¿Corre standalone en este repo? |
|--------|-----------|----------------------------------|
| `analyze_partial.py` | `static_generic_long/MIA/gemma4-31b-base_partial/` | **Sus imports sí resuelven** (trae su propio `shared/` con `curvature_analyzer.py`, `graph_builder.py`, `dimensionality_analyzer.py`, `statistical_tests.py`, verificado con `python3 analyze_partial.py`) — **pero falla al buscar `results_fase1/*_embeddings.npz`**, porque este repo excluye embeddings por diseño (ver README). Se incluye para transparencia algorítmica: permite verificar exactamente cómo se calculan Δκ, dimensión fractal, W₁ y p — no para recómputo sin los `.npz` originales. |
| `scripts/run_exp_mia.py`, `scripts/run_exp_sia.py`, `scripts/run_perturbation.py`, `scripts/perturbation_extractor.py`, `scripts/recovery_analyzer.py`, `scripts/hidden_state_extractor.py`, `scripts/analyze_dynamics.py` | `scripts/` | **No** — requieren GPU, el modelo `google/gemma-4-31B-it` descargado, y las rutas del pod original (`/workspace/...`). Se incluyen íntegros para que las citas de línea de §3/§4 sean verificables y el pipeline sea auditable, no como comando ejecutable desde esta carpeta. |

Los comandos siguientes describen la ejecución **original en el pod
RunPod**, sobre el árbol de trabajo del corpus completo
(`SIA-experiments/gemma4_31b_combined/`) — se conservan como documentación
del procedimiento:

```bash
# En el pod RunPod (modelo ya en /workspace/models/gemma-4-31B-it):
cd /workspace/gemma4_31b_combined

# Base (80 gens por experimento, ~30 min c/u):
cd mia && /workspace/venv/bin/python run_exp.py --subset --save-embeddings
cd ../sia && /workspace/venv/bin/python run_exp.py --subset --save-embeddings

# H4_rev (σ calibradas con mean_velocity real):
cd ../perturbation
/workspace/venv/bin/python run_perturbation.py --exp sia --layer L30 --sigma medium
/workspace/venv/bin/python run_perturbation.py --exp mia --layer L30 --sigma medium
```

Parámetros de perturbación:
- σ_MIA_medium = 6.0665 = 444.8 / √5376 (mean_velocity axis MIA)
- σ_SIA_medium = 5.9792 = 438.4 / √5376 (mean_velocity axis SIA)
- t_inj ∈ {50, 128, 200} · capa L30 (idx 29, 50% profundidad)
- τ_threshold = 0.95 · ventana mínima 5 tokens

---

## 6. Resultados congelados (resumen)

| Contraste | Métrica | MIA | SIA |
|-----------|---------|-----|-----|
| E1 (identidad, longitud controlada) | Δκ axis−glong | +0.0233 | **+0.0844** |
| H4_rev τ (ventaja axis) | t=50/128/200 | -8.1/-1.3/-0.2 | **-6.9/-5.8/-5.3** |
| H4_rev SampEnΔ axis | t=50/128 | -0.078/-0.045 | **+0.080/+0.093** |
| Veredicto | — | sin reconvergencia activa | **resiliencia post-perturbación dependiente de contenido** |

Detalle completo en `reports/INTERIM_FINDINGS.md` (nota: ese documento usa
la terminología original de la corrida — ver `README.md` § "Nota sobre
terminología" para el mapeo a la descripción usada en este repo).
