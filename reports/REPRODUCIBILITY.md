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

| Manifest | Alcance | Verificación |
|----------|---------|--------------|
| **`reports/MANIFEST.sha256`** | Los 78 archivos que existen *en este repo* (JSON, prompts, scripts, reportes — sin `.npz`) | `sha256sum -c <(grep -vE '^#\|^$' reports/MANIFEST.sha256)` desde la raíz del repo → 78/78 OK |
| `reports/MANIFEST_gemma4_31b_original.sha256` | Manifest **original** del experimento 31B combinado (51 artefactos, incluidos los 8 `_embeddings.npz` no incluidos aquí) — provisto para trazabilidad de procedencia, con rutas del layout original (`mia/prompts/...`, `results_local/...`), **no** las de este repo | Solo verificable contra el árbol de trabajo original (`SIA-experiments/gemma4_31b_combined/`), no contra este repo |

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
modelo (Gemma4-31B: `verify_tokens.py`; Qwen/DeepSeek: tokenizador 152K
compartido) — de ahí los dos archivos `generic_long_*` por arquitectura.

---

## 3. Seeds

| Componente | Seed | Fuente |
|------------|------|--------|
| Permutation test (Wasserstein) | **42** | `shared/statistical_tests.py:20` (`np.random.RandomState(42)`) |
| Dimensión fractal MLE | **42** | `shared/dimensionality_analyzer.py:132,171` |
| PCA (silhouette / PC1) | **42** | `shared/analyze_autopoiesis.py:174,193` |
| Decodificación | determinista (greedy argmax) | `perturbation_extractor.py` / `hidden_state_extractor.py` |
| Ruido de perturbación H4_rev | **NO sembrado** (ver §4) | `perturbation_extractor.py:166,169` |
| `_correlation_dimension` | no usado (pipeline usa MLE) | `shared/dimensionality_analyzer.py:51` |

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
   si se desea, añadir `torch.manual_seed(hash(key))` en el hook y re-extraer.

Todo lo demás es determinista: decodificación greedy, permutaciones con
seed 42, MLE con RandomState(42).

---

## 5. Cómo re-ejecutar

**Nota:** los comandos de esta sección describen la ejecución original en
el pod RunPod, sobre el árbol de trabajo del corpus completo
(`SIA-experiments/gemma4_31b_combined/`) — no sobre este repo curado, que
no incluye los scripts `run_exp.py`/`run_perturbation.py` de esa corrida
(sí incluye `static_generic_long/MIA/gemma4-31b-base_partial/analyze_partial.py`,
usado para el recálculo local de la corrida parcial 31B-base). Se
conservan aquí como documentación del procedimiento, no como comando
ejecutable directamente desde esta carpeta.

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
| Veredicto | — | especificidad sin autopoiesis | **autopoiesis débil** |

Detalle completo en `results_local/INTERIM_FINDINGS.md`.
