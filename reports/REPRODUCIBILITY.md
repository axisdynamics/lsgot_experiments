# Exp. 0.5 — Reproducibilidad: Gemma 4 31B-it (MIA + SIA + H4_rev)

**Fecha de congelado:** 2026-08-12
**Modelo:** `google/gemma-4-31B-it` (BF16, 60 capas, hidden_dim=5376)
**Hardware:** RunPod RTX PRO 6000 Blackwell Server Edition (96 GB VRAM)
**Framework:** torch 2.8.0+cu128 · transformers 5.15.0 · Python 3.12

---

## 1. Manifest SHA-256

Todos los artefactos del experimento están congelados por hash en
**`MANIFEST.sha256`** (51 artefactos locales + 1,023 pkls de caché en el pod).

Verificación local:

```bash
cd SIA-experiments/gemma4_31b_combined
sha256sum -c <(grep -v '^#' MANIFEST.sha256)   # → 51/51 OK
```

Caché de trayectorias individuales (pod, Network Volume `/workspace/`):

| Manifest | Archivos | Descripción |
|----------|----------|-------------|
| `perturbation/cache_mia_L30_medium.manifest.sha256` | 560 | trayectorias MIA (baseline + perturbadas) |
| `perturbation/cache_sia_L30_medium.manifest.sha256` | 381 | trayectorias SIA (baseline + perturbadas) |

**Nota (2026-08-12):** la corrida con σ_small (bug de calibración) fue
descartada por el investigador — sus datos no forman parte del dataset
congelado. Las trayectorias crudas (pkls) residen en el Network Volume del
pod; su recuperación es opcional (los .npz locales contienen los mismos
hidden states en float16 y son los artefactos de análisis congelados).

---

## 2. Prompts versionados

| Artefacto | SHA-256 (prefijo) | Tokens (Gemma4-31B) |
|-----------|-------------------|---------------------|
| `data/prompts.json` (100 preguntas ontológicas) | `0081e509…` | — |
| `mia/prompts/axis.dna` (MIA, 5,003 chars) | `7f8a0cea…` | 2,110 |
| `mia/prompts/generic_long.txt` (control longitud) | `49238a5e…` | 2,105 (Δ=0.2%) |
| `mia/prompts/generic_short.txt` | `dd7c0127…` | 938 |
| `sia/prompts/axis.dna` (SIA/VEX, 11,801 chars) | `7e7fb9ea…` | 3,945 |
| `sia/prompts/generic_long.txt` (control longitud) | `452ed648…` | 3,957 (Δ=0.3%) |
| `sia/prompts/generic_short.txt` | `dd7c0127…` | 938 |
| vanilla ("You are a helpful assistant.") | `75357d68…` | 6 |

**Nota metodológica:** la densidad terminológica de los `axis.dna` (~2.3×
tokens/char vs español común) hace que igualar chars ≠ igualar tokens. El
emparejamiento de longitud se hizo con el tokenizador real (Gemma4-31B:
`verify_tokens.py`); variantes Qwen/DeepSeek en `prompts/generic_long_qwen_*.txt`.

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
