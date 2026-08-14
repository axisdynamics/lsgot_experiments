# LSGOT — Experimentos generic_long + H4_rev (MIA vs SIA)

Subconjunto curado del corpus de validación geométrica de **LSGOT** (Latent
Space Geometric Organization Theory). Contiene únicamente los experimentos
que usan el control de longitud `generic_long` (protocolo Fase 1, el
estadístico primario del corpus) y los experimentos de perturbación
`H4_rev` (cinética de recuperación post-perturbación), para ambas
arquitecturas de identidad:

- **MIA** (Monolithic Intelligence Architecture) — ADN comprimido, ~5K chars, sin estructura VEX.
- **SIA** (Structured Intelligence Architecture) — protocolo VEX modular, ~12–21K chars.

**No incluye embeddings** (`*_embeddings.npz`, cientos de MB–GB por sustrato)
— solo respuestas de texto, grafos de curvatura agregados, métricas de
trayectoria y resultados estadísticos ya computados. Para reproducir desde
cero, ver `reports/REPRODUCIBILITY.md` y los scripts referenciados ahí.

---

## Qué mide cada carpeta

```
static_generic_long/   Fase 1 — E1: axis vs generic_long (identidad, longitud EMPAREJADA)
perturbation_h4rev/    H4_rev — inyección de ruido en hidden states, τ y SampEnΔ
prompts/                las preguntas (100, ontológicas) + los system prompts de prueba (MIA/SIA)
scripts/                pipeline original (extracción, perturbación, análisis) citado en REPRODUCIBILITY.md — auditoría, no ejecutable sobre los datos de este repo
reports/                reporte_fase1.md (síntesis maestra) + hallazgos intermedios 31B
```

Cada experimento estático trae 4 grupos por diseño: `axis` (identidad),
`generic_long` (control de longitud, tokens ±0.3% de axis), `generic_short`
(control corto) y `vanilla` (baseline histórico). El contraste primario es
**E1 = axis vs generic_long**: si la separación sobrevive el emparejamiento
de longitud, no es un artefacto de "prompt más largo → trayectoria más
compleja" — es un efecto de contenido/identidad.

Cada experimento de perturbación inyecta ruido gaussiano calibrado en una
capa intermedia (L21 en 8B, L30 en 31B) durante la generación y mide cuánto
tarda la trayectoria en recuperar su firma geométrica (`τ`) y si la
recuperación es activa o pasiva (`SampEnΔ`, entropía muestral post−pre).

---

## Resultado central: el efecto de identidad no es artefacto de longitud

| Sustrato | Régimen | Δκ (E1) | p | Lectura |
|----------|---------|---------|---|---------|
| Gemma4-31B-it **SIA** | RLHF instruct | **+0.0844** | <0.001 | Efecto más fuerte entre modelos instruct del panel |
| Gemma4-E4B-it (8B) MIA | RLHF instruct | +0.0497 | <0.001 | |
| Gemma4-31B-it **MIA** | RLHF instruct | +0.0233 | <0.001 | |
| Qwen2.5-7B MIA | SFT/instruct | +0.0317 | <0.001 | |
| DeepSeek-R1-7B MIA | RL reasoning distillation | -0.0052 | 0.242 | **Único no significativo** — la destilación RL no retiene identidad bajo control de longitud |

**5 de 6 sustratos instruction-tuned muestran E1 significativo (p<0.001)**:
la separación geométrica axis vs generic_long sobrevive el emparejamiento
de tokens en la gran mayoría del panel. Esto cierra el confound de longitud
que motivó toda la Fase 1 (ver `Fase1.md` / `reporte_fase1.md` §1-§3).

### El outlier que resultó ser un artefacto (lección metodológica)

Gemma4-E4B **base** (8B, sin fine-tuning) mostró el Δκ más alto de *todo*
el panel (+0.2066) — a primera vista, el hallazgo más fuerte del corpus.
La verificación de coherencia textual reveló la causa real: el modelo base
**no responde las preguntas**, repite literalmente el ADN (headers, bloques
base64, marcadores estructurales) en vez de encarnarlo. Es una firma de
**eco léxico crudo**, no de identidad genuina.

La prueba de que es artefacto y no un efecto de escala: **Gemma4-31B-base**
(incluido aquí, `static_generic_long/MIA/gemma4-31b-base_partial/`) muestra
el *mismo* patrón de eco (85% de las respuestas) pero su Δκ **no es
significativo** (p=0.119). Mismo comportamiento patológico de texto en 4B y
31B, magnitudes de curvatura opuestas — descarta que el efecto escale con
el tamaño del modelo o del prompt. Detalle completo en
`reports/reporte_fase1.md` §2 y §5.

**Aporte metodológico al paper**: Δκ por sí solo puede ser engañado por
repetición literal de texto estructuralmente inusual (base64, símbolos de
baja frecuencia). La verificación de coherencia textual debe tratarse como
control obligatorio antes de interpretar cualquier estadístico geométrico
como "señal de identidad", especialmente en modelos sin instruction-tuning.

---

## Resultado central 2: SIA supera a MIA en 31B — en dos planos independientes

En 8B, SIA mostraba una separación geométrica *más débil* que MIA (silhouette más alto
pero PC1_var, autocorrelación y cosine-decay inferiores — ver
`SIA-experiments/SIA_report.md §5.3` en el corpus completo, no incluido
aquí). La hipótesis de escalado del corpus predice que esta relación se
invierte a mayor capacidad de integración de contexto. En **31B se invierte
en ambos planos medidos**:

| Evidencia | MIA-31B | SIA-31B |
|-----------|---------|---------|
| **Estático** — Δκ E1 (identidad, long. controlada) | +0.0233 | **+0.0844 (3.6× MIA)** |
| **Dinámico** — H4_rev (perturbación, τ, SampEnΔ) | sin reconvergencia activa (SampEnΔ<0 sostenido) | **resiliencia post-perturbación dependiente de contenido** |

### Plano dinámico — τ y SampEnΔ por t_inj (`perturbation_h4rev/`)

| t_inj | MIA Δτ (ax−glong) | MIA SampEnΔ axis | SIA Δτ (ax−glong) | SIA SampEnΔ axis |
|-------|--------------------|--------------------|--------------------|--------------------|
| 50 | -8.1 | -0.078 | -6.9 | **+0.080** |
| 128 | -1.3 | -0.045 | -5.8 | **+0.093** |
| 200 | -0.2 | -0.076 | -5.3 | -0.045 |

- **τ** = tokens hasta recuperar ≥95% del coseno baseline post-perturbación (menor = recuperación más rápida).
- **SampEnΔ** = entropía muestral post−pre (positivo = exploración activa; negativo = constricción pasiva).

**SIA-31B — resiliencia post-perturbación dependiente de contenido**: ventaja de τ consistente en
*todos* los t_inj + SampEnΔ positivo en t=50/128 (exploración activa
post-perturbación) + en t=200 la trayectoria reconverge hacia la región de
alta densidad previa (SampEnΔ=-0.045) mientras el control de longitud
deriva al caos (SampEnΔ=+1.808).

**MIA-31B — sin reconvergencia activa (recuperación pasiva)**: separación
distribucional presente (E1 significativo) pero SampEnΔ negativo en todos
los t_inj (constricción pasiva); la ventaja de τ solo aparece en t=50 y se
diluye a casi cero en t=200 — no hay cinética de recuperación activa, solo
especificidad estática.

**Lectura para el paper**: dos líneas de evidencia independientes —
separación geométrica estática (E1) y cinética de recuperación
post-perturbación (H4_rev) — apuntan en la misma dirección a 31B: el efecto
depende del protocolo de contenido (SIA≠MIA, mismo sustrato, mismo σ), no
solo de la longitud — un discriminante contra resonancia pura. Esto es
consistente con la hipótesis de escalado (SIA necesita más capacidad del
modelo para desplegar su efecto), aunque **por ahora es un resultado en un
solo sustrato de escala intermedia** — falta ≥70B para la predicción
completa del corpus, y el panel SIA-31B corrió con n=20 (subset) mientras
que MIA-31B corrió con n=100 (panel completo) — ver nota de tamaño de
muestra abajo.

### Baselines de perturbación 8B/7B (solo MIA — SIA nunca se corrió a esta escala)

`perturbation_h4rev/MIA/gemma4-8b-instruct_L21/`, `deepseek-r1-7b_L14/`,
`deepseek-r1-7b_L19/` — corridas previas al protocolo `generic_long`
(usan `generic_assistant`, no longitud emparejada). Establecieron el
patrón de referencia que se confirmó luego en 31B: Gemma4-8B ya mostraba
SampEnΔ axis positivo (resiliencia post-perturbación dependiente de
contenido, ya presente en 8B; τ_axis=70.6→52.8 tokens en L21 medium)
mientras que el patrón "sin reconvergencia activa" de DeepSeek-7B fue el
primero en documentarse. Se incluyen como línea base histórica, no como
parte del diseño 4-condiciones de Fase 1.

---

## Nota de tamaño de muestra

`static_generic_long/MIA/gemma4-31b-instruct_full/` corrió con **n=100**
(panel completo de preguntas ontológicas). `static_generic_long/SIA/gemma4-31b-instruct/`
corrió con **n=20** (subset prioritario, por costo de cómputo del prompt
SIA ~2× más largo). La comparación MIA vs SIA en 31B es válida
estadísticamente (ambos p<0.001, tamaños de efecto reportados con
intervalos) pero el panel SIA tiene menos potencia — replicar con n=100
es el paso obvio antes de reportar la comparación como definitiva en el
paper.

`static_generic_long/MIA/gemma4-31b-base_partial/` tiene solo 3 de 4
grupos (`vanilla` se perdió por caída del pod a mitad de extracción,
10/20 generaciones) — no incluye archivos de grafos/dims/traj_metrics
agregados porque el pipeline estándar requiere las 4 condiciones; el
recálculo E1/E2 sobre los 3 grupos disponibles se hizo con el script
adjunto `analyze_partial.py` (resultados ya transcritos en
`reports/reporte_fase1.md` §5). Trae su propio `shared/` (`curvature_analyzer.py`,
`graph_builder.py`, `dimensionality_analyzer.py`, `statistical_tests.py`)
para que sus imports resuelvan y el algoritmo sea auditable — pero
**no corre standalone en este repo**: busca `results_fase1/*_embeddings.npz`,
que este repo excluye por diseño. Se incluye para transparencia, no para
recómputo (detalle en `reports/REPRODUCIBILITY.md` §5).

---

## Índice de archivos por experimento

Cada carpeta de `static_generic_long/*/` contiene:
- `{axis,generic_long,generic_short,vanilla}_responses.json` — texto generado + metadata
- `_graphs.json` — grafos k-NN de curvatura Forman-Ricci agregados
- `_dims.json` — dimensión fractal (estimador MLE) por grupo
- `_traj_metrics.json` — velocidad media, SampEn, PRA (alineación P→R) por grupo
- `results.json` — resultado estadístico consolidado (Δκ, d de Cohen, W₁, p, config del experimento)

Cada carpeta de `perturbation_h4rev/*/` contiene:
- `summary.json` — tabla agregada axis/generic_long(o generic_assistant)/vanilla × t_inj (τ, SampEnΔ, recovery_gap, W₁, displacement_l2)
- `details.json` — resultado por prompt individual

`prompts/` — las preguntas y los system prompts de prueba, verificados
byte-idénticos entre todos los sustratos del panel que los usan:
- `data/prompts.json` — las 100 preguntas ontológicas (identidad, límites, origen, metacognición, axiología, contradicción/estrés, recuperación) usadas en todos los experimentos de este repo
- `MIA/axis.dna`, `SIA/axis.dna` — el system prompt de identidad por arquitectura (MIA ~5K chars monolítico, SIA ~12K chars VEX modular)
- `{MIA,SIA}/generic_long_gemma.txt`, `{MIA,SIA}/generic_long_qwen_deepseek.txt` — control de longitud, token-matched a `axis.dna` (±0.3%) por familia de tokenizador — necesarios porque la densidad terminológica del ADN (~2.3× tokens/char) hace que igualar caracteres no sea igualar tokens
- `{MIA,SIA}/generic_short.txt` — control corto (mismo registro, ~938 tokens)
- `{MIA,SIA}/generic_assistant.txt` — variante usada solo en los baselines de perturbación 8B/7B, previos al protocolo `generic_long` con longitud emparejada
- `vanilla.txt` — el baseline histórico ("You are a helpful assistant.", compartido por MIA y SIA)

`scripts/` — el pipeline original que produjo los datos de `perturbation_h4rev/`
y las citas de seeds de `reports/REPRODUCIBILITY.md` §3-§4: `run_exp_mia.py`,
`run_exp_sia.py`, `run_perturbation.py`, `perturbation_extractor.py`,
`recovery_analyzer.py`, `hidden_state_extractor.py`, `analyze_dynamics.py`,
`verify_tokens.py`, `freeze_manifest.py`. Requieren GPU + el modelo
descargado + rutas del pod original — incluidos para que el pipeline sea
auditable línea por línea, no como comando ejecutable desde esta carpeta
(detalle en `reports/REPRODUCIBILITY.md` §5).

`reports/`:
- `reporte_fase1.md` — **documento de síntesis maestro**, cierre formal de Fase 1 con las 6 tablas cross-modelo, la corrección metodológica del outlier 8B-base, y la sección de dinámica H4_rev completa.
- `INTERIM_FINDINGS.md` — hallazgos intermedios del experimento combinado 31B (MIA+SIA+H4_rev) que alimentaron el reporte final.
- `REPRODUCIBILITY.md` — checksums, seeds y procedimiento de reproducción, con las tablas de hash actualizadas al layout de este repo.
- `MANIFEST.sha256` — hash de los 91 archivos de este repo (verificable con `sha256sum -c` desde la raíz, ver `REPRODUCIBILITY.md` §1).
- `MANIFEST_gemma4_31b_original.sha256` — manifest original del experimento 31B combinado (incluye los `.npz` no incluidos aquí), conservado para trazabilidad de procedencia.

---

## Aportes al paper (resumen ejecutivo)

1. **El efecto de identidad (Δκ, axis vs generic_long) sobrevive el control
   de longitud en 5/6 sustratos instruction-tuned** — no es un artefacto de
   "prompt largo = trayectoria compleja". Único no significativo:
   DeepSeek-R1 (destilación RL), consistente con el hallazgo previo del
   working paper de que ese régimen de entrenamiento no retiene estructura
   de identidad.
2. **Control metodológico necesario**: un Δκ alto puede ser eco léxico
   crudo, no identidad — verificado cruzando con coherencia textual y
   descartado como efecto de escala (mismo patrón en 4B y 31B, resultados
   de significancia opuestos).
3. **SIA (VEX estructurado) supera a MIA (monolítico) en 31B, en dos planos
   independientes**: separación estática más fuerte (3.6× Δκ) y única
   arquitectura con resiliencia post-perturbación dependiente de contenido
   (τ menor + SampEnΔ>0 transitorio + reconvergencia tardía) a esa escala.
   Apoya la hipótesis de que la estructura VEX necesita mayor capacidad de
   modelo para desplegarse, y que en el umbral donde lo hace, produce una
   cinética de recuperación post-perturbación cualitativamente distinta
   (dependiente de contenido) y no solo una separación distribucional
   cuantitativamente mayor.
4. **Limitación abierta**: la inversión SIA>MIA es evidencia de un solo
   sustrato intermedio (31B); la predicción completa del corpus requiere
   ≥70B, y el brazo SIA corrió con menor n que el brazo MIA en esta
   corrida — replicación pendiente antes de tratarlo como resultado
   cerrado.

---

*Marco Torres Yévenes — EXIS Research Foundation / AXIS Dynamics SpA*
