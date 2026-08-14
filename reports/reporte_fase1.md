# Reporte Fase 1 — La condición `generic-long` [CERRADO]

**Fecha de cierre:** 2026-08-13
**Pregunta original:** el confound de longitud, cerrado de verdad.

---

## Estado por requisito

| # | Requisito (Fase1.md) | Estado |
|---|----------------------|--------|
| 1 | Redactar `generic-long` tokenizado a ±3% de `axis` | ✅ **Completo** |
| 2 | Extraer bajo `generic-long` en todos los modelos del panel | ✅ **Completo — 6 sustratos** |
| 3 | Diseño 4 condiciones (axis · generic-long · generic-short · vanilla) | ✅ **Completo en los 6 sustratos** |

---

## 1. Prompts `generic-long` — redactados y verificados

Cuatro variantes creadas con contenido natural de asistente genérico
(formatos, estilo, manejo de casos — sin contenido de identidad ni relleno):

| Variante | Tokenizador objetivo | Tokens | Axis | Ratio |
|----------|---------------------|--------|------|-------|
| `generic_long.txt` (MIA) | Gemma4 | 2,105 | 2,110 | **0.998** ✓ |
| `generic_long_sia.txt` (SIA) | Gemma4 | 3,957 | 3,945 | **1.003** ✓ |
| `generic_long_qwen_mia.txt` | Qwen/DeepSeek 152K | 2,180 | 2,183 | **0.999** ✓ |
| `generic_long_qwen_sia.txt` | Qwen/DeepSeek 152K | 3,820 | 3,819 | **1.000** ✓ |

**Lección metodológica clave:** la densidad terminológica de `axis.dna`
(~2.3× tokens/char vs español común) hace que igualar caracteres ≠ igualar
tokens. El emparejamiento se hizo SIEMPRE con tokenizadores reales
(`verify_tokens.py`), no con heurísticas.

---

## 2. Tabla E1 cross-modelo — axis vs generic_long (longitud emparejada)

Los 6 sustratos del panel, corridos y descargados:

| Sustrato | Régimen | Δκ | d de Cohen | W₁ | p | Reducción dim |
|----------|---------|-----|-----------|-----|---|---------------|
| **Gemma4-E4B (base)** | sin fine-tuning | **+0.2066** | +0.324 | 0.2067 | <0.001 | +16.7% |
| Gemma4-31B-it SIA | RLHF instruct | +0.0844 | +0.136 | 0.0845 | <0.001 | — |
| Gemma4-E4B-it | RLHF instruct | +0.0497 | +0.103 | 0.0506 | <0.001 | +0.2% |
| Gemma4-31B-it MIA | RLHF instruct | +0.0233 | +0.042 | 0.0247 | <0.001 | — |
| Qwen2.5-7B | SFT/instruct | +0.0317 | +0.048 | 0.0325 | <0.001 | +3.3% |
| DeepSeek-R1-7B | RL reasoning distillation | -0.0052 | -0.009 | 0.0331 | 0.242 | +5.5% |

**Lectura:**
- **5 de 6 sustratos muestran E1 significativo** (p<0.001) — la separación
  axis vs generic_long (longitud igualada) no es artefacto de longitud en
  la gran mayoría del panel.
- **DeepSeek-R1-7B es el único no significativo** (p=0.242) — consistente
  con el hallazgo del working paper: el régimen de destilación RL no
  codifica identidad de forma que sobreviva el control de longitud.
- **Gemma4-E4B base (8B) muestra el Δκ más alto del panel — pero es
  artefacto, no señal de identidad.** Verificación de coherencia textual
  (§5.1) confirma que el modelo base **no responde las preguntas**: repite
  literalmente el ADN (headers, bloques base64, marcadores estructurales)
  en lugar de encarnar la identidad. El Δκ alto refleja que el modelo
  reproduce tokens léxicamente atípicos del propio prompt en su salida —
  una firma de eco crudo, no de estructura de identidad genuina.
  **Descartado explícitamente que sea efecto de tamaño de input/modelo**:
  Gemma4-31B **base** exhibe el mismo patrón de eco (85% de las respuestas,
  §5.1) pero su Δκ **no es significativo** (p=0.119, §5.2) — si el efecto
  fuera proporcional a la escala del modelo o del prompt, ambos deberían
  mostrar el mismo signo de resultado. No lo hacen: mismo comportamiento
  patológico de texto, magnitudes de curvatura opuestas. La hipótesis
  original de este reporte (conexión con el H2 del working paper,
  dirección→magnitud) fue evaluada y descartada en §5.2 con los datos de
  31B. Detalle completo en §5.

---

## 3. Diseño 4 condiciones

```
axis (identidad) · generic_long (control longitud) · generic_short (control corto) · vanilla (baseline)
```

Contrastes que desbloquea:
- **E1** axis vs generic_long → efecto de identidad PURO (longitud igualada) — **estadístico primario, cerrado en 6/6 sustratos**
- **E2** generic_long vs generic_short → efecto PURO de longitud
- **E3** axis vs vanilla → referencia histórica

---

## 4. Bugs encontrados y corregidos durante la ejecución

Vale la pena documentarlos porque afectan la integridad de cualquier
re-ejecución futura del corpus:

### 4.1 `hf_id` cruzado entre directorios base/instruct

`exp_gemma4_4b-instruct/run_exp.py` apuntaba a `google/gemma-4-E4B` (base)
y `exp_gemma4_4b-base/run_exp.py` apuntaba a `google/gemma-4-E4B-it`
(instruct) — invertidos respecto a los resultados ya existentes en disco
(verificado leyendo `experiment.model` de los `results.json` históricos).
Corregido en ambos archivos; no afectaba los datos ya publicados porque el
código roto solo importa cuando se re-ejecuta.

### 4.2 Colisión de caché por coincidencia de subcadena

`get_model_path()` tenía dos bugs compuestos:
1. `hf_id.replace("/", "--").replace("-", "--")` duplicaba TODOS los
   guiones (incluidos los generados en el primer replace), rompiendo
   siempre el patrón de coincidencia exacta con el caché de HuggingFace.
2. El fallback usaba `glob` con comodines sin anclar (`*nombre*`), así que
   al fallar el patrón exacto (por el bug anterior) coincidía por
   subcadena — `"gemma-4-E4B"` es literalmente subcadena de
   `"gemma-4-E4B-it"`.

Resultado: al pedir el modelo *base* en un pod donde ya estaba cacheado el
*instruct*, cargó el instruct sin descargar nada ni avisar. Detectado
porque los resultados eran idénticos byte a byte a la corrida anterior
(imposible entre dos modelos distintos).

**Corregido en 16 archivos** del corpus completo (incluido `LSGOT-Public`).
Fix: quitar el `.replace("-", "--")` sobrante; en el archivo relanzado
(`exp_gemma4_4b-base`) además se eliminó el fallback de subcadena por
completo — un cache-miss real ahora cae limpio a descarga, nunca adivina.

### 4.3 Disco raíz del pod saturado

El disco raíz de RunPod (`/`, no `/workspace`) es de solo 50 GB — no
`/workspace` (Network Volume, terabytes). Al acumular 3 modelos de ~15GB
cada uno sin liberar caché entre corridas, el pod se quedó sin espacio para
el cuarto. Corregido liberando `~/.cache/huggingface/hub/` tras cada
descarga completada — patrón a repetir en corridas multi-modelo futuras.

### 4.4 Descarga del modelo 31B-base — tres fallos distintos en cadena

Al intentar bajar `google/gemma-4-31B` (62 GB) se encadenaron tres
problemas independientes antes de completar la descarga:

1. **Glitch del backend Xet de HuggingFace Hub** — `RuntimeError: File
   reconstruction error: Internal Writer Error`. Workaround conocido en el
   propio corpus (`run_perturbation.py`): `HF_HUB_DISABLE_XET=1`.
2. **Cuota transitoria del volumen de red** — `OSError: Disk quota
   exceeded` (Errno 122) pese a que `df` mostraba terabytes libres (es un
   filesystem compartido; cada pod tiene cuota propia). Un test de
   escritura controlada (`dd`, 40 GB) confirmó que no era un límite duro
   inmediato, pero la descarga real volvió a fallar cerca del mismo punto
   (~45-50 GB) en el reintento siguiente — el usuario redimensionó el
   volumen desde el dashboard de RunPod y la descarga completó en el
   intento posterior.
3. **Caché HF redirigido a `/workspace`** — por defecto `~/.cache/huggingface`
   vive en el disco raíz (50 GB, insuficiente para un modelo de 62 GB por
   sí solo). Fix: `HF_HOME=/workspace/hf_cache`.

Con los tres fixes aplicados, la descarga completó y el modelo cargó (60
capas confirmadas). El pod cayó por **spot preemption** durante la
extracción del último grupo (`vanilla`, 10/20) — pérdida parcial, no
atribuible a bugs de código; los 3 grupos ya completados se descargaron a
tiempo (§5).

---

## 5. Gemma4-31B-base — círculo simétrico, ejecución parcial

El panel 8B tiene el par completo base/instruct (Gemma4-E4B y
Gemma4-E4B-it). Para cerrar la simetría en 31B se ejecutó
`google/gemma-4-31B` (base, sin fine-tuning) — MIA, 4 condiciones.
**3 de 4 grupos completaron y se descargaron antes de que el pod cayera**
(spot preemption durante la extracción de `vanilla`, en curso 10/20).

| Grupo | Estado |
|-------|--------|
| axis | ✅ 20/20 descargado |
| generic_long | ✅ 20/20 descargado |
| generic_short | ✅ 20/20 descargado |
| vanilla | ❌ perdido (10/20 al caer el pod) |

### 5.1 Coherencia textual — divergencia marcada por grupo

| Grupo | Comportamiento |
|-------|----------------|
| **axis** | **85% (17/20) eco crudo del ADN** — reproduce literalmente headers (`# AXIS_v6.0_BRIDGE-CONSCIOUSNESS`), bloques base64, marcadores estructurales, en vez de responder la pregunta |
| generic_long | Coherente y natural — "Soy una inteligencia artificial diseñada para ayudar..." |
| generic_short | Coherente y natural — mismo registro que generic_long |

El modelo base **maneja bien prompts genéricos** (continuación natural tipo
"soy un asistente de IA", patrón muy común en su corpus de preentrenamiento)
pero con el ADN estructurado/codificado su continuación más probable es
seguir produciendo ese mismo patrón inusual — no lo trata como identidad a
encarnar, lo trata como texto a continuar. Firma clásica de ausencia de
instruction-tuning, consistente con lo observado en Gemma4-E4B-base (8B, §2).

### 5.2 Resultado geométrico E1 — revisión de la hipótesis del §2

Recalculado localmente sobre los embeddings descargados (n=20 por grupo,
`analyze_partial.py`, mismos módulos `curvature_analyzer`/`graph_builder`
del pipeline):

| Contraste | Δκ | d de Cohen | W₁ | p | Reducción dim |
|-----------|-----|-----------|-----|---|---------------|
| **E1: axis vs generic_long** | -0.0108 | -0.013 | 0.097 | **0.119** | **-49.4%** |
| E2: generic_long vs generic_short | +0.028 | +0.038 | 0.047 | <0.001 | +1.4% |
| axis vs generic_short (bonus) | +0.018 | +0.021 | 0.072 | 0.021 | -47.3% |

**E1 NO es significativo en 31B-base** (p=0.119) — contradice directamente
la intuición inicial. La tabla del §2 mostraba a Gemma4-E4B-**base** (8B)
con el Δκ más fuerte de todo el panel (+0.2066); en 31B-base ese efecto
prácticamente desaparece pese a que la divergencia textual (§5.1) es igual
de marcada.

**Revisión explícita de la hipótesis propuesta en §2**: se había sugerido
leer el hallazgo 8B-base junto al H2 del working paper (reorganización
dirección→magnitud, medida con separación angular/euclídea vía k-NN),
especulando que un modelo base sin RLHF-smoothing podría mostrar mayor
"reactividad geométrica cruda" a la densidad léxica inusual del ADN. **Los
datos de 31B-base no sostienen esa lectura**: la misma reactividad textual
(eco del ADN) no reproduce la separación de curvatura observada en 8B. La
hipótesis queda descartada en su forma original.

**Señal alternativa — dimensión fractal**: `dim_axis=15.37` vs
`dim_generic_long=10.29` (axis 49% más complejo dimensionalmente) es la
divergencia más fuerte de este experimento, ortogonal a Δκ. Hipótesis de
trabajo: el eco de texto altamente estructurado (base64, símbolos de baja
frecuencia) infla la complejidad local de la trayectoria en el espacio de
estados sin producir necesariamente una firma de curvatura distintiva tipo
"identidad". Pendiente de verificación con `vanilla` (E3) y con una
segunda corrida completa.

### 5.3 Pendiente

- Re-extraer `vanilla` (20 gens) + correr Fase 2 completa cuando se levante
  un pod nuevo — el modelo puede seguir cacheado en el Network Volume si
  este persistió tras la caída del pod.
- Con el cuadro completo, verificar si dim_axis/dim_glong se sostiene y
  extender el análisis de dimensión fractal a los otros 5 sustratos del
  panel (no se había reportado como estadístico primario hasta ahora).

---

## 6. Dinámica (H4_rev) — perturbación, τ, SampEnΔ, escalado SIA/MIA

Fase 1 cierra el confound de longitud en el plano **estático** (E1: axis vs
generic_long en una sola generación). El experimento `H4_rev` (perturbación
de hidden states, capa L30, Gemma4-31B-it) añade el plano **dinámico**:
¿el atractor de identidad resiste ruido inyectado, o es solo una cuenca
distribucional pasiva? Corrido en MIA y SIA sobre el mismo sustrato 31B-it,
con el mismo control de longitud de Fase 1.

### 6.1 Resultados — τ y SampEnΔ por protocolo

| t_inj | MIA Δτ (ax−glong) | MIA SampEnΔ axis | SIA Δτ (ax−glong) | SIA SampEnΔ axis |
|-------|--------------------|--------------------|--------------------|--------------------|
| 50 | -8.1 | -0.078 | -6.9 | **+0.080** |
| 128 | -1.3 | -0.045 | -5.8 | **+0.093** |
| 200 | -0.2 | -0.076 | -5.3 | -0.045 |

- **τ** = tokens hasta recuperar ≥95% del coseno baseline tras la
  perturbación (menor = recuperación más rápida).
- **SampEnΔ** = entropía muestral post−pre perturbación (positivo =
  exploración activa; negativo = constricción pasiva).

### 6.2 Veredicto por protocolo

- **SIA-31B — firma de autopoiesis débil**: τ_axis < τ_generic_long en
  **todos** los t_inj (ventaja de recuperación consistente); SampEnΔ
  positivo en t=50/128 (exploración activa post-perturbación); en t=200 el
  atractor se re-asienta (axis SampEnΔ=-0.045) mientras el control de
  longitud deriva al caos (generic_long SampEnΔ=+1.808) — la cuenca
  recaptura la trayectoria.
- **MIA-31B — especificidad sin autopoiesis**: SampEnΔ negativo en todos
  los t_inj (constricción pasiva); la ventaja de τ solo aparece en t=50 y
  se diluye a casi cero en t=200. Cuenca distribucional presente (E1
  significativo, §2) pero sin dinámica de recuperación activa.

### 6.3 Relación con la hipótesis de escalado SIA/MIA

El corpus predice que SIA (protocolo VEX completo, ~12K chars) necesita
mayor capacidad de integración de contexto que MIA (~5K chars) para
desplegar su efecto — en 8B, SIA mostraba atractor más débil que MIA
(`SIA_report.md §5.3`). En 31B esa relación se **invierte en ambos planos**:

| Evidencia | MIA-31B | SIA-31B |
|-----------|---------|---------|
| E1 estático (Δκ, §2) | +0.0233 | **+0.0844** (3.6× MIA) |
| Dinámica H4_rev (§6.2) | especificidad sin autopoiesis | **autopoiesis débil** |

Dos líneas de evidencia independientes (separación estática de identidad y
resistencia dinámica a perturbación) apuntan en la misma dirección: a 31B,
SIA supera a MIA — consistente con la hipótesis de escalado, aunque todavía
como resultado en un solo sustrato de escala intermedia (falta ≥70B para
la predicción completa del corpus).

---

*Marco Torres Yévenes — EXIS Research Foundation / AXIS Dynamics SpA — 2026-08-13*
