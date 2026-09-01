# Exp. 0.2 — Baselines vs Δ_norm (E1) y r (E2), Gemma 8B base/instruct

**Fecha:** 2026-09-01
**Alcance:** único par base/instruct con datos completos hoy (Gemma4-E4B /
Gemma4-E4B-it, MIA, n=20 por condición). Ver
`reports/DECISION_FASES_2-5.md` §1 para por qué no hay más pares.
**Script:** `scripts/exp02_baselines_norm_r.py` — protocolo de permutación
n=1000, seed=42, sobre v₁ (`embeddings[:, 0, :]`, el hidden state del
primer paso de generación).
**Resultados crudos:** `reports/exp02_base.json`, `reports/exp02_instruct.json`.

---

## 1. E1 — Δ_norm(axis, generic_long): resultado limpio, y confirma la predicción central

| Modelo | Δ_norm | perm. p | Mann-Whitney p | Cohen's d |
|---|---|---|---|---|
| base | +6.67 | 0.063 (n.s.) | 0.047 | +0.59 (medio) |
| instruct | **−78.42** | **<0.001** | **6.8×10⁻⁸** (U=0, separación total) | **−5.45** (enorme) |

**Δ_norm(instruct) − Δ_norm(base) = −85.1 → signo negativo**, exactamente
la predicción central de la guía para este contraste ("el prompt de
identidad deprime la norma, y lo hace específicamente bajo instruction
tuning"). Es 1 de los 5 pares que pide el pre-registro — el signo va en la
dirección correcta, pero un solo par no alcanza significancia por test de
signos (mínimo n=3 para p=0.125).

**Base es ambiguo, no "≈0" limpio.** La guía predice Δ_norm≈0 en base; acá
sale un efecto medio, de signo positivo (no negativo), con p=0.063 —
border-line, no un cero limpio. No contradice la predicción central (que
es sobre el signo de la *diferencia entre* instruct y base, no sobre el
valor absoluto en base), pero tampoco es la ausencia total de efecto que
el texto de la guía sugiere.

## 2. Baselines vs Δ_norm — ¿el índice de norma detecta algo que los simples no?

| Baseline | base | instruct |
|---|---|---|
| Centroide (perm. p) | **0.001** | **0.001** |
| MMD-RBF (perm. p) | **0.001** | **0.001** |
| 1−CKA lineal (pareado) | 0.136 | 0.623 |
| Probe lineal AUC (GroupKFold) | **1.000** | **1.000** |
| Mann-Whitney sobre ‖v₁‖ | 0.047 | 6.8×10⁻⁸ |

**En `base`, los baselines ganan claramente.** Centroide y MMD detectan
separación con p=0.001 mientras Δ_norm apenas roza significancia (p=0.063,
Mann-Whitney p=0.047). El probe lineal separa perfecto (AUC=1.0). Esto
significa que en el modelo base, axis y generic_long sí son distinguibles
— pero mayormente por *forma/dirección* de la representación, no por la
norma sola. Δ_norm como estadístico único **no** habría detectado esta
separación con la misma confianza que un centroide o un probe.

**En `instruct`, Δ_norm por sí solo ya es tan fuerte como los baselines**
(Mann-Whitney p=6.8×10⁻⁸, más extremo que el propio umbral de permutación
de 0.001 de los demás) — acá la norma no es un detector débil, es *el*
efecto.

**Lectura combinada:** esto es consistente con, y agrega evidencia directa
a, la premisa completa del título de la v3 ("relocates persona conditioning
from direction to magnitude") — en base la señal está más repartida
(forma/dirección detectable por baselines, norma marginal); en instruct
se concentra casi enteramente en la norma. Con un solo par no se puede
generalizar, pero el patrón es exactamente el que motiva rediseñar el
paper alrededor de esta pregunta.

## 3. E2 — r = W₁_angular/W₁_euclidean: resultado computado, con un defecto de escala que hay que corregir antes de usarlo

> **Actualización — ver `EXP03_REPORT.md`:** la razón cruda de esta
> sección quedó normalizada contra un piso de ruido split-half, y **el
> orden entre base e instruct se invierte** una vez corregido el
> desajuste de escala. Los números de esta sección (§3) están acá por
> transparencia del proceso, no como resultado final — usar
> `EXP03_REPORT.md` para cualquier cita.

| Modelo | r | IC 95% (bootstrap) | W₁ angular (rad) | W₁ euclidiano |
|---|---|---|---|---|
| base | 0.00713 | [0.00696, 0.00730] | 0.706 | 99.01 |
| instruct | 0.00626 | [0.00608, 0.00643] | 1.242 | 198.37 |

**Problema real que encontré al revisar mi propia implementación, antes de
reportar esto como cerrado:** `r` tal como está definido en la guía —
razón directa entre una distancia angular (acotada en [0, π] ≈ 3.14) y una
distancia euclidiana cruda (aquí, ~100-200 en un espacio de 2560
dimensiones) — va a dar un número minúsculo **siempre**, sin importar
dónde viva realmente la señal, por puro desajuste de unidades. No es un
resultado interpretable como "el X% de la separación sobrevive al
normalizar la dirección" — ese no es el problema que r resuelve tal como
lo implementé.

**Lo único que sí es válido de esta tabla:** los intervalos de confianza
(bootstrap, resampleo pareado por prompt) de base e instruct **no se
solapan** — `r(base) > r(instruct)` de forma estadísticamente estable, no
por azar de muestreo. La dirección coincide con lo que predice la guía
(base retiene relativamente más señal direccional que instruct). Pero no
puedo afirmar con esta implementación cuánta señal vive en dirección vs
magnitud en términos absolutos, ni comparar la *magnitud* de r entre
modelos con confianza, porque W₁ euclidiano también cambia de escala entre
base (99) e instruct (198) — dos fuentes de variación mezcladas en un solo
número.

**Lo que falta antes de usar r en cualquier tabla del paper:** normalizar
W₁_angular y W₁_euclidean cada uno contra su propio piso de ruido
split-half (exactamente lo que la guía pide en Exp. 0.3, "aplícalo también
a los baselines de 0.2" — se aplica igual acá) antes de tomar el cociente.
Sin eso, r es un número calculado correctamente pero no interpretable en
su valor absoluto — lo dejo reportado tal cual, con esta salvedad
explícita, y no como una tabla lista para el paper.

---

## 4. Actualización del checklist de `DECISION_FASES_2-5.md`

- [x] Baselines simples sobre Δ_norm — **corrido**, Gemma 8B: baselines
  ganan en base, Δ_norm domina en instruct (§2)
- [x] Piso de ruido split-half sobre r — corrido en `EXP03_REPORT.md`:
  **el resultado se invierte** una vez normalizado (`r(instruct) >
  r(base)`, no al revés) — ver ese reporte, no lo repito acá para no
  duplicar un número que ya quedó obsoleto en cuanto se corrigió
- [x] Δ_norm(instruct) − Δ_norm(base): signo correcto (1/5 pares
  disponibles, no alcanza el test de signos con n=1)

*Marco Torres Yévenes — EXIS Research Foundation / AXIS Dynamics SpA — 2026-09-01*
