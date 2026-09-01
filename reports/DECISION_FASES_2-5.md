# Decisión Fases 2-5 — GUIA_EXPERIMENTAL_v3.md a la luz de lo ya corrido

**Fecha:** 2026-09-01
**Insumo:** `~/Descargas/GUIA_EXPERIMENTAL_v3.md` (ruta a la v3 de `lsgot_3`,
arXiv:2607.09842) contra el estado real de este repo (Fase 0 y Fase 1
cerradas) y los hallazgos de `LSGOT_v4` (panel Gemma-4-31B-it, identidad ×
densidad de restricción — repo hermano `lsgot_experiments_v4`).

**Qué es esto:** no es un paper ni un re-análisis nuevo. Es el mapa de qué
ítem de cada fase sigue vigente tal cual, cuál queda resuelto por trabajo ya
hecho (en este repo o en LSGOT_v4), cuál queda descartado por un hallazgo
posterior, y cuál sigue necesitando GPU que nadie corrió todavía — para que
la próxima decisión de invertir tiempo/GPU se tome con el cuadro completo,
no fase por fase como aparecen en la guía.

---

## 0. El hecho que reordena todo lo que sigue

El criterio de decisión de Exp. 0.2 de la guía es explícito: *"Si ORC-W₁ no
supera a los baselines → se reporta, y ORC pasa a apéndice."* Ese criterio
**ya se disparó, dos veces**:

- `LSGOT_v4/evidence/REPORTE_FASE0.md` — panel Gemma-4-E4B/-it,
  DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-7B-Instruct: la estadística de
  curvatura (mal atribuida como Ollivier-Ricci; es Forman-Ricci, ver
  `LSGOT_v4/evidence/QUE_SOBREVIVE_REPORT.md` §3) **no supera baselines
  simples en 12/12 comparaciones**, y cae dentro del ruido split-half para
  la comparación identidad-vs-genérico específicamente.
- `LSGOT_v4/evidence/CURVATURE_SELF_AUDIT_REPORT.md` — mismo protocolo
  corrido sobre el panel Gemma-4-31B-it: **4/4**, mismo resultado.

**Importante — esto NO cierra automáticamente Exp. 0.2 para esta guía.** La
guía define dos estadísticos primarios nuevos para la v3 — `Δ_norm` (índice
de norma) y `r = W₁_angular/W₁_euclidean` (ratio, no la curvatura discreta
misma) — que **no son la misma construcción** que el Δκ/W₁ de curvatura ya
auditado. El fallo de curvatura no se hereda automáticamente a estos dos.
Exp. 0.2 sigue pendiente, pero ahora aplicado específicamente a `Δ_norm` y
`r`, no a Δκ. Ver §2 más abajo.

Lo que sí queda resuelto: **Δκ/W₁ de curvatura (la métrica de Fase 1 y de
`INTERIM_FINDINGS.md`) ya no puede presentarse como estadístico primario en
ninguna versión futura de este paper** — pasa a apéndice/evidencia
secundaria, con la misma redacción que ya usa `LSGOT_v4/paper/lsgot_4.md`
§3.4.

---

## 1. Fase 2 — Pares base/instruct: estado real vs objetivo

| Familia | Objetivo guía | Estado real | Fuente |
|---|---|---|---|
| Gemma (8B) | E4B / E4B-it | ✅ **completo**, MIA, 4 condiciones | `reports/reporte_fase1.md` §2, cerrado 2026-08-13 |
| Gemma (31B) | base / it | ⚠️ **parcial** — base: 3/4 grupos (axis, generic_long, generic_short; falta `vanilla`, pod cayó por spot preemption); it: existe extracción completa pero en protocolo distinto (SIA `sia_extended_v5` de LSGOT_v4, no el mismo MIA/H4_rev de este repo) | `reports/reporte_fase1.md` §5; `LSGOT_v4/data/` |
| Qwen2.5-7B | base + instruct | ❌ solo el instruct existe (MIA); la base no se extrajo nunca | `reports/reporte_fase1.md` §2 (solo fila instruct) |
| Llama-3.1-8B | ambos | ❌ ninguno extraído | — |
| Mistral-7B-v0.3 | ambos | ❌ ninguno extraído | — |
| OLMo-2-7B | ambos | ❌ ninguno extraído | — |

**Pares completos y limpios hoy: 1 (Gemma 8B).** El par 31B no es
directamente utilizable como está — el lado base es MIA/H4_rev y el lado it
es SIA/`sia_extended_v5`; emparejarlos exige o bien re-extraer `vanilla` en
31B-base y aceptar el descalce de protocolo, o re-extraer el lado it bajo
MIA (no se ha hecho ninguna de las dos cosas).

**Decisión:** la guía misma define su propio piso de significancia — 5
pares para p=0.031 en el test de signos pareado, y su propia tabla de
criterios ya contempla el escenario de quedar muy por debajo: *"≤2/5 → El
efecto es específico de Gemma. Paper distinto y más corto... Esto no es un
fracaso."* Con 1 par completo estamos debajo incluso de ese piso de
fracaso-aceptable. **No hay forma de "terminar" Fase 2 con información ya
existente — cada familia faltante (Qwen-base, Llama, Mistral, OLMo) requiere
extracción nueva, ~30-60 GPU-h cada una según el presupuesto de la propia
guía.** Esto no es una decisión metodológica, es un hecho de qué se corrió
y qué no.

Ítems de Fase 2 que quedan en espera, sin poder resolverse solo con
análisis:
- "Añade: guarda también el estado de cada 4ª capa" — instrucción para
  extracciones *futuras*; no se puede aplicar retroactivamente a lo ya
  extraído.
- "Añade: guarda el token id generado en cada paso" — mismo caso.
- Orden de ejecución recomendado (Qwen base → OLMo → Llama/Mistral) — sigue
  siendo el orden correcto *si* se decide invertir el presupuesto de GPU-h
  restante; no cambia por nada de lo encontrado en LSGOT_v4.

---

## 2. Fase 3 — Análisis pareado

### 3.1 Tabla principal / 3.2 Slopegraph

Ambas requieren ≥2 filas por familia (base + instruct) en ≥3 familias para
decir algo distinto de "un solo punto de datos". Con el estado real de §1,
**solo se puede llenar la fila Gemma 8B** hoy. No hay tabla de 5 familias
ni slopegraph de 5 segmentos posible con datos existentes — sería
fabricar filas vacías o inventar un panel que no existe.

### 3.3 Tests

- **Test de signos pareado, una cola:** matemáticamente no arroja nada
  informativo con 1 familia (la propia tabla de la guía empieza en n=3,
  p=0.125, ya no-significativo). **Descartado hasta tener ≥3 pares reales.**
- **Test de permutación por familia:** aplicable a Gemma 8B ya mismo — el
  protocolo ya existe (`scripts/analyze_dynamics.py` de este repo). Pero
  falta correrlo específicamente sobre `Δ_norm` y `r`, no sobre Δκ (ver
  siguiente punto).
- **BH-FDR, Bootstrap CI:** aplicables a Gemma 8B, sin cambios de alcance.

### El paso que falta antes de cualquier tabla: Exp. 0.2 sobre `Δ_norm` y `r`

Ni este repo ni LSGOT_v4 corrieron nunca los baselines simples (distancia
de centroides, MMD, CKA lineal, probe AUC) contra estos dos estadísticos
específicos — solo contra Δκ/W₁ de curvatura, que es una construcción
distinta. **Antes de escribir la tabla 3.1 hay que correr Exp. 0.2 sobre
`Δ_norm` y `r`, aunque sea solo en Gemma 8B.** Si el patrón de Fase 0 se
repite (baselines ganan), `r` (el diagnóstico angular/euclídeo, el aporte
metodológico central de la v3) queda igual de debilitado que Δκ lo estuvo
para `lsgot_3`. Si no se repite, es la justificación de método que hoy
falta — vale la pena saberlo antes de invertir en Qwen/Llama/Mistral/OLMo,
no después.

### 3.4 Robustez (apéndice)

Vigente sin cambios para lo que ya existe (Gemma 8B); se mueve a apéndice
tal cual la guía indica, no hay nada que recalcular hasta tener más pares.

---

## 3. Fase 4 — Opcionales de alto valor

| Ítem | Estado |
|---|---|
| **4.1 Perfil por capas** | Depende de haber guardado cada 4ª capa en la extracción — instrucción nueva de la guía, no se aplicó retroactivamente a lo ya extraído (Fase 1 cerró antes de que existiera este requisito). **Pendiente, requiere re-extracción o verificación de si algún log intermedio sobrevivió.** |
| **4.2 Ablación semántica del template** | **Ya resuelto — no rehacer.** Esto es literalmente `axis_pec_only` en `LSGOT_v4`: descompone `axis` quitando la arquitectura de reglas (triggers, filtro de prioridad, jerarquía de bloques) y mide qué carga la señal. Resultado ya obtenido: la identidad (v̂) no cambia al quitar la arquitectura de reglas (p=0.229, `LSGOT_v4/paper/lsgot_4.md` §3.5) — la señal la carga el contenido identitario, no el andamiaje de reglas. Va más allá de lo que pedía 4.2 (bloque por bloque) pero responde la misma pregunta con más precisión. |
| **4.3 Segunda familia de prompts de identidad** | **Ya resuelto — no rehacer.** `soul_md_corto`/"Witness" (`LSGOT_v4/paper/lsgot_4.md` §3.8) es exactamente esto: una plantilla de identidad de otro autor, otro proyecto, sin ninguna línea de herencia con `axis` — y converge con el perfil de identidad en las tres señales de v̂, recuperación y fidelidad de ruta. Cumple el objetivo declarado de 4.3 ("elimina de golpe la lectura de que el paper existe para validar el producto del autor"). |
| **4.4 Sweep de N** | No corrido en ningún repo (LSGOT_v4 usa N=256 fijo en todo su panel). **Sigue pendiente**, barato según la guía (N=128 y N=512 solo en el par Gemma), pero requiere GPU nueva. |

---

## 4. Fase 5 — Reescritura

| Ítem | Estado |
|---|---|
| H1 reformulada con `generic-long` | Vigente — dato ya existe (Fase 1), redacción pendiente cuando se fije alcance final. |
| H2 en versión pareada | Bloqueado por Fase 2/3 (§1-2 arriba) — solo describible para Gemma 8B por ahora. |
| **H3 eliminada** | **Ya confirmado, no hace falta re-derivarlo.** `LSGOT_v4/paper/lsgot_4.md` §3.7: perturbación a lo largo de v̂ vs ortogonal, null result en 6/6 comparaciones (p>0.14). La guía ya anticipaba eliminar H3 antes de que este dato existiera; ahora hay evidencia directa (aunque corrida solo en Gemma-31B, el mismo alcance real que tendría este proyecto de todos modos) que lo confirma. |
| Nueva sección de baselines | Pendiente de Exp. 0.2 sobre `Δ_norm`/`r` (§2 arriba) — no se puede escribir sin ese resultado. |
| §5 limitaciones se acorta sola | Parcial: "single identity template" desaparece (4.3 resuelto); "narrow regime panel" **no desaparece** — sigue siendo 1 familia real, no 5; "length-matched contradictorio" desaparece (Fase 1 cerrada). |
| Metadata arXiv, título/abstract | Acciones editoriales, no dependen de datos — ejecutables cuando se decida el alcance final (Gemma-only vs esperar más extracción). |

---

## 5. Checklist final de la guía — releído contra lo de arriba

**Confounds cerrados**
- [x] Longitud de prompt igualada por construcción (`generic-long`, Fase 1) — cerrado
- [ ] Identidad del primer token controlada con **token forzado** — la versión proxy (léxica) corrió en `LSGOT_v4` (§3.4) y no revierte nada; el forward-pass definitivo con token forzado **sigue sin correr**, en ningún repo
- [x] Longitudes reales de tokenización reportadas por modelo — cerrado, tabla en `reporte_fase1.md` §1
- [ ] Cuantización — no se usó en las corridas hechas hasta ahora (verificado por diseño, sin necesidad de control adicional)

**Justificación de método**
- [ ] Baselines simples sobre `Δ_norm`/`r` — **no corrido**, distinto del Δκ ya auditado (§2 arriba)
- [ ] Piso de ruido split-half para `Δ_norm`/`r` — mismo caso, no corrido para estos dos estadísticos específicamente (sí existe para Δκ)
- [x] Claim de novedad de Δκ/curvatura ya suavizado — resuelto en `LSGOT_v4`, aplica igual acá

**Diseño**
- [ ] ≥5 pares base/instruct — **1 de 5**
- [ ] Test de signos pareado — no ejecutable todavía (§3.3 arriba)
- [ ] Pre-registro `PREREG.md` con fecha y commit hash — no existe todavía en este repo
- [x] H3 eliminada — confirmado por `LSGOT_v4`
- [x] DeepSeek movido a apéndice exploratorio — ya tratado así en `reporte_fase1.md`

**Framing**
- [ ] Título/abstract sin "identity" — pendiente de redacción final
- [ ] Al menos una segunda familia de prompt de identidad — **resuelto** (4.3 = `soul_md_corto`)
- [ ] Comments de arXiv sin "Working draft" — pendiente
- [x] Repo público con scripts, prompts y hashes — ya existe (`lsgot_experiments`, público)

**La pregunta de control de la guía** ("¿puede un lector hostil desarmar el
claim central en 30 segundos?"): con 1 de 5 pares, sí, en menos de 30
segundos — "esto es un case study de un solo modelo, no una comparación
entre familias". No está lista para ese framing; sí podría estarlo para el
framing de repliegue que la propia guía ya previó.

---

## 6. Recomendación

No hay ningún ítem de Fase 2-5 que se pueda "terminar" hoy sin GPU nueva
**si el objetivo sigue siendo el paper de 5 familias tal como está escrito
en la guía**. Lo que sí se puede cerrar con lo que ya existe:

1. Correr Exp. 0.2 (baselines) sobre `Δ_norm` y `r`, en Gemma 8B — barato,
   ya extraído, decide si el diagnóstico angular/euclídeo tiene piso
   metodológico antes de invertir en más extracción.
2. Escribir Fase 3 §3.1-3.2 como **case study de una familia** (Gemma 8B,
   con Gemma 31B como robustez parcial si se resuelve el descalce de
   protocolo), citando explícitamente el escenario "≤2/5" que la propia
   guía ya definió como resultado válido, no como fracaso.
3. Dejar Qwen-base/Llama/Mistral/OLMo como un ítem de decisión aparte —
   son ~210 GPU-h de las ~240 estimadas para Fases 1+2, y es la única pieza
   de todo este documento que ninguna cantidad de análisis puede sustituir.

*Marco Torres Yévenes — EXIS Research Foundation / AXIS Dynamics SpA — 2026-09-01*
