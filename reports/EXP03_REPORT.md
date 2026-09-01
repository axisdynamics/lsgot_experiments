# Exp. 0.3 — piso de ruido split-half, aplicado a r (E2)

**Fecha:** 2026-09-01
**Continúa:** `EXP02_REPORT.md` §3, donde `r = W₁_angular/W₁_euclidean` quedó
computado pero marcado explícitamente como no interpretable en valor
absoluto por un desajuste de unidades (radianes acotados en [0,π] vs
distancia euclidiana cruda en 2560 dimensiones).
**Script:** `scripts/exp03_splithalf_null.py` — null de permutación
(axis ∪ vanilla, 1000 particiones aleatorias en dos mitades de n=20,
mismo tamaño de muestra que la comparación real; ver el docstring del
script para por qué no se usó el split-half de una sola condición que
describe la guía literalmente).
**Resultados crudos:** `reports/exp03_base.json`, `reports/exp03_instruct.json`.

---

## 1. Método: normalizar cada componente de r contra su propio piso de ruido

```
W1_eucl_norm = W1_euclidean(axis, vanilla) / mediana(null_euclidean)
W1_ang_norm  = W1_angular(axis, vanilla)   / mediana(null_angular)
r_normalizado = W1_ang_norm / W1_eucl_norm
```

Numerador y denominador quedan en la misma unidad adimensional ("múltiplos
del piso de ruido"), en vez de mezclar radianes con distancia cruda.

## 2. Resultado — se invierte respecto de lo reportado en EXP02_REPORT.md §3

| Modelo | W₁ eucl. (×mediana null) | W₁ ang. (×mediana null) | r normalizado |
|---|---|---|---|
| base | 1.39× | 1.35× | **0.971** |
| instruct | 2.40× | 2.67× | **1.113** |

Ambos, en ambos modelos, están muy por encima de su propio piso de ruido
(percentil 100 del null en las 4 celdas — la separación axis-vanilla es
real en las dos dimensiones, angular y euclidiana, en base y en instruct).

**Con la razón cruda (sin normalizar), reporté la vuelta pasada
`r(base)=0.0071 > r(instruct)=0.0063`** — dirección consistente con la
predicción de la guía. **Con la razón normalizada, el orden se invierte:
`r(instruct)=1.113 > r(base)=0.971`.** El número crudo estaba dominado por
el desajuste de escala que ya había señalado como sospechoso, no por una
señal real — al corregirlo, la lectura cambia de signo, no solo de
magnitud.

## 3. Qué dice esto sobre la hipótesis central de la v3

No apoya, en este único par, la lectura literal de "instruction tuning
relocaliza la señal de dirección a magnitud" como *reemplazo* (dirección
baja, magnitud sube). Lo que se ve es distinto:

- El canal euclidiano (magnitud) sí crece fuerte en términos absolutos de
  base a instruct — coherente con Δ_norm pasando de marginal (p=0.063) a
  casi total (Mann-Whitney p=6.8×10⁻⁸, Exp. 0.2 §1).
- Pero el canal angular (dirección) **también** crece, proporcionalmente
  un poco más (2.67× vs 2.40× su propio piso de ruido) — no se queda
  atrás ni colapsa.
- Es decir: en este par, instruction tuning no parece mover la señal
  *de* dirección *a* magnitud — parece **amplificar ambos canales**, con
  el euclidiano creciendo en términos absolutos mucho más (Δ_norm es
  enorme) pero sin que el angular se quede plano o retroceda en términos
  relativos a su propio ruido.

**Salvedad obligatoria: n=1 par.** Esto puede ser el patrón real, un
artefacto de esta familia específica (Gemma), o ruido de un solo punto de
datos — exactamente el mismo problema estructural que documenta
`DECISION_FASES_2-5.md` §1 para todo lo demás. No se puede generalizar el
título de la v3 ("relocates... from direction to magnitude") a partir de
esto; si algo, este resultado es un motivo adicional para no comprometerse
con ese framing en el título/abstract (Fase 5, punto 1) hasta tener más
pares, no menos.

## 4. Actualización de checklists

`DECISION_FASES_2-5.md` §5, ítem `r` — actualizar:

- [x] `r` normalizado contra piso de ruido split-half — resuelto
  metodológicamente (`exp03_splithalf_null.py`); **el resultado en el
  único par disponible no confirma el framing "dirección→magnitud" de la
  v3** — ver §3 arriba. Antes de escribir Fase 5 punto 1 (título/abstract)
  con ese framing, hace falta al menos un segundo par que lo sostenga.

*Marco Torres Yévenes — EXIS Research Foundation / AXIS Dynamics SpA — 2026-09-01*
