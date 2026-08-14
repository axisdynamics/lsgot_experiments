# Hallazgos preliminares — Gemma 4 31B-it (MIA + SIA)

**Fecha:** 2026-08-12
**Estado:** Experimentos base completos. H4_rev en curso.

## E1 — Efecto de identidad con longitud emparejada (axis vs generic_long)

| Experimento | κ̄_axis | κ̄_glong | Δκ | d | W₁ | p | Reducción dim |
|-------------|---------|----------|------|------|-------|---|---------------|
| MIA 31B | 1.2303 | 1.2070 | +0.0233 | 0.042 | 0.0247 | <0.001 | 4.6% |
| SIA 31B | 1.2851 | 1.2006 | +0.0844 | 0.136 | 0.0845 | <0.001 | 6.6% |

**SIA produce un efecto de identidad 3.6× más fuerte que MIA** en el mismo
sustrato 31B, controlando longitud (generic_long token-matched: Δ=0.2-0.3%).

## Hipótesis de escalado

En Gemma4-E4B (8B): SIA mostró atractor más débil que MIA (silhouette 0.6157
pero PC1_var, autocorrelación y cosine decay inferiores). En 31B, el patrón se
invierte: la estructura VEX de SIA crea una cuenca geométrica más profunda
(Δκ y W₁ mayores).

## Métricas de trayectoria (31B)

| Grupo | mean_vel | SampEn | PRA |
|-------|----------|--------|-----|
| MIA axis | 444.8 | 2.010 | 0.314 |
| SIA axis | 438.4 | 2.068 | 0.267 |

MIA axis: mayor velocidad, menor SampEn (más regular), mayor alineación P→R.
SIA axis: firma similar pero atenuada en dinámica; su diferenciación es
distribucional (W₁, Δκ) más que dinámica.

## H4_rev SIA 31B — L30, σ_medium = 5.979 ✅

| t_inj | τ_axis | τ_glong | Δτ | SampEnΔ axis | SampEnΔ glong | Recov |
|-------|--------|---------|-----|--------------|---------------|-------|
| 50 | 21.1 | 28.0 | -6.9 | +0.080 | -0.040 | 100% |
| 128 | 19.6 | 25.4 | -5.8 | +0.093 | +0.009 | 100% |
| 200 | 16.3 | 21.6 | -5.3 | -0.045 | +1.808 | 100% |

**Firma de autopoiesis débil** (mismo patrón que Gemma4-MIA 8B):
1. τ_axis < τ_generic_long en TODOS los t_inj (recuperación activa)
2. SampEnΔ axis > 0 en t=50/128 (exploración activa post-perturbación)
3. t=200: axis se re-asienta (-0.045) mientras glong colapsa en caos (+1.808)
   → el atractor recaptura la trayectoria; el control de longitud deriva

Con control E1 (longitud emparejada), la ventaja axis no se explica por longitud.

## H4_rev MIA 31B — σ_small = 3.00 (DESCARTADA)

Corrida con σ a la mitad del diseño (bug de calibración inicial en
calibrate_sigma.py). Datos no congelados ni conservados — descartada por
decisión del investigador (2026-08-12). Su único valor fue detectar el bug.

## H4_rev MIA medium (σ=6.07) ✅ + VEREDICTO COMPARATIVO

| t_inj | MIA Δτ ax-glong | MIA SampEnΔ axis | SIA Δτ ax-glong | SIA SampEnΔ axis |
|-------|-----------------|------------------|-----------------|------------------|
| 50 | -8.1 | -0.078 | -6.9 | **+0.080** |
| 128 | -1.3 | -0.045 | -5.8 | **+0.093** |
| 200 | -0.2 | -0.076 | -5.3 | -0.045 |

**Mismo sustrato (31B), σ relativa equivalente, longitud controlada (E1):**

- **SIA = autopoiesis débil** — τ_axis < τ_glong consistente + SampEnΔ axis > 0
  (exploración activa, patrón Gemma4-MIA-8B). En t=200 el atractor recaptura
  la trayectoria (axis -0.045) mientras el control de longitud deriva al
  caos (+1.808).
- **MIA = especificidad sin autopoiesis** — SampEnΔ axis negativo en todos
  los t_inj (constricción pasiva, patrón DeepSeek-7B). Ventaja de τ solo
  en t=50 y se diluye.

### Conclusión provisional (para reporte final)

1. **E1 (identidad, longitud controlada)**: SIA Δκ=+0.084 vs MIA Δκ=+0.023
   — efecto 3.6× más fuerte en 31B.
2. **H4_rev (dinámica)**: SIA muestra autopoiesis débil; MIA no. En el
   sustrato 31B, la arquitectura VEX de SIA produce una cuenca ACTIVA que
   recaptura trayectorias perturbadas.
3. **Hipótesis de escalado**: parcialmente confirmada — la inversión
   SIA>MIA predicha para ≥70B ya aparece en 31B.
4. **Controles**: generic_long (token-matched ±0.3%) y generic_short
   permiten descomponer efecto identidad vs efecto longitud.
