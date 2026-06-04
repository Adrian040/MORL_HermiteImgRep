# MORL Hermite-Gaussian Component Selection

Este repositorio formula la seleccion adaptativa de componentes Hermite-Gauss como un MOMDP/MORL. Los scripts historicos Days1-4, Days5-8, Days9-14, Days15-18 y Days19-22 se conservan; la configuracion nueva vive principalmente en `configs/default.yaml`.

## Comandos principales

```bash
# Entrenamiento multi-seed
python scripts/run_multiseed_training.py --config configs/default.yaml

# Entrenamiento directo con varias seeds
python scripts/train_Days9_14_envelope.py --config configs/default.yaml --seeds 0 1 2 3 4

# Agregacion multi-seed
python scripts/aggregate_multiseed_results.py --results_dir results/seeds

# Evaluacion principal
python scripts/run_full_evaluation.py --config configs/default.yaml

# Analisis justo: hipervolumen equal-budget y comparacion same-K
python scripts/run_fair_analysis.py --config configs/default.yaml

# Comparacion same-K solamente
python scripts/run_same_k_analysis.py --config configs/default.yaml

# Ablacion de reconstruccion
python scripts/run_reconstruction_ablation.py --config configs/default.yaml

# Barrido Hermite por sigma, kernel_size y orden maximo
python scripts/run_hermite_sweep.py --config configs/default.yaml

# Evaluacion externa opcional DFU
python scripts/evaluate_external_dfu.py --config configs/default.yaml --raw_dir data/external_dfu
```

## Defaults defendibles

`detail_only_h00_free=true` esta activado por default. En este modo H00 se incluye desde `reset()` como componente base gratuito; el agente conserva el mismo espacio de acciones, pero seleccionar H00 de nuevo se trata como accion repetida/no valida. Los resultados reportan `total_k` y `paid_k`; por default la parsimonia y el costo usan `paid_k`.

La recompensa sigue teniendo cuatro objetivos: mejora de MSE, mejora de SSIM, costo computacional y parsimonia. Cuando `reward.mode=relative`, las mejoras de MSE y SSIM se normalizan con respecto a las metricas iniciales del episodio.

`K` mide parsimonia: cantidad de componentes pagados. `Cost` mide costo computacional adicional por kernels mayores que `free_kernel_size=13`. Con `cost.mode=extra_kernel_flops`, kernels de tamano 9 y 13 tienen costo adicional cero, mientras que kernels mayores, como 17, aportan costo.

Las metricas `psnr`, `gradient_mse` y `edge_corr` son solo de evaluacion por default. No cambian la dimension de recompensa ni la arquitectura del agente.

## Nota conceptual

Este proyecto no implementa de forma completa la transformada Hermite orientada de Escalante-Ramirez con lattice local, steering, verificacion de consistencia y sintesis exacta. En su lugar, implementa una reconstruccion aproximada por banco de filtros Hermite-Gauss, adecuada para formular la seleccion adaptativa de componentes como un MOMDP. La version detail-only con H00 gratuito se introduce para evitar que el agente colapse a la componente de baja frecuencia y para estudiar mejor la seleccion de detalles, bordes y estructuras locales.

## Actualizacion: entrenamiento con banco Hermite mixto

Para entrenar un solo agente por seed usando todas las combinaciones de sigma y kernel dentro del banco de filtros, usa:

```bash
python scripts/run_multiseed_hermite_grid_training.py --config configs/default.yaml
```

Aunque el script conserva el nombre `grid` por compatibilidad, ya no entrena un modelo por configuracion. Ahora cada seed entrena un solo modelo con un banco mixto definido por:

```text
sigmas: [1.0, 1.5, 2.0]
kernel_sizes: [9, 13, 17]
max_order: 4
```

Con `max_order=4` hay 15 componentes Hermite unicos `(m,n)` con `m+n <= 4`. El agente ve 15 acciones de componentes mas la accion `STOP`, para un total de 16 acciones. Internamente, cada componente se evalua con las variantes sigma/kernel del banco mixto.

Por default entrena una corrida por seed en `experiment.seeds`. Las salidas quedan separadas en:

```text
results/mixed_hermite_multiseed/seed_0/
results/mixed_hermite_multiseed/seed_1/
...
```

Para una prueba corta antes de lanzar todo:

```bash
python scripts/run_multiseed_hermite_grid_training.py --config configs/default.yaml --seeds 0 --episodes 5 --eval-every 5
```

Para correr el experimento completo, pero controlando parametros desde CLI:

```bash
python scripts/run_multiseed_hermite_grid_training.py --config configs/default.yaml --seeds 0 1 2 3 4 --device auto --max-order 4 --sigmas 1.0 1.5 2.0 --kernel-sizes 9 13 17
```

Despues de que termine el entrenamiento completo, corre estos scripts en orden para obtener resultados de reporte:

```bash
# 1. Agregar resultados multi-seed del banco Hermite mixto
python scripts/aggregate_mixed_hermite_results.py --results-dir results/mixed_hermite_multiseed --output-root results

# 2. Evaluar baselines y, si existe checkpoint en la ruta default, agente principal
python scripts/run_full_evaluation.py --config configs/default.yaml

# 3. Analisis justo: hipervolumen equal-budget y comparacion same-K
python scripts/run_fair_analysis.py --config configs/default.yaml

# 4. Ablacion de reconstruccion direct_sum vs least_squares
python scripts/run_reconstruction_ablation.py --config configs/default.yaml

# 5. Barrido de sensibilidad Hermite para reconstruccion/baselines
python scripts/run_hermite_sweep.py --config configs/default.yaml

# 6. Evaluacion externa DFU opcional; no falla si no hay imagenes
python scripts/evaluate_external_dfu.py --config configs/default.yaml --raw_dir data/external_dfu
```

Tablas clave para el reporte:

```text
results/tables/mixed_hermite_multiseed_summary.csv
results/tables/mixed_hermite_preference_summary.csv
results/tables/mixed_hermite_ranked_configs.csv
results/tables/equal_budget_hypervolume_summary.csv
results/tables/same_k_comparison_summary.csv
results/tables/reconstruction_ablation_summary.csv
results/tables/hermite_sweep_summary.csv
```

Figuras clave:

```text
results/figures/equal_budget_hypervolume.png
results/figures/same_k_comparison.png
results/figures/reconstruction_ablation.png
results/figures/detail_only_example.png
results/figures/hermite_sweep_summary.png
```
