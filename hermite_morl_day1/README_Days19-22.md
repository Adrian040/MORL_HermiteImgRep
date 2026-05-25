# Days19-22 — Pareto, hipervolumen, figuras finales y ablación

Este patch agrega el análisis cuantitativo y visual posterior a los baselines Days15-18.
No reemplaza el ambiente, las representaciones Hermite ni los scripts de entrenamiento/base.

## Archivos agregados

```text
src/days19_22_analysis.py
src/days19_22_plots.py
src/days19_22_cifar_utils.py
scripts/run_Days19_22_analysis.py
scripts/run_Days19_22_cifar_smoke.py
scripts/run_Days19_22_ablation.py
tests/test_Days19_22_analysis.py
configs/Days19-22_analysis.yaml
configs/Days19-22_cifar_smoke.yaml
configs/Days19-22_ablation.yaml
configs/Days19-22_ablation_smoke.yaml
README_Days19-22.md
```

## Prueba mínima

```bash
cd hermite_morl_day1
python tests/test_Days19_22_analysis.py
```

## Smoke con CIFAR o fallback tipo CIFAR

```bash
python scripts/run_Days19_22_cifar_smoke.py
```

El script intenta usar CIFAR-10 local mediante `torchvision.datasets.CIFAR10(download=False)`. Si no existe en caché, genera una muestra compatible tipo CIFAR desde `skimage`, sin descargar datos.

## Análisis principal después de Days15-18

Primero genera los baselines:

```bash
python scripts/run_Days15_18_baselines.py --config configs/Days15-18_baselines.yaml
```

Luego ejecuta:

```bash
python scripts/run_Days19_22_analysis.py \
  --input-csv results/tables/Days15-18_all_methods_by_image.csv \
  --prefix Days19-22
```

## Ablación N=2 vs N=3

Corrida rápida sin agente, útil para validar rutas:

```bash
python scripts/run_Days19_22_ablation.py --config configs/Days19-22_ablation_smoke.yaml --skip-training --skip-agent
```

Corrida principal. Por default entrena dos modelos si no se indica lo contrario, uno para N=2 y otro para N=3, luego corre baselines/evaluación y resume resultados:

```bash
python scripts/run_Days19_22_ablation.py --config configs/Days19-22_ablation.yaml
```

## Salidas principales

```text
results/tables/Days19-22_objective_space.csv
results/tables/Days19-22_pareto_front_global.csv
results/tables/Days19-22_pareto_front_by_method.csv
results/tables/Days19-22_hypervolume_summary.csv
results/tables/Days19-22_method_summary.csv
results/tables/Days19-22_table1_method_summary_report.csv/.md/.tex
results/figures/Days19-22_fig3_quality_cost.png
results/figures/Days19-22_fig4_mse_vs_k.png
results/figures/Days19-22_fig5_pareto_front.png
results/figures/Days19-22_hypervolume_bar.png
results/figures/Days19-22_objectives_summary.png
```
