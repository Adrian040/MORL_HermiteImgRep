# Days15-18 — Baselines obligatorios

Este patch agrega los baselines del Día 4 del plan compacto: selección aleatoria, selección por energía, top-k y greedy por mejora inmediata. También puede evaluar el agente Envelope-DQN de Days9-14 si existe el checkpoint.

## Archivos agregados

```text
src/baselines.py
src/evaluate.py
src/days15_18_plots.py
scripts/run_Days15_18_baselines.py
tests/test_Days15_18_baselines.py
configs/Days15-18_baselines.yaml
configs/Days15-18_smoke.yaml
README_Days15-18.md
```

No se reemplaza `env_hermite_momdp.py`, `env_utils.py`, `default.yaml` ni los scripts de Days9-14.

## Prueba rápida

```bash
cd hermite_morl_day1
python tests/test_Days15_18_baselines.py
python scripts/run_Days15_18_baselines.py --config configs/Days15-18_smoke.yaml --skip-agent
```

## Corrida principal

```bash
python scripts/run_Days15_18_baselines.py --config configs/Days15-18_baselines.yaml
```

Si todavía no existe el checkpoint de Days9-14, el script ejecuta solo los baselines y avisa que no evaluó el agente. Para forzar solo baselines:

```bash
python scripts/run_Days15_18_baselines.py --config configs/Days15-18_baselines.yaml --skip-agent
```

## Salidas principales

```text
results/tables/Days15-18_random_baseline.csv
results/tables/Days15-18_energy_baseline.csv
results/tables/Days15-18_topk_baseline.csv
results/tables/Days15-18_greedy_baseline.csv
results/tables/Days15-18_baselines_by_image.csv
results/tables/Days15-18_baselines_summary.csv
results/tables/Days15-18_all_methods_by_image.csv
results/tables/Days15-18_all_methods_summary.csv
results/figures/Days15-18_ssim_vs_k_baselines.png
results/figures/Days15-18_mse_vs_k_baselines.png
results/figures/Days15-18_visual_baseline_comparison.png
results/Days15-18_manifest.json
```
