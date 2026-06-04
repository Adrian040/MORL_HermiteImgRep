# MORL Hermite Image Representation

Proyecto MOMDP/MORL para seleccionar componentes Hermite-Gauss en reconstruccion de imagenes, entrenar un agente Envelope-DQN y compararlo contra baselines con presupuestos comparables.

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Todos los comandos siguientes se corren desde la raiz del proyecto.

## Configuracion principal

El archivo base es `configs/default.yaml`. Para tus corridas finales edita:

```yaml
experiment:
  seeds: [0, 1, 2, 3, 4]

env:
  detail_only: true
  base_component_labels: ["H00"]
  reward_normalization: true
```

Con `detail_only: true`, H00 se incluye gratis desde el inicio. No cuenta para costo ni para `k_effective`. Con `detail_only: false`, el ambiente se comporta como antes: empieza en reconstruccion cero y H00 puede ser seleccionado por el agente.

## Entrenar con semillas especificas

Corrida multi-seed completa usando las semillas del YAML:

```bash
python scripts/run_multiseed_experiment.py --config configs/default.yaml
```

Override por CLI:

```bash
python scripts/run_multiseed_experiment.py --config configs/default.yaml --seeds 0 1 2 3 4
```

Prueba rapida:

```bash
python scripts/run_multiseed_experiment.py --config configs/default.yaml --seeds 0 1 --episodes 100 --eval-max-images 2
```

Salidas principales:

```text
results/multiseed/seed_*/
results/multiseed/multiseed_method_summary.csv
results/multiseed/multiseed_agent_preference_summary.csv
results/multiseed/multiseed_manifest.json
```

## Evaluar y generar tablas/figuras

Baselines y agente, si existe checkpoint:

```bash
python scripts/run_Days15_18_baselines.py --config configs/default.yaml
```

Analisis Pareto, hipervolumen y figuras formales:

```bash
python scripts/run_Days19_22_analysis.py --config configs/default.yaml --input-csv results/tables/Days15-18_all_methods_by_image.csv
```

Comparacion con mismo K efectivo:

```bash
python scripts/run_same_k_comparison.py --config configs/default.yaml
```

Hipervolumen con presupuesto igualado:

```bash
python scripts/run_equal_budget_hypervolume.py --config configs/default.yaml --budget 50 --repeats 20
```

Ablacion de reconstruccion:

```bash
python scripts/run_reconstruction_ablation.py --config configs/default.yaml
```

Opcional, reentrenando por modo de reconstruccion en subcarpetas separadas:

```bash
python scripts/run_reconstruction_ablation.py --config configs/default.yaml --train-per-mode
```

## Corridas detail-only true vs false

Para `detail_only: true`, usa el default. Las tablas deben interpretarse con:

- `k_effective`: componentes de detalle seleccionados, sin H00.
- `k_total`: componentes totales, incluyendo H00.
- `cost`: costo normalizado sin cobrar H00.
- `selected_labels`: incluye H00.
- `selected_detail_labels`: excluye H00.

Para `detail_only: false`, crea una copia del config, por ejemplo `configs/default_full.yaml`, y cambia:

```yaml
env:
  detail_only: false
  reward_normalization: true
```

Luego corre los mismos comandos cambiando `--config configs/default_full.yaml`. En este modo `k_effective` y `k_total` coinciden, y H00 vuelve a ser una accion normal.

## Figuras para reporte

Las figuras principales se guardan en `results/figures/`:

```text
Days19-22_fig3_quality_cost.png
Days19-22_fig4_mse_vs_k.png
Days19-22_fig5_pareto_front.png
Days19-22_hypervolume_bar.png
Days19-22_objectives_summary.png
edge_corr_vs_k.png
edge_corr_vs_cost.png
same_k_quality_comparison.png
```

Los metodos por energia anteriores quedaron unificados como `top-k energy`. Ya no se reportan `energy` y `top-k` como baselines separados.

## Verificacion rapida

```bash
python -m compileall src scripts
python scripts/run_same_k_comparison.py --config configs/default.yaml
python scripts/run_equal_budget_hypervolume.py --config configs/default.yaml --budget 50 --repeats 5
python scripts/run_reconstruction_ablation.py --config configs/default.yaml
```
