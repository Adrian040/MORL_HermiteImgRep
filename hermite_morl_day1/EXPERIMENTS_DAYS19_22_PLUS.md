# Experiments Days19-22 Plus

## Detail-only / H00 gratis

Con `env.detail_only: true`, el ambiente inicia cada episodio con H00 ya seleccionado. H00 aparece en `selected_labels`, pero no cuenta para `cost`, `k_effective` ni `k_norm_effective`. Los componentes seleccionables por el agente son solo detalles.

Con `env.detail_only: false`, el ambiente mantiene el comportamiento historico: mascara inicial vacia, reconstruccion inicial cero y H00 seleccionable como cualquier componente.

## Reward normalization

Con `env.reward_normalization: true`, las mejoras de MSE y SSIM se escalan por la dificultad inicial de la imagen:

```text
delta_mse = (old_mse - new_mse) / initial_mse
delta_ssim = (new_ssim - old_ssim) / (1 - initial_ssim)
```

La recompensa sigue teniendo dimension 4: MSE, SSIM, costo y K.

## Edge correlation

`edge_corr` mide la correlacion de Pearson entre magnitudes de gradiente Sobel de la imagen original y la reconstruccion. Mayor es mejor. Un valor cercano a 1 indica buena preservacion de bordes; cerca de 0 indica poca coincidencia estructural; valores negativos sugieren bordes desalineados o invertidos.

## Comandos

Multi-seed:

```bash
python scripts/run_multiseed_experiment.py --config configs/default.yaml --seeds 0 1 2 3 4
```

Smoke multi-seed:

```bash
python scripts/run_multiseed_experiment.py --config configs/default.yaml --seeds 0 1 --episodes 100 --eval-max-images 2
```

Equal-budget hypervolume:

```bash
python scripts/run_equal_budget_hypervolume.py --config configs/default.yaml --budget 50 --repeats 20
```

Same-K comparison:

```bash
python scripts/run_same_k_comparison.py --config configs/default.yaml
```

Reconstruction ablation:

```bash
python scripts/run_reconstruction_ablation.py --config configs/default.yaml
```

Reconstruction ablation reentrenando por modo:

```bash
python scripts/run_reconstruction_ablation.py --config configs/default.yaml --train-per-mode
```

## CSV principales

`results/multiseed/multiseed_method_summary.csv`:
media y desviacion entre semillas por metodo y metrica (`mse`, `ssim`, `edge_corr`, `cost`, `k_effective`, `hypervolume`).

`results/multiseed/multiseed_agent_preference_summary.csv`:
media y desviacion entre semillas por preferencia del agente.

`results/tables/equal_budget_hypervolume.csv`:
hipervolumen comparable usando el mismo numero efectivo de soluciones por metodo.

`results/tables/same_k_comparison_per_image.csv`:
comparacion por imagen y preferencia usando el mismo `k_effective` del agente.

`results/tables/same_k_comparison_summary.csv`:
resumen de calidad/costo por preferencia y metodo.

`results/tables/reconstruction_ablation_summary.csv`:
comparacion entre `direct_sum` y `least_squares`.

## Notas

Los baselines de energia quedaron unificados como `top-k energy`. No se implementa DFU, fusion multimodal, PCN ni cambios en la dimension de recompensa.
