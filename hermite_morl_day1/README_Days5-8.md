# Days5-8 — Ambiente MOMDP para selección Hermite-Gauss

Esta etapa agrega el ambiente `HermiteSelectionEnv`, donde cada episodio corresponde a seleccionar secuencialmente componentes Hermite-Gauss para reconstruir una imagen.

## Estado

Para `N=3`, hay 10 componentes Hermite-Gauss. El estado tiene dimensión 24:

```text
[energías normalizadas (10), máscara seleccionada (10), MSE actual, SSIM actual, costo, K normalizado]
```

## Acciones

```text
0..9  -> seleccionar componente Hermite-Gauss
10    -> STOP
```

## Recompensa vectorial

```text
r = [MSE_old - MSE_new, SSIM_new - SSIM_old, -(C_new-C_old), -(K_new-K_old)]
```

## Cómo correr

Desde `hermite_morl_day1/`:

```bash
pip install -r requirements.txt
python tests/test_Days5_8_env.py
python scripts/run_Days5_8.py
```

## Archivos agregados

```text
src/env_hermite_momdp.py
src/env_utils.py
scripts/run_Days5_8.py
tests/test_Days5_8_env.py
README_Days5-8.md
```

## Salidas principales

```text
results/Days5-8_env_checks.json
results/Days5-8_manifest.json
results/tables/Days5-8_random_episode.csv
results/figures/Days5-8_random_episode_summary.png
results/reconstructions/Days5-8_random_episode_reconstruction.npy
```
