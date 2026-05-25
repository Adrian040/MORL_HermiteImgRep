# Days9-14 — Patch compatible con Days5-8

Este paquete **no reemplaza** `src/env_hermite_momdp.py`, `src/env_utils.py`, `src/plots.py` ni `configs/default.yaml`.

Archivos que se agregan o reemplazan:

```text
src/morl_envelope.py
src/days9_14_env_adapter.py
src/training_plots.py
scripts/train_Days9_14_envelope.py
scripts/evaluate_Days9_14_policy.py
tests/test_Days9_14_training.py
configs/Days9-14_train.yaml
configs/Days9-14_smoke.yaml
requirements_Days9-14_extra.txt
README_Days9-14_patch.md
```

Comandos:

```bash
pip install -r requirements.txt -r requirements_Days9-14_extra.txt
python tests/test_Days9_14_training.py
python scripts/train_Days9_14_envelope.py --config configs/Days9-14_smoke.yaml
python scripts/train_Days9_14_envelope.py --config configs/Days9-14_train.yaml
```

La penalización por acción repetida se conserva como magnitud positiva:

```yaml
repeated_action_penalty: 0.05
```

El adaptador la pasa al ambiente Days5-8 como valor positivo, consistente con la implementación donde el ambiente la resta internamente en las componentes de costo/parsimonia.
