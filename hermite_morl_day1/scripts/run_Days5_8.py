from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.env_utils import load_config, make_env_from_config, run_random_episode
from src.plots import mask_to_image, save_image_grid


def assert_env_contract(env) -> dict:
    obs, info = env.reset(options={"image_index": 0})
    checks = {
        "obs_shape_ok": tuple(obs.shape) == tuple(env.observation_space.shape),
        "obs_finite": bool(np.all(np.isfinite(obs))),
        "reward_dim": int(env.reward_dim),
        "n_components": int(env.n_components),
        "n_actions": int(env.n_actions),
        "stop_action": int(env.stop_action),
        "initial_k": int(info["k"]),
    }

    first_action = 0
    obs, reward, terminated, truncated, info_valid = env.step(first_action)
    checks.update({
        "valid_step_reward_shape_ok": tuple(reward.shape) == (env.reward_dim,),
        "valid_step_reward_finite": bool(np.all(np.isfinite(reward))),
        "valid_step_k": int(info_valid["k"]),
        "valid_step_mse_finite": bool(np.isfinite(info_valid["mse"])),
        "valid_step_ssim_finite": bool(np.isfinite(info_valid["ssim"])),
    })

    obs, reward, terminated, truncated, info_repeated = env.step(first_action)
    checks.update({
        "repeated_action_detected": info_repeated["event"] == "repeated_action",
        "repeated_reward_shape_ok": tuple(reward.shape) == (env.reward_dim,),
    })

    obs, reward, terminated, truncated, info_stop = env.step(env.stop_action)
    checks.update({
        "stop_terminates": bool(terminated),
        "stop_event_ok": info_stop["event"] == "stop",
    })
    checks["all_checks_passed"] = bool(all(v for k, v in checks.items() if k != "reward_dim" and k != "n_components" and k != "n_actions" and k != "stop_action" and k != "initial_k" and k != "valid_step_k"))
    return checks


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    config = load_config(config_path)
    root = Path(config.get("project_root", "."))
    tables_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    recon_dir = root / "results" / "reconstructions"
    for directory in [tables_dir, figures_dir, recon_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    env, splits = make_env_from_config(config, split="train")
    checks = assert_env_contract(env)
    with open(root / "results" / "Days5-8_env_checks.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)

    trajectory = run_random_episode(
        env,
        max_interactions=int(config.get("env", {}).get("random_episode_steps", env.max_steps + 1)),
        avoid_repeated=True,
    )
    trajectory_path = tables_dir / "Days5-8_random_episode.csv"
    trajectory.to_csv(trajectory_path, index=False)

    final_info = env._get_info(event="final")
    np.save(recon_dir / "Days5-8_random_episode_reconstruction.npy", final_info["reconstruction"])
    save_image_grid(
        [
            env.state.image,
            final_info["reconstruction"],
            mask_to_image(final_info["mask"]),
        ],
        [
            "Original",
            f"Reconstrucción K={final_info['k']}",
            "Máscara seleccionada",
        ],
        figures_dir / "Days5-8_random_episode_summary.png",
        cols=3,
    )

    manifest = {
        "task": "Days5-8 MOMDP environment",
        "state_dim": int(env.observation_dim),
        "reward_dim": int(env.reward_dim),
        "n_components": int(env.n_components),
        "n_actions": int(env.n_actions),
        "stop_action": int(env.stop_action),
        "max_steps": int(env.max_steps),
        "n_train": int(len(splits["train"])),
        "n_val": int(len(splits["val"])),
        "n_test": int(len(splits["test"])),
        "checks_file": str(root / "results" / "Days5-8_env_checks.json"),
        "trajectory_file": str(trajectory_path),
        "summary_figure": str(figures_dir / "Days5-8_random_episode_summary.png"),
    }
    with open(root / "results" / "Days5-8_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Days5-8 completado: ambiente MOMDP verificado.")
    print(pd.DataFrame([manifest]).T.rename(columns={0: "value"}).to_string())
    print("\nTrayectoria aleatoria:")
    print(trajectory.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
