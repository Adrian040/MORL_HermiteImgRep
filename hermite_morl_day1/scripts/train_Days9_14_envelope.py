from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from src.data_utils import prepare_dataset, save_processed_dataset
from src.days9_14_env_adapter import env_state_dim, env_valid_action_mask, make_selection_env_from_images
from src.morl_envelope import (
    PreferenceSampler,
    ReplayBuffer,
    VectorQNetwork,
    default_preferences,
    envelope_dqn_loss,
    epsilon_by_step,
    preference_names,
    scalarized_score,
    select_action,
)
from src.metrics import reconstruction_metrics
from src.training_plots import save_training_curves


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = dict(config)
    config.setdefault("training", {})
    config.setdefault("dataset", {})
    if args.episodes is not None:
        config["training"]["episodes"] = int(args.episodes)
    if args.train_images is not None:
        config["dataset"]["train_images"] = int(args.train_images)
    if args.val_images is not None:
        config["dataset"]["val_images"] = int(args.val_images)
    if args.test_images is not None:
        config["dataset"]["test_images"] = int(args.test_images)
    if args.eval_every is not None:
        config["training"]["eval_every"] = int(args.eval_every)
    if args.device is not None:
        config["training"]["device"] = args.device
    return config


@torch.no_grad()
def evaluate_policy(env, network: VectorQNetwork, preferences: np.ndarray, device: torch.device, max_images: int | None = None) -> pd.DataFrame:
    rows = []
    n_eval = len(env.images) if max_images is None else min(int(max_images), len(env.images))
    names = preference_names()
    rng = np.random.default_rng(12345)
    for pref_idx, pref in enumerate(preferences):
        for image_id in range(n_eval):
            state, info = env.reset(options={"image_index": image_id})
            done = False
            truncated = False
            rewards = []
            actions = []
            while not (done or truncated):
                action = select_action(
                    network=network,
                    state=state,
                    preference=pref,
                    epsilon=0.0,
                    valid_action_mask=env_valid_action_mask(env),
                    device=device,
                    rng=rng,
                )
                next_state, reward, done, truncated, info = env.step(action)
                rewards.append(reward)
                actions.append(action)
                state = next_state
            reward_sum = np.sum(rewards, axis=0) if rewards else np.zeros(env.reward_dim, dtype=np.float32)
            scalar_return = float(np.dot(pref, reward_sum))
            metrics_full = reconstruction_metrics(env.images[image_id], info["reconstruction"])
            metrics = {
                "mse": metrics_full["mse"],
                "ssim": metrics_full["ssim"],
                "psnr": metrics_full["psnr"],
                "gradient_mse": metrics_full["gradient_mse"],
                "edge_corr": metrics_full["edge_corr"],
                "cost": float(info["cost"]),
                "k_norm": float(info["k_norm"]),
            }
            rows.append({
                "preference_id": pref_idx,
                "preference_name": names[pref_idx] if pref_idx < len(names) else f"w{pref_idx}",
                "preference": " ".join(f"{v:.3f}" for v in pref),
                "image_id": image_id,
                "mse": metrics["mse"],
                "ssim": metrics["ssim"],
                "psnr": metrics["psnr"],
                "gradient_mse": metrics["gradient_mse"],
                "edge_corr": metrics["edge_corr"],
                "k": int(info.get("paid_k", info["k"])),
                "paid_k": int(info.get("paid_k", info["k"])),
                "total_k": int(info.get("total_k", info["k"])),
                "cost": metrics["cost"],
                "k_norm": metrics["k_norm"],
                "reward_raw": " ".join(f"{v:.8g}" for v in np.asarray(info.get("reward_raw", np.zeros(env.reward_dim)))),
                "reward_used": " ".join(f"{v:.8g}" for v in np.asarray(info.get("reward_used", np.zeros(env.reward_dim)))),
                "scalar_return": scalar_return,
                "score": scalarized_score(metrics),
                "selected_indices": " ".join(map(str, info["selected_indices"])),
                "selected_labels": " ".join(info["selected_labels"]),
                "actions": " ".join(map(str, actions)),
            })
    return pd.DataFrame(rows)


def train(config: dict) -> Dict:
    seed = int(config.get("seed", 0))
    set_seed(seed)
    hermite_cfg = config.get("hermite", {})
    multi_cfg = config.get("hermite_multi_config", {})
    experiment_cfg = config.get("experiment", {})
    active_slug = str(experiment_cfg.get("active_hermite_grid_slug", "main_config"))
    active_sigma = float(hermite_cfg.get("sigma", 1.5))
    active_kernel_size = int(hermite_cfg.get("kernel_size", 13))
    active_max_order = int(hermite_cfg.get("max_order", 3))
    multi_sigmas = " ".join(map(str, multi_cfg.get("sigmas", []))) if bool(multi_cfg.get("enabled", False)) else ""
    multi_kernel_sizes = " ".join(map(str, multi_cfg.get("kernel_sizes", []))) if bool(multi_cfg.get("enabled", False)) else ""

    project_root = Path(config.get("project_root", "."))
    output_root = Path(config.get("experiment", {}).get("output_root", "results"))
    out_root = output_root if output_root.is_absolute() else project_root / output_root
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    ckpt_dir = out_root / "checkpoints"
    processed_dir = Path(config.get("project_root", ".")) / "data" / "processed"
    for d in [tables_dir, figures_dir, ckpt_dir, processed_dir]:
        d.mkdir(parents=True, exist_ok=True)
    with open(out_root / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    splits = prepare_dataset(config)
    save_processed_dataset(splits, processed_dir / "dataset_Days9-14.npz")

    train_env = make_selection_env_from_images(splits["train"], config, split="train")
    val_env = make_selection_env_from_images(splits["val"], config, split="val")
    state_dim = env_state_dim(train_env)

    training_cfg = config.get("training", {})
    device_name = training_cfg.get("device", "auto")
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    preferences = np.asarray(training_cfg.get("preferences", default_preferences().tolist()), dtype=np.float32)
    preferences = preferences / np.maximum(preferences.sum(axis=1, keepdims=True), 1e-12)
    pref_sampler = PreferenceSampler(preferences, seed=seed)
    preference_bank = torch.as_tensor(preferences, dtype=torch.float32, device=device)

    policy_net = VectorQNetwork(state_dim, train_env.reward_dim, train_env.n_actions, hidden_dim=int(training_cfg.get("hidden_dim", 128))).to(device)
    target_net = VectorQNetwork(state_dim, train_env.reward_dim, train_env.n_actions, hidden_dim=int(training_cfg.get("hidden_dim", 128))).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=float(training_cfg.get("lr", 1e-3)))
    buffer = ReplayBuffer(capacity=int(training_cfg.get("replay_size", 50000)), state_dim=state_dim, reward_dim=train_env.reward_dim, seed=seed)

    episodes = int(training_cfg.get("episodes", 3000))
    batch_size = int(training_cfg.get("batch_size", 64))
    gamma = float(training_cfg.get("gamma", 0.95))
    warmup_steps = int(training_cfg.get("warmup_steps", 128))
    target_update_every = int(training_cfg.get("target_update_every", 250))
    eval_every = int(training_cfg.get("eval_every", 250))
    epsilon_start = float(training_cfg.get("epsilon_start", 1.0))
    epsilon_end = float(training_cfg.get("epsilon_end", 0.05))
    epsilon_decay_steps = int(training_cfg.get("epsilon_decay_steps", 3000))
    use_envelope_target = bool(training_cfg.get("use_envelope_target", True))
    eval_max_images = training_cfg.get("eval_max_images", None)

    rng = np.random.default_rng(seed)
    history: List[Dict] = []
    global_step = 0
    best_score = -float("inf")
    best_eval = None

    for episode in range(1, episodes + 1):
        state, info = train_env.reset()
        pref = pref_sampler.sample()
        done = False
        truncated = False
        episode_rewards = []
        losses = []
        actions = []

        while not (done or truncated):
            epsilon = epsilon_by_step(global_step, epsilon_start, epsilon_end, epsilon_decay_steps)
            action = select_action(policy_net, state, pref, epsilon, env_valid_action_mask(train_env), device, rng)
            next_state, reward, done, truncated, info = train_env.step(action)
            buffer.add(state, pref, action, reward, next_state, done or truncated)
            episode_rewards.append(reward)
            actions.append(action)
            state = next_state
            global_step += 1

            if len(buffer) >= max(batch_size, warmup_steps):
                batch = buffer.sample(batch_size, device)
                loss = envelope_dqn_loss(policy_net, target_net, batch, gamma, preference_bank, train_env.n_components, use_envelope_target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
                optimizer.step()
                losses.append(float(loss.item()))

            if global_step % target_update_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

        reward_sum = np.sum(episode_rewards, axis=0) if episode_rewards else np.zeros(train_env.reward_dim, dtype=np.float32)
        log_row = {
            "episode": episode,
            "seed": seed,
            "hermite_config": active_slug,
            "sigma": active_sigma,
            "kernel_size": active_kernel_size,
            "max_order": active_max_order,
            "multi_sigmas": multi_sigmas,
            "multi_kernel_sizes": multi_kernel_sizes,
            "global_step": global_step,
            "epsilon": epsilon_by_step(global_step, epsilon_start, epsilon_end, epsilon_decay_steps),
            "scalar_return": float(np.dot(pref, reward_sum)),
            "loss": float(np.mean(losses)) if losses else np.nan,
            "mse": float(info["mse"]),
            "ssim": float(info["ssim"]),
            "k": int(info.get("paid_k", info["k"])),
            "paid_k": int(info.get("paid_k", info["k"])),
            "total_k": int(info.get("total_k", info["k"])),
            "cost": float(info["cost"]),
            "actions": " ".join(map(str, actions)),
        }
        history.append(log_row)

        if episode % eval_every == 0 or episode == episodes:
            eval_df = evaluate_policy(val_env, policy_net, preferences, device, max_images=eval_max_images)
            mean_score = float(eval_df["score"].mean())
            if mean_score > best_score:
                best_score = mean_score
                best_eval = eval_df.copy()
                torch.save({
                    "model_state_dict": policy_net.state_dict(),
                    "target_state_dict": target_net.state_dict(),
                    "config": config,
                    "preferences": preferences,
                    "state_dim": state_dim,
                    "reward_dim": train_env.reward_dim,
                    "n_actions": train_env.n_actions,
                    "n_components": train_env.n_components,
                    "hidden_dim": int(training_cfg.get("hidden_dim", 128)),
                    "episode": episode,
                    "best_score": best_score,
                }, ckpt_dir / "Days9-14_best_envelope_dqn.pt")
            print(
                f"[Days9-14] episode={episode:04d}/{episodes} return={log_row['scalar_return']:.4f} "
                f"loss={log_row['loss']:.4f} val_score={mean_score:.4f} best={best_score:.4f}",
                flush=True,
            )

    target_net.load_state_dict(policy_net.state_dict())
    history_df = pd.DataFrame(history)
    history_df.to_csv(tables_dir / "Days9-14_training_log.csv", index=False)
    save_training_curves(history, figures_dir / "Days9-14_training_curves.png")

    final_eval = evaluate_policy(val_env, policy_net, preferences, device, max_images=None)
    final_eval["seed"] = seed
    final_eval["hermite_config"] = active_slug
    final_eval["sigma"] = active_sigma
    final_eval["kernel_size"] = active_kernel_size
    final_eval["max_order"] = active_max_order
    final_eval["multi_sigmas"] = multi_sigmas
    final_eval["multi_kernel_sizes"] = multi_kernel_sizes
    final_eval.to_csv(tables_dir / "Days9-14_eval_by_preference.csv", index=False)
    summary = final_eval.groupby(["preference_id", "preference_name", "preference"]).agg(
        mse=("mse", "mean"),
        ssim=("ssim", "mean"),
        psnr=("psnr", "mean"),
        gradient_mse=("gradient_mse", "mean"),
        edge_corr=("edge_corr", "mean"),
        paid_k=("paid_k", "mean"),
        k=("k", "mean"),
        cost=("cost", "mean"),
        scalar_return=("scalar_return", "mean"),
        score=("score", "mean"),
    ).reset_index()
    summary.to_csv(tables_dir / "Days9-14_eval_summary.csv", index=False)
    if best_eval is not None:
        best_eval.to_csv(tables_dir / "Days9-14_best_eval_by_preference.csv", index=False)

    torch.save({
        "model_state_dict": policy_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "config": config,
        "preferences": preferences,
        "state_dim": state_dim,
        "reward_dim": train_env.reward_dim,
        "n_actions": train_env.n_actions,
        "n_components": train_env.n_components,
        "hidden_dim": int(training_cfg.get("hidden_dim", 128)),
        "episode": episodes,
        "best_score": best_score,
        "detail_only_h00_free": bool(config.get("env", {}).get("detail_only_h00_free", True)),
        "reward_mode": str(config.get("reward", {}).get("mode", "relative")),
        "reconstruction_mode": str(config.get("reconstruction", {}).get("mode", "least_squares")),
        "cost_mode": str(config.get("cost", {}).get("mode", "extra_kernel_flops")),
        "free_kernel_size": int(config.get("cost", {}).get("free_kernel_size", 13)),
        "hermite_main_config": {
            "max_order": int(config.get("hermite", {}).get("max_order", 3)),
            "sigma": float(config.get("hermite", {}).get("sigma", 1.5)),
            "kernel_size": int(config.get("hermite", {}).get("kernel_size", 13)),
        },
        "hermite_sweep": config.get("hermite_sweep", {}),
        "seeds": config.get("experiment", {}).get("seeds", [seed]),
        "image_quality_metrics": config.get("evaluation", {}).get("image_quality_metrics", {}).get("metrics", ["psnr"]),
        "edge_metrics": config.get("evaluation", {}).get("edge_metrics", {}).get("metrics", ["gradient_mse", "edge_corr"]),
        "edge_metrics_affect_training": bool(config.get("evaluation", {}).get("edge_metrics", {}).get("affect_training", False)),
        "equal_budget_hv": bool(config.get("evaluation", {}).get("equal_budget", {}).get("enabled", True)),
        "same_k_analysis": bool(config.get("evaluation", {}).get("same_k", {}).get("enabled", True)),
        "external_dfu_eval": "optional",
    }, ckpt_dir / "Days9-14_final_envelope_dqn.pt")

    with open(out_root / "Days9-14_preferences.json", "w", encoding="utf-8") as f:
        json.dump({"names": preference_names(), "preferences": preferences.tolist()}, f, indent=2, ensure_ascii=False)

    manifest = {
        "state_dim": state_dim,
        "reward_dim": train_env.reward_dim,
        "n_actions": train_env.n_actions,
        "n_components": train_env.n_components,
        "episodes": episodes,
        "seed": seed,
        "hermite_config": active_slug,
        "sigma": active_sigma,
        "kernel_size": active_kernel_size,
        "max_order": active_max_order,
        "multi_sigmas": multi_sigmas,
        "multi_kernel_sizes": multi_kernel_sizes,
        "hermite_multi_config": multi_cfg,
        "device": str(device),
        "best_score": best_score,
        "outputs": {
            "training_log": str(tables_dir / "Days9-14_training_log.csv"),
            "eval_summary": str(tables_dir / "Days9-14_eval_summary.csv"),
            "final_checkpoint": str(ckpt_dir / "Days9-14_final_envelope_dqn.pt"),
            "best_checkpoint": str(ckpt_dir / "Days9-14_best_envelope_dqn.pt"),
            "training_curves": str(figures_dir / "Days9-14_training_curves.png"),
        },
    }
    with open(out_root / "Days9-14_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\nResumen de evaluación Days9-14:")
    print(summary.round(4).to_string(index=False))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/Days9-14_train.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--train-images", type=int, default=None)
    parser.add_argument("--val-images", type=int, default=None)
    parser.add_argument("--test-images", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    seeds = args.seeds if args.seeds else [int(config.get("seed", 0))]
    base_output_root = config.get("experiment", {}).get("output_root", "results")
    for seed in seeds:
        run_config = dict(config)
        run_config["seed"] = int(seed)
        if len(seeds) > 1:
            run_config.setdefault("experiment", {})
            run_config["experiment"] = dict(run_config["experiment"])
            run_config["experiment"]["output_root"] = str(Path(base_output_root) / "seeds" / f"seed_{seed}")
        train(run_config)


if __name__ == "__main__":
    main()
