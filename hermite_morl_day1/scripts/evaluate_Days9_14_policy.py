from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from scripts.train_Days9_14_envelope import evaluate_policy, load_config
from src.data_utils import prepare_dataset
from src.days9_14_env_adapter import env_state_dim, make_selection_env_from_images
from src.morl_envelope import VectorQNetwork, default_preferences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/Days9-14_train.yaml")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/Days9-14_best_envelope_dqn.pt")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--output", type=str, default="results/tables/Days9-14_policy_eval.csv")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    preferences = np.asarray(checkpoint.get("preferences", default_preferences()), dtype=np.float32)

    splits = prepare_dataset(config)
    env = make_selection_env_from_images(splits[args.split], config, split=args.split)
    hidden_dim = int(checkpoint.get("hidden_dim", config.get("training", {}).get("hidden_dim", 128)))
    state_dim = int(checkpoint.get("state_dim", env_state_dim(env)))
    net = VectorQNetwork(state_dim, env.reward_dim, env.n_actions, hidden_dim=hidden_dim).to(device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    df = evaluate_policy(env, net, preferences, device, max_images=None)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    summary = df.groupby(["preference_id", "preference_name"]).agg(
        mse=("mse", "mean"),
        ssim=("ssim", "mean"),
        k=("k", "mean"),
        cost=("cost", "mean"),
        score=("score", "mean"),
    ).reset_index()
    print(summary.round(4).to_string(index=False))
    print(f"Evaluación guardada en: {output}")


if __name__ == "__main__":
    main()
