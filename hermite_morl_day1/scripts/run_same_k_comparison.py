from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    _NUMPY_MAJOR = int(np.__version__.split(".", 1)[0])
except Exception:
    _NUMPY_MAJOR = 1

try:
    if _NUMPY_MAJOR >= 2:
        raise ImportError("Skip matplotlib fallback under NumPy 2 runtime.")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from src.baselines import _detail_candidates, _energy_order, _evaluate_selected, greedy_selection
from src.data_utils import prepare_dataset
from src.days9_14_env_adapter import make_selection_env_from_images
from src.env_utils import load_config
from src.evaluate import evaluate_agent_policy


def _agent_or_budget_rows(env, config: dict, checkpoint: str | Path | None, max_images: int | None = None) -> pd.DataFrame:
    device = str(config.get("training", {}).get("device", "cpu"))
    if device == "auto":
        device = "cpu"
    agent_df = evaluate_agent_policy(env, checkpoint, config, device=device, max_images=max_images)
    if not agent_df.empty:
        return agent_df

    preferences = config.get("training", {}).get("preferences", [[0.25, 0.25, 0.25, 0.25]])
    budgets = config.get("evaluation", {}).get("ks", [1, 3, 5])
    n_images = len(env.images) if max_images is None else min(int(max_images), len(env.images))
    rows = []
    for pref_id, _pref in enumerate(preferences):
        budget = int(budgets[min(pref_id, len(budgets) - 1)])
        for image_id in range(n_images):
            rows.append({
                "method": "budget_reference",
                "preference_id": pref_id,
                "preference_name": f"pref_{pref_id}",
                "image_id": image_id,
                "k_effective": min(budget, len(_detail_candidates(env))),
            })
    return pd.DataFrame(rows)


def _same_k_rows(env, agent_like_df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    candidates = _detail_candidates(env)
    greedy_cache: dict[tuple[int, int], list[int]] = {}

    for _, ref in agent_like_df.iterrows():
        image_id = int(ref["image_id"])
        image = env.images[image_id]
        k = int(ref.get("k_effective", ref.get("k", 0)))
        k = max(0, min(k, len(candidates)))
        pref_name = str(ref.get("preference_name", "pref"))

        if str(ref.get("method", "")) == "Envelope-DQN":
            agent_row = ref.to_dict()
            agent_row["preference_name"] = pref_name
            rows.append(agent_row)

        if k == 0:
            baseline_selected = {
                "random": [],
                "top-k energy": [],
                "greedy": [],
            }
        else:
            order = [int(i) for i in _energy_order(env, image) if int(i) in set(candidates)]
            key = (image_id, k)
            if key not in greedy_cache:
                greedy_cache[key] = greedy_selection(env, image, max_k=k)
            baseline_selected = {
                "random": rng.choice(candidates, size=k, replace=False).astype(int).tolist(),
                "top-k energy": order[:k],
                "greedy": greedy_cache[key][:k],
            }
        for method, selected in baseline_selected.items():
            row = _evaluate_selected(env, image, image_id, selected, method, k)
            row["preference_name"] = pref_name
            rows.append(row)
    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["preference_name", "method"]).agg(
        k_effective_mean=("k_effective", "mean"),
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        ssim_mean=("ssim", "mean"),
        ssim_std=("ssim", "std"),
        edge_corr_mean=("edge_corr", "mean"),
        edge_corr_std=("edge_corr", "std"),
        cost_mean=("cost", "mean"),
        cost_std=("cost", "std"),
    ).reset_index()


def _plot(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if plt is None:
        width, height = 960, 560
        margin = 80
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
            small = ImageFont.truetype("DejaVuSans.ttf", 11)
        except Exception:
            font = ImageFont.load_default()
            small = font
        data = summary.copy()
        prefs = list(data["preference_name"].drop_duplicates())
        methods = list(data["method"].drop_duplicates())
        max_y = max(float(data["ssim_mean"].max()), 1e-8)
        bar_group_w = (width - 2 * margin) / max(1, len(prefs))
        bar_w = bar_group_w * 0.75 / max(1, len(methods))
        colors = {"Envelope-DQN": "#9467bd", "random": "#7f7f7f", "top-k energy": "#1f77b4", "greedy": "#d62728"}
        draw.text((margin, 24), "Same-K quality comparison", fill="black", font=font)
        draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
        draw.line((margin, margin, margin, height - margin), fill="black", width=2)
        for i, pref in enumerate(prefs):
            for j, method in enumerate(methods):
                row = data[(data["preference_name"] == pref) & (data["method"] == method)]
                if row.empty:
                    continue
                val = float(row["ssim_mean"].iloc[0])
                x0 = margin + i * bar_group_w + j * bar_w + bar_group_w * 0.1
                x1 = x0 + bar_w * 0.9
                y1 = height - margin
                y0 = y1 - (val / max_y) * (height - 2 * margin)
                draw.rectangle((x0, y0, x1, y1), fill=colors.get(method, "#333333"))
            draw.text((margin + i * bar_group_w + 4, height - margin + 8), pref[:16], fill="black", font=small)
        for j, method in enumerate(methods):
            y = margin + 20 * j
            draw.rectangle((width - 230, y, width - 215, y + 12), fill=colors.get(method, "#333333"))
            draw.text((width - 210, y - 2), method, fill="black", font=small)
        canvas.save(output_path)
        return

    data = summary.copy()
    prefs = list(data["preference_name"].drop_duplicates())
    methods = list(data["method"].drop_duplicates())
    x = np.arange(len(prefs))
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = {"Envelope-DQN": "#9467bd", "random": "#7f7f7f", "top-k energy": "#1f77b4", "greedy": "#d62728"}
    for i, method in enumerate(methods):
        sub = data[data["method"] == method].set_index("preference_name").reindex(prefs)
        ax.bar(x - 0.4 + width / 2 + i * width, sub["ssim_mean"], width=width, label=method, color=colors.get(method, "#333333"), alpha=0.9)
    ax.set_title("Same-K quality comparison", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Preference")
    ax.set_ylabel("Mean SSIM")
    ax.set_xticks(x)
    ax.set_xticklabels(prefs, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--agent-checkpoint", type=str, default="results/checkpoints/Days9-14_best_envelope_dqn.pt")
    parser.add_argument("--eval-max-images", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config.get("project_root", "."))
    tables_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    splits = prepare_dataset(config)
    env = make_selection_env_from_images(splits[args.split], config, split=args.split)
    agent_like = _agent_or_budget_rows(env, config, root / args.agent_checkpoint, max_images=args.eval_max_images)
    per_image = _same_k_rows(env, agent_like, seed=int(config.get("seed", 0)))
    summary = _summary(per_image)

    per_image.to_csv(tables_dir / "same_k_comparison_per_image.csv", index=False)
    summary.to_csv(tables_dir / "same_k_comparison_summary.csv", index=False)
    _plot(summary, figures_dir / "same_k_quality_comparison.png")

    print(f"Same-K comparison saved to {tables_dir / 'same_k_comparison_summary.csv'}")


if __name__ == "__main__":
    main()
