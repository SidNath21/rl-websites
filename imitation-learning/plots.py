from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["diffusion", "flow", "mse"]

MODEL_COLORS = {
    "flow": "#F0434F",
    "mse": "#229487",
    "diffusion": "#538AE5",
}


def _get_metric_columns(df, keyword):
    """Return (main, min, max) columns for a given keyword if they exist."""
    main_col = None
    min_col = None
    max_col = None
    for col in df.columns:
        if keyword in col and "__MIN" not in col and "__MAX" not in col:
            main_col = col
        elif keyword in col and "__MIN" in col:
            min_col = col
        elif keyword in col and "__MAX" in col:
            max_col = col
    return main_col, min_col, max_col


def plot_single_curve(
    csv_path: Path,
    metric_keyword: str,
    title: str,
    ylabel: str,
    output_path: Path,
    color: str,
):
    """Create a clean line plot (with optional min/max shading) and save to PNG."""
    df = pd.read_csv(csv_path)

    if "Step" not in df.columns:
        raise ValueError(f'"Step" column not found in {csv_path}')

    main_col, min_col, max_col = _get_metric_columns(df, metric_keyword)
    if main_col is None:
        raise ValueError(f'No column containing "{metric_keyword}" found in {csv_path}')

    sns.set_theme(style="whitegrid", font_scale=1.2)

    fig, ax = plt.subplots(figsize=(7, 4))

    x = df["Step"]
    y = df[main_col]

    ax.plot(x, y, label=metric_keyword, color=color, linewidth=2.0)

    if min_col is not None and max_col is not None:
        ax.fill_between(x, df[min_col], df[max_col], color=color, alpha=0.2, linewidth=0)

    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)

    sns.despine(ax=ax)
    fig.tight_layout()

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_all_plots():
    """Generate per-model plots and combined comparison plots."""
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # Per-model plots
    for model in MODELS:
        # Training loss
        loss_csv = DATA_DIR / f"{model}_loss.csv"
        loss_png = IMAGES_DIR / f"{model}_loss.png"
        color = MODEL_COLORS.get(model, "#1f77b4")
        plot_single_curve(
            csv_path=loss_csv,
            metric_keyword="loss",
            title=f"{model.capitalize()} Policy – Training Loss",
            ylabel="Loss",
            output_path=loss_png,
            color=color,
        )

        # Evaluation reward
        reward_csv = DATA_DIR / f"{model}_reward.csv"
        reward_png = IMAGES_DIR / f"{model}_reward.png"
        color = MODEL_COLORS.get(model, "#1f77b4")
        plot_single_curve(
            csv_path=reward_csv,
            metric_keyword="eval/mean_reward",
            title=f"{model.capitalize()} Policy – Evaluation Mean Reward",
            ylabel="Mean reward",
            output_path=reward_png,
            color=color,
        )

    # Combined loss plot
    fig, ax = plt.subplots(figsize=(7, 4))
    for model in MODELS:
        csv_path = DATA_DIR / f"{model}_loss.csv"
        df = pd.read_csv(csv_path)
        main_col, _, _ = _get_metric_columns(df, "loss")
        if main_col is None:
            continue
        color = MODEL_COLORS.get(model, "#1f77b4")
        ax.plot(df["Step"], df[main_col], label=model.capitalize(), color=color, linewidth=2.0)
    ax.set_title("Training Loss – All Policies")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.legend()
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "combined_loss.png", dpi=200)
    plt.close(fig)

    # Combined reward plot
    fig, ax = plt.subplots(figsize=(7, 4))
    for model in MODELS:
        csv_path = DATA_DIR / f"{model}_reward.csv"
        df = pd.read_csv(csv_path)
        main_col, _, _ = _get_metric_columns(df, "eval/mean_reward")
        if main_col is None:
            continue
        color = MODEL_COLORS.get(model, "#1f77b4")
        ax.plot(df["Step"], df[main_col], label=model.capitalize(), color=color, linewidth=2.0)
    ax.set_title("Evaluation Mean Reward – All Policies")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean reward")
    ax.legend()
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "combined_reward.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    generate_all_plots()