import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def _get_learning_curve_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col.endswith(" - Train_AverageReturn") and "__MIN" not in col and "__MAX" not in col
    ]


def _experiment_name(column_name: str) -> str:
    return column_name.replace(" - Train_AverageReturn", "")


def _experiment_name_with_suffix(column_name: str, suffix: str) -> str:
    return column_name.replace(suffix, "")


def _get_metric_columns(df: pd.DataFrame, suffix: str) -> list[str]:
    return [
        col for col in df.columns if col.endswith(suffix) and "__MIN" not in col and "__MAX" not in col
    ]


def _plot_metric_csv(
    csv_path: Path,
    metric_suffix: str,
    title: str,
    y_label: str,
    output_path: Path,
    envsteps_per_iter: int = 1000,
) -> None:
    df = pd.read_csv(csv_path)
    if "Step" not in df.columns:
        raise ValueError(f'"Step" column not found in {csv_path}')

    metric_cols = sorted(_get_metric_columns(df, metric_suffix))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = df["Step"] * envsteps_per_iter

    if not metric_cols:
        ax.text(
            0.5,
            0.5,
            f'No columns ending with "{metric_suffix}" found',
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        for col in metric_cols:
            label = _experiment_name_with_suffix(col, metric_suffix)
            ax.plot(x, df[col], label=label, linewidth=2.0)
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel("Number of environment steps")
    ax.set_ylabel(y_label)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_group(
    df: pd.DataFrame,
    columns: list[str],
    title: str,
    output_path: Path,
    envsteps_per_iter: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = df["Step"] * envsteps_per_iter

    if not columns:
        # Keep output generation deterministic even when a group is absent in the CSV.
        ax.text(
            0.5,
            0.5,
            "No matching experiments found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        for col in sorted(columns):
            label = _experiment_name(col)
            ax.plot(x, df[col], label=label, linewidth=2.0)
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel("Number of environment steps")
    ax.set_ylabel("Average return")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_cartpole_comparison_plots():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    csv_path = DATA_DIR / "cartpole_experiments.csv"
    df = pd.read_csv(csv_path)

    if "Step" not in df.columns:
        raise ValueError(f'"Step" column not found in {csv_path}')

    learning_curve_cols = _get_learning_curve_columns(df)
    small_batch_cols = [
        col for col in learning_curve_cols if _experiment_name(col).startswith("cartpole") and "_lb" not in _experiment_name(col)
    ]
    large_batch_cols = [
        col for col in learning_curve_cols if _experiment_name(col).startswith("cartpole_lb")
    ]

    _plot_group(
        df=df,
        columns=small_batch_cols,
        title="CartPole (Small Batch): Average Return vs Environment Steps",
        output_path=IMAGES_DIR / "cartpole_small_batch_learning_curves.png",
        envsteps_per_iter=1000,
    )
    _plot_group(
        df=df,
        columns=large_batch_cols,
        title="CartPole (Large Batch): Average Return vs Environment Steps",
        output_path=IMAGES_DIR / "cartpole_large_batch_learning_curves.png",
        envsteps_per_iter=5000,
    )


def generate_lunar_lander_gae_lambda_plot(
    *,
    smooth: bool = True,
    ewm_span: int = 25,
) -> None:
    """Eval return vs steps for GAE-λ ablation (LunarLander).

    When ``smooth`` is True, raw eval returns are drawn faintly and an
    exponentially weighted mean (``ewm_span``) is overlaid for readability.
    """
    if smooth and ewm_span < 1:
        raise ValueError("ewm_span must be >= 1 when smooth=True")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    csv_path = DATA_DIR / "lunar_lander.csv"
    df = pd.read_csv(csv_path)
    if "Step" not in df.columns:
        raise ValueError(f'"Step" column not found in {csv_path}')

    # (legend label, column suffix after run name) — run dirs use lambda0, lambda0.95, …, lambda1
    lambda_runs: list[tuple[str, str]] = [
        ("λ = 0", "lunar_lander_lambda0 - Eval_AverageReturn"),
        ("λ = 0.95", "lunar_lander_lambda0.95 - Eval_AverageReturn"),
        ("λ = 0.98", "lunar_lander_lambda0.98 - Eval_AverageReturn"),
        ("λ = 0.99", "lunar_lander_lambda0.99 - Eval_AverageReturn"),
        ("λ = 1.0", "lunar_lander_lambda1 - Eval_AverageReturn"),
    ]
    envsteps_per_iter = 1000

    fig, ax = plt.subplots(figsize=(9, 5))
    x = df["Step"].astype(float) * envsteps_per_iter
    colors = sns.color_palette("husl", n_colors=len(lambda_runs))

    for (legend_label, col), color in zip(lambda_runs, colors, strict=True):
        if col not in df.columns:
            raise ValueError(f"Expected column {col!r} not found in {csv_path}")
        y = pd.to_numeric(df[col], errors="coerce")
        if smooth:
            y_smooth = y.ewm(span=ewm_span, adjust=False).mean()
            ax.plot(x, y, color=color, alpha=0.22, linewidth=1.0, zorder=1)
            ax.plot(x, y_smooth, color=color, linewidth=2.6, label=legend_label, zorder=2)
        else:
            ax.plot(x, y, color=color, linewidth=2.6, label=legend_label)

    ax.legend(title="GAE-λ", frameon=True, loc="best")
    base_title = "LunarLander: Evaluation Return vs Environment Steps (GAE-λ)"
    if smooth:
        ax.set_title(
            f"{base_title}\n(faint: raw; bold: EWM smoothed, span={ewm_span})",
            fontsize=11,
        )
    else:
        ax.set_title(base_title, fontsize=11)
    ax.set_xlabel("Number of environment steps")
    ax.set_ylabel("Average evaluation return")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "lunar_lander_gae_lambda_eval_return.png", dpi=200)
    plt.close(fig)


def generate_pendulum_eval_return_plot(
    *,
    smooth: bool = True,
    ewm_span: int = 25,
) -> None:
    """Eval return vs steps for InvertedPendulum: NA, RTG, and value-baseline ablations.

    Run names encode which options are on: ``na`` = normalized advantages,
    ``rtg`` = return-to-go rewards, ``baseline`` = value-function baseline.
    """
    if smooth and ewm_span < 1:
        raise ValueError("ewm_span must be >= 1 when smooth=True")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    csv_path = DATA_DIR / "pendulum_eval_return.csv"
    df = pd.read_csv(csv_path)
    if "Step" not in df.columns:
        raise ValueError(f'"Step" column not found in {csv_path}')

    # Order: simplest config → full stack (legend matches typical ablation narrative).
    runs: list[tuple[str, str]] = [
        ("Default PG (no NA, no RTG)", "pendulum - Eval_AverageReturn"),
        ("+ Normalized advantage (NA)", "pendulum_na - Eval_AverageReturn"),
        ("+ NA, return-to-go (RTG)", "pendulum_na_rtg - Eval_AverageReturn"),
        ("+ NA, RTG, value baseline", "pendulum_na_rtg_baseline - Eval_AverageReturn"),
    ]
    envsteps_per_iter = 1000

    fig, ax = plt.subplots(figsize=(9, 5))
    x = df["Step"].astype(float) * envsteps_per_iter
    colors = sns.color_palette("husl", n_colors=len(runs))

    for (legend_label, col), color in zip(runs, colors, strict=True):
        if col not in df.columns:
            raise ValueError(f"Expected column {col!r} not found in {csv_path}")
        y = pd.to_numeric(df[col], errors="coerce")
        if smooth:
            y_smooth = y.ewm(span=ewm_span, adjust=False).mean()
            ax.plot(x, y, color=color, alpha=0.22, linewidth=1.0, zorder=1)
            ax.plot(x, y_smooth, color=color, linewidth=2.6, label=legend_label, zorder=2)
        else:
            ax.plot(x, y, color=color, linewidth=2.6, label=legend_label)

    ax.legend(title="Configuration", frameon=True, loc="best")
    base_title = "InvertedPendulum: Evaluation Return vs Environment Steps (NA / RTG / baseline)"
    if smooth:
        ax.set_title(
            f"{base_title}\n(faint: raw; bold: EWM smoothed, span={ewm_span})",
            fontsize=11,
        )
    else:
        ax.set_title(base_title, fontsize=11)
    ax.set_xlabel("Number of environment steps")
    ax.set_ylabel("Average evaluation return")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "pendulum_eval_return.png", dpi=200)
    plt.close(fig)


def generate_cheetah_plots() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.1)
    _plot_metric_csv(
        csv_path=DATA_DIR / "cheetah_baseline_loss.csv",
        metric_suffix=" - Baseline Loss",
        title="Cheetah: Baseline Loss vs Environment Steps",
        y_label="Baseline loss",
        output_path=IMAGES_DIR / "cheetah_baseline_loss.png",
    )
    _plot_metric_csv(
        csv_path=DATA_DIR / "cheetah_eval_return.csv",
        metric_suffix=" - Eval_AverageReturn",
        title="Cheetah: Evaluation Average Return vs Environment Steps",
        y_label="Average evaluation return",
        output_path=IMAGES_DIR / "cheetah_eval_return.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate website plots.")
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="LunarLander GAE-λ plot: raw series only (no EWM overlay).",
    )
    parser.add_argument(
        "--ewm-span",
        type=int,
        default=25,
        help="EWM span when smoothing is enabled (default: 25).",
    )
    args = parser.parse_args()
    # generate_cartpole_comparison_plots()
    # generate_cheetah_plots()
    # generate_lunar_lander_gae_lambda_plot(
    #     smooth=not args.no_smooth,
    #     ewm_span=args.ewm_span,
    # )
    generate_pendulum_eval_return_plot(
        smooth=not args.no_smooth,
        ewm_span=args.ewm_span,
    )