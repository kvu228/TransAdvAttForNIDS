"""Line plot Fig. 8/9: detection vs **số đặc trưng đầu vào surrogate** (cột số), theo attack.

``input`` có ``Target Model``, ``Attack`` và các cột int (27, 30, …).
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from pathlib import Path

def plot_line2(input: pd.DataFrame, fp_out: str = None):
    """Melt long-format rồi một đồ thị con cho mỗi target model."""
    df = input.reset_index()

    feature_cols = [c for c in df.columns if c not in ["Target Model", "Attack"]]
    long_df = df.melt(id_vars=["Target Model", "Attack"],
                    value_vars=feature_cols,
                    var_name="Removed",
                    value_name="Detection")
    long_df["Removed"] = long_df["Removed"].astype(int)

    sns.set_theme(style="whitegrid")
    models = ["mlp_t", "cnn_t", "rescnn_t", "lstm_t", "Selfattention_t"]
    idx    = ['a', 'b', 'c', 'd', 'e']
    x_vals = [int(c) for c in feature_cols]

    fig = plt.figure(figsize=(8, 6))
    gs  = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 1])

    ax_map = {
        "mlp_t":            fig.add_subplot(gs[0, 0:2]),
        "cnn_t":            fig.add_subplot(gs[0, 2:4]),
        "rescnn_t":         fig.add_subplot(gs[1, 0:2]),
        "lstm_t":           fig.add_subplot(gs[1, 2:4]),
        "Selfattention_t": fig.add_subplot(gs[2, 1:3])
    }

    axes = []
    for j, model in enumerate(models):
        ax = ax_map[model]

        sns.lineplot(
            data=long_df[long_df["Target Model"] == model],
            x="Removed",
            y="Detection",
            hue="Attack",
            marker="o",
            markersize=3,
            linewidth=1,
            palette="Set2",
            dashes=False,
            ax=ax
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)

        ax.set_title(f"({idx[j]}) {model}", fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_xticks(x_vals)
        ax.tick_params(axis="x", labelsize=7, pad=1)
        ax.set_xlabel("The number of input features", fontsize=9, labelpad=1)
        ax.set_ylabel("Detection rate(%)",   fontsize=9, labelpad=2)
        ax.tick_params(axis="y", labelsize=7, pad=1)

        ax.get_legend().remove()

        axes.append(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
            bbox_to_anchor=(0.8, 0.07),
            loc="lower right",
            fontsize=7)

    plt.tight_layout(rect=[0, 0, 0.93, 1])
    if fp_out:
        out_path = Path(fp_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()