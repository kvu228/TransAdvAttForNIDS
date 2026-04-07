import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from pathlib import Path

def plot_hm(input: pd.DataFrame, fp_out: str = None):
    df = input.reset_index()

    model_names = ['mlp_t', 'cnn_t', 'rescnn_t', 'lstm_t', 'Selfattention_t']

    col_labels = df.columns.tolist()[2:]
    matrices = []
    for one_name in model_names:
        df_temp = df.loc[df['Model'] == one_name].sort_values(by='step_size')
        df_temp2 = df_temp.iloc[:, 2:].values
        matrices.append(df_temp.iloc[:, 2:].values)
        row_labels = df_temp['step_size'].tolist()

    # Create a new figure with a custom grid layout
    fig = plt.figure(figsize=(6.5, 4))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
    titles = ["(a) MLP-t", "(b) CNN-t", "(c) ResCNN-t", "(d) LSTM-t", "(e) Self-attention-t"]

    # First row: three heatmaps
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for i in range(3):
        sns.heatmap(matrices[i], annot=True,
                    annot_kws={"size": 5}, fmt=".1f",
                    ax=axes[i], cmap='magma_r',
                        xticklabels=col_labels, yticklabels=row_labels,
                        square=True, cbar=False)
        axes[i].set_title(titles[i], pad=4, fontsize=10)
        axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0, fontsize=7)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), fontsize=7)

    # Second row: two heatmaps centered (occupying middle and right positions)
    ax4 = fig.add_subplot(gs[1, 0:2])  # First heatmap spans two columns
    ax5 = fig.add_subplot(gs[1, 1:3])    # Second heatmap in the third column

    sns.heatmap(matrices[3], ax=ax4, cmap='magma_r',
                annot=True, annot_kws={"size": 5},
                xticklabels=col_labels, yticklabels=row_labels,
                fmt=".1f", square=True, cbar=False)
    ax4.set_title(titles[3], pad=4, fontsize=10)
    ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0, fontsize=7)
    ax4.set_xticklabels(ax4.get_xticklabels(), fontsize=7)

    sns.heatmap(matrices[4], ax=ax5, cmap='magma_r', 
                annot=True, annot_kws={"size": 5}, 
                xticklabels=col_labels, yticklabels=row_labels,
                fmt=".1f", square=True, cbar=False)
    ax5.set_title(titles[4], pad=4, fontsize=10)
    ax5.set_yticklabels(ax5.get_yticklabels(), rotation=0, fontsize=7)
    ax5.set_xticklabels(ax5.get_xticklabels(), fontsize=7)

    # Create a single colorbar below the heatmaps
    cbar_ax = fig.add_axes([0.8, 0.18, 0.015, 0.7])  # Position: [left, bottom, width, height]
    # sns.heatmap(matrices[0], ax=axes[0], cmap='magma_r', cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"})
    sns.heatmap(matrices[0], ax=axes[0], 
                cmap='magma_r', cbar_ax=cbar_ax, 
                xticklabels=col_labels, yticklabels=row_labels,
                cbar_kws={"label": "Detection rate (%)"})
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, fontsize=7)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), fontsize=7)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Leave space for the colorbar
    if fp_out:
        out_path = Path(fp_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
