"""Vẽ Fig. 6/7: bar chart tỷ lệ phát hiện (%) theo dropout cho từng surrogate.

``input``: MultiIndex/DataFrame có cột ``Model`` (``*_s``) và ``Dropout_rate`` sau ``reset_index``.
Lưu ``fp_out`` (150 dpi) và gọi ``plt.show()``.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from pathlib import Path

def plot_bar(input: pd.DataFrame, fp_out: str = None):
    """Vẽ 5 subplot (mỗi loại surrogate) + legend target trên subplot cuối."""
    sns.set_theme(style="whitegrid")
    df = input.reset_index()
    
    SM_model_nams = ['mlp_s', 'cnn_s', 'rescnn_s', 'lstm_s', 'Selfattention_s']
    dfs = []
    for one_model_name in SM_model_nams:
        df_temp = df.loc[df['Model'] == one_model_name].sort_values('Dropout_rate').set_index('Dropout_rate')
        df_temp = df_temp.iloc[:, 1:]
        dfs.append(df_temp)

    fig = plt.figure(figsize=(9, 5))
    gs = GridSpec(2, 6, width_ratios=[1,1,1,1,1,1])

    ax1 = plt.subplot(gs[0, :2])
    ax2 = plt.subplot(gs[0, 2:4])
    ax3 = plt.subplot(gs[0, 4:])

    ax4 = plt.subplot(gs[1, 1:3])
    ax5 = plt.subplot(gs[1, 3:5])

    def plot_bar(ax, df, title, leg):
        xticks = df.index.astype(str)
        df_long = df.reset_index().melt('Dropout_rate', var_name='Columns', value_name='Values')
        sns.barplot(data=df_long, x='Dropout_rate', y='Values', hue='Columns', 
                ax=ax, palette='Set2', saturation=0.8, legend=leg)
        
        ax.set_title(title, fontsize=10, pad=4)

        ax.set_xticks(xticks)
        ax.tick_params(axis='x', labelsize=7, pad=-2)
        ax.set_xlabel('Dropout rate', fontsize=9, labelpad=1)

        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.tick_params(axis='y', labelsize=7, pad=-4)
        ax.set_ylabel('Detection rate(%)', fontsize=9, labelpad=2)

        if leg:
            handles, original_labels = ax.get_legend_handles_labels()
            custom_labels = ["MLP-t", "CNN-t", "ResCNN-t", "LSTM-t", "Self-attention-t", 'Average']
            ax.legend(
                    handles,
                    custom_labels,
                    fontsize=7,
                    handleheight=0.5,
                    handlelength=2,
                    handletextpad=0.5,
                    borderaxespad=0.3,
                    bbox_to_anchor=(1.05, 0.51), 
                    loc='upper left'
                )

        ax.grid(axis='y', alpha=0.3)

    plot_bar(ax1, dfs[0], "(a) MLP-s", False)
    plot_bar(ax2, dfs[1], "(b) CNN-s", False)
    plot_bar(ax3, dfs[2], "(c) ResCNN-s", False)
    plot_bar(ax4, dfs[3], "(d) LSTM-s", False)
    plot_bar(ax5, dfs[4], "(e) Self-attention-s", True)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.5)
    if fp_out:
        out_path = Path(fp_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()