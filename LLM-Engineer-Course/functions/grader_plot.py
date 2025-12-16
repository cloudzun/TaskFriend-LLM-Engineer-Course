# functions/grader_plot.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_grader(scores_dict, title="Prompt Evolution: Response Quality", figsize=(10, 6), palette="coolwarm"):
    # Convert to DataFrame
    df = pd.DataFrame(scores_dict).T  # Transpose: versions as rows
    df = df.reset_index().rename(columns={'index': 'Version'})
    df_melted = df.melt(id_vars='Version', var_name='Dimension', value_name='Score')

    # Create plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df_melted,
        x="Dimension",
        y="Score",
        hue="Version",
        palette=palette
    )

    # Annotate bars with scores
    for p in ax.patches:
        height = p.get_height()
        if height > 0:  # Only label non-zero
            ax.annotate(
                f"{height:.1f}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                xytext=(0, 3),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold'
            )

    # Customize appearance
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Score (1–5)", fontsize=12)
    ax.set_xlabel("")
    ax.set_title(title, fontsize=16, pad=20)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=10)

    # Legend
    plt.legend(title="", fontsize=10, frameon=True, fancybox=True, shadow=False)
    plt.tight_layout()
    plt.show()