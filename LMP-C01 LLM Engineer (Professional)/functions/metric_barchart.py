#/functions/metric_barchart.py
import matplotlib.pyplot as plt

def plot_metric_comparison(scores_dict, title, ylabel, colors=None):
    """
    A reusable function to create a bar chart for comparing metric scores.
    
    Args:
        scores_dict (dict): Dictionary with labels as keys and metric scores as values.
        title (str): The title for the chart.
        ylabel (str): The label for the y-axis.
        colors (list, optional): List of colors for the bars. Defaults to a standard palette.
    """
    if colors is None:
        colors = ['#28a745', '#ffc107', '#dc3545']  # Green, Amber, Red
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(scores_dict.keys(), scores_dict.values(), color=colors)
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel(ylabel, fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores_dict.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{score:.3f}', 
                 ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()