import pandas as pd
from IPython.display import HTML

# Define metrics to be styled (can be changed as needed)
METRIC_COLUMNS = [
    "context_recall", 
    "context_precision",
    "answer_correctness",
    "faithfulness", 
]

def create_rag_evaluation_table(results_df, show_contexts=True, column_widths=None):
    """
    Creates a styled HTML table for RAG evaluation results with contexts split into individual rows
    and color-coded metrics.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing evaluation results.
        column_widths (list): List of widths in % for [index, question, contexts, ground_truth, answer, metrics...]
                             If None, auto-distributes widths
    
    Returns:
        HTML object containing the styled table
    """
    metric_columns = METRIC_COLUMNS

    # Round metrics to 3 decimal places
    results_df = results_df.copy()
    results_df[metric_columns] = results_df[metric_columns].round(3)
    
    has_contexts = 'contexts' in results_df.columns

    # Split contexts into individual rows
    expanded_rows = []

    for idx, row in results_df.iterrows():
        if has_contexts:
            contexts = row['contexts']
            if isinstance(contexts, list) and len(contexts) > 0:
                for i, context in enumerate(contexts):
                    new_row = row.copy()
                    new_row['contexts'] = f"Context {i+1}: {str(context)[:200]}..."
                    expanded_rows.append(new_row)
            else:
                new_row = row.copy()
                new_row['contexts'] = "No contexts"
                expanded_rows.append(new_row)
        else:
            # If no 'contexts' column originally, keep row as-is (no expansion)
            expanded_rows.append(row.copy())

    expanded_df = pd.DataFrame(expanded_rows)

    # Get all columns in the DataFrame
    all_columns = list(expanded_df.columns)
    
    # If column_widths is provided, map them to columns
    width_mapping = {}
    if column_widths is not None:
        if len(column_widths) >= len(all_columns):
            # Use provided widths for each column
            for i, col in enumerate(all_columns):
                width_mapping[col] = column_widths[i] if i < len(column_widths) else 10  # Fallback to 10%
        else:
            # If not enough widths provided, auto-distribute
            width_per_col = 100 / len(all_columns)
            for col in all_columns:
                width_mapping[col] = width_per_col
    else:
        # Auto-distribute widths if no column_widths provided
        width_per_col = 100 / len(all_columns)
        for col in all_columns:
            width_mapping[col] = width_per_col

    # Define color scale for metrics (green = good, yellow = medium, red = poor)
    def get_metric_color(value):
        if value >= 0.8:
            return '#d4edda'  # Light green
        elif value >= 0.5:
            return '#fff3cd'  # Light yellow
        else:
            return '#f8d7da'  # Light red

    # Create HTML table manually for better control
    html = '<div style="margin: 20px 0; font-family: Arial, sans-serif; overflow-x: auto;">\n'
    html += '<table style="border-collapse: collapse; width: 100%; border: 1px solid black; font-size: 12px;">\n'
    
    # Header row
    html += '  <thead>\n    <tr>\n'
    for col in expanded_df.columns:
        width = width_mapping.get(col, 10)  # Default 10% if not in mapping
        html += f'      <th style="border: 1px solid black; padding: 8px; text-align: center; background-color: #f8f9fa; color: black; font-weight: bold; width: {width}%;">{col}</th>\n'
    html += '    </tr>\n  </thead>\n'
    
    # Data rows
    html += '  <tbody>\n'
    for idx, row in expanded_df.iterrows():
        html += '    <tr>\n'
        for col in expanded_df.columns:
            value = row[col]
            width = width_mapping.get(col, 10)
            
            if col in metric_columns:
                # Apply metric-specific styling
                metric_val = float(row[col]) if pd.notna(row[col]) else 0
                color = get_metric_color(metric_val)
                html += f'      <td style="border: 1px solid black; padding: 8px; text-align: center; font-weight: bold; background-color: {color}; color: black; width: {width}%;">{metric_val}</td>\n'
            else:
                # Apply general styling for non-metric columns
                str_value = str(value)
                if len(str_value) > 150 and col != 'contexts':  # Don't truncate contexts since they're already processed
                    str_value = str_value[:150] + "..."
                elif col == 'contexts' and len(str_value) > 300:
                    str_value = str_value[:300] + "..."
                
                html += f'      <td style="border: 1px solid black; padding: 8px; text-align: left; background-color: white; color: black; width: {width}%;">{str_value}</td>\n'
        html += '    </tr>\n'
    html += '  </tbody>\n'
    html += '</table>\n'

    # Add summary statistics (ordered by table column order)
    if any(col in results_df.columns for col in metric_columns):
        # Calculate averages from original data (not expanded)
        original_avg_scores = results_df[[col for col in metric_columns if col in results_df.columns]].mean()
        
        # Get metric columns in the order they appear in the table
        table_metric_cols = [col for col in all_columns if col in metric_columns]
        
        summary = f"<div style='margin-top: 15px; padding: 10px; background-color: #f8f9fa; " \
                  f"border-radius: 5px; border: 1px solid #dee2e6; color: black; text-align: right;'>"
        summary += "<strong>Average Scores:</strong> "
        
        # Order the averages based on table column order
        summary_items = []
        for col in table_metric_cols:
            avg_score = original_avg_scores[col]
            color_code = '#28a745' if avg_score >= 0.7 else '#ffc107' if avg_score >= 0.5 else '#dc3545'
            summary_items.append(f"{col}: <span style='color: {color_code}; font-weight: bold;'>{avg_score:.2f}</span>")
        
        summary += " | ".join(summary_items)
        summary += "</div>"
        html += summary
    
    html += '</div>'
    
    return HTML(html)