# ./functions/html_table.py
from IPython.display import HTML
import pandas as pd

def create_html_table(data, title="Results Table", show_index=True, 
                     header_style="background-color: white; color: #1f77b4; font-weight: bold; border: 1px solid black; text-align: center;",
                     cell_style="background-color: white; color: black; border: 1px solid black; padding: 8px; text-align: left;",
                     column_widths=None):
    """
    Creates a styled HTML table from any dictionary or list of dictionaries with percentage-based column widths.
    
    Args:
         Dictionary or list of dictionaries containing the data
        title: Title to display above the table
        show_index: Whether to show the pandas index (row numbers)
        header_style: CSS style for header cells
        cell_style: CSS style for data cells
        column_widths: List of percentages for each column (e.g., [10, 20, 20, 35, 15])
                     If None, widths are auto-distributed
    
    Returns:
        HTML object containing the styled table
    """
    
    # Convert input to DataFrame
    if isinstance(data, dict):
        # If data is a single dictionary, convert values to lists
        df = pd.DataFrame({k: [v] if not isinstance(v, (list, tuple)) else v 
                          for k, v in data.items()})
    else:
        # Assume it's a list of dictionaries
        df = pd.DataFrame(data)
    
    # Calculate column widths
    total_columns = len(df.columns) + (1 if show_index else 0)
    if column_widths is None:
        # Auto-distribute widths
        width_per_col = 100 / total_columns
        column_widths = [width_per_col] * total_columns
    elif len(column_widths) != total_columns:
        raise ValueError(f"column_widths must have {total_columns} values for {total_columns} columns")
    
    # Create HTML table
    html = f"""
    <div style="margin: 20px 0; font-family: Arial, sans-serif;">
        <h3>{title}</h3>
        <table style="border-collapse: collapse; width: 100%; border: 1px solid black;">
            <thead>
                <tr>
    """
    
    # Add row number header if show_index is True
    if show_index:
        html += f'<th style="{header_style}; width: {column_widths[0]}%;">#</th>'
    
    # Add header row
    start_idx = 0 if show_index else 0
    for i, col in enumerate(df.columns):
        width = column_widths[i + (1 if show_index else 0)]
        html += f'<th style="{header_style}; width: {width}%;">{col}</th>'
    
    html += """
                </tr>
            </thead>
            <tbody>
    """
    
    # Add data rows
    for idx, row in df.iterrows():
        html += "<tr>"
        
        # Add row number if show_index is True
        if show_index:
            html += f'<td style="{cell_style}; width: {column_widths[0]}%;">{idx + 1}</td>'
            
        # Add data cells
        for i, col in enumerate(df.columns):
            value = row[col]
            # Handle lists or other complex types by converting to string
            if isinstance(value, (list, tuple)):
                value = ", ".join(str(x) for x in value)
            elif isinstance(value, dict):
                value = str(value)
            width = column_widths[i + (1 if show_index else 0)]
            html += f'<td style="{cell_style}; width: {width}%;">{value}</td>'
        html += "</tr>"
    
    html += """
            </tbody>
        </table>
    </div>
    """
    
    return HTML(html)

# Example usage function
def display_test_results(query, answer, ground_truth, contexts, title="RAG Test Results"):
    """
    Specialized function for displaying RAG test results using the general table function.
    """
    data = {
        "Question": [query],
        "Answer": [answer],
        "Ground Truth": [ground_truth],
        "Retrieved Contexts": ["\n\n".join(contexts) if isinstance(contexts, list) else contexts]
    }
    
    return create_html_table(data, title=title)