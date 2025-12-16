# token_table.py
import pandas as pd
from IPython.display import HTML
from collections import Counter

def create_token_table(tokens, token_ids):
    """
    Creates a styled HTML table showing tokenization results with color-coded token IDs.
    Only highlights token ID groups that appear at least twice in the token sequence.
    
    Args:
        tokens: List of token strings
        token_ids: List of corresponding token IDs
    
    Returns:
        HTML object containing the styled table
    """
    # Add quotes to tokens to show whitespace explicitly
    quoted_tokens = [f"'{t}'" for t in tokens]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Pos#': range(1, len(tokens) + 1),
        'Token': quoted_tokens,
        'Token ID': token_ids
    })
    
    # Count token ID frequencies
    token_id_counts = Counter(token_ids)
    
    # Identify token IDs that appear at least twice (the key requirement)
    frequent_ids = [tid for tid, count in token_id_counts.items() if count >= 2]
    
    # Limit to top 5 most frequent token IDs (that appear at least twice)
    top_ids = sorted(frequent_ids, key=lambda x: token_id_counts[x], reverse=True)[:5]
    
    # Generate color palette for top token IDs (limit to 5)
    color_palette = _generate_color_palette(len(top_ids))
    id_to_color = {token_id: color_palette[i] for i, token_id in enumerate(top_ids)}
    
    # Apply styling with borders and proper formatting
    def style_row(row):
        styles = []
        
        # Pos# column - default styling
        styles.append('background-color: white; color: black; border: 1px solid #ddd; padding: 5px; text-align: center')
        
        # Token column - highlight only if in top frequent IDs
        if row['Token ID'] in id_to_color:
            styles.append(f'background-color: {id_to_color[row["Token ID"]]}; color: black; border: 1px solid #ddd; padding: 5px; font-family: monospace')
        else:
            styles.append('background-color: white; color: black; border: 1px solid #ddd; padding: 5px; font-family: monospace')
        
        # Token ID column - highlight only if in top frequent IDs
        if row['Token ID'] in id_to_color:
            styles.append(f'background-color: {id_to_color[row["Token ID"]]}; color: black; border: 1px solid #ddd; padding: 5px; text-align: center')
        else:
            styles.append('background-color: white; color: black; border: 1px solid #ddd; padding: 5px; text-align: center')
        
        return styles
    
    # Apply styling
    styled_df = df.style.apply(style_row, axis=1)
    
    # Add color key explanation (only if there are frequent IDs)
    color_key = ""
    if top_ids:
        color_key = "<div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee;'>"
        color_key += "<strong>Repeated tokens (shows max 5 groups):</strong> "
        color_key += " ".join([f"<span style='background-color: {id_to_color[tid]}; padding: 2px 5px; margin-right: 5px;'>Token ID: {tid} ({token_id_counts[tid]}x)</span>" 
                              for tid in top_ids])
        color_key += "</div>"
    else:
        color_key = "<div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee; color: #666;'>"
        color_key += "No repeated token patterns found (all tokens appear only once)"
        color_key += "</div>"
    
    # Return HTML with proper styling
    return HTML(styled_df.set_table_styles([
        {'selector': 'th', 'props': [('border', '1px solid #ddd'), ('padding', '8px'), ('text-align', 'left')]},
        {'selector': 'td', 'props': [('border', '1px solid #ddd'), ('padding', '5px')]},
        {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '100%'), ('margin', '15px 0')]}
    ]).to_html() + color_key)

def _generate_color_palette(n):
    """Generate a palette of 5 visually distinct soft colors"""
    base_colors = [
        '#FFB3BA',  # Light red
        '#BAE1FF',  # Light blue
        '#BAFFC9',  # Light green
        '#FFDFBA',  # Light orange
        '#D4BAFF'   # Light purple
    ]
    
    # Return only the first n colors with 40% opacity
    return [color + "66" for color in base_colors[:n]]
