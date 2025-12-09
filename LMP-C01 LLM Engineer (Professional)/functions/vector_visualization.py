# vector_visualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def visualize_task_elements(key_tokens, key_vectors, key_scores, 
                          work_terms=None, wellbeing_terms=None, urgency_terms=None):
    """
    Visualizes task elements in a 2D space with semantic clustering.
    
    Args:
        key_tokens: List of tokens to visualize
        key_vectors: Corresponding vector representations (shape: [n, embedding_dim])
        key_scores: Importance scores for each token
        work_terms: List of terms that belong to WORK category
        wellbeing_terms: List of terms that belong to WELLBEING category
        urgency_terms: List of terms that belong to URGENCY category
    
    Returns:
        None (displays the plot)
    """
    # Set default cluster terms if not provided
    if work_terms is None:
        work_terms = ['report', 'task', 'assignment', 'project', 'analysis', 'deadline', 'due', 'submit']
    if wellbeing_terms is None:
        wellbeing_terms = ['gym', 'exercise', 'workout', 'burnout', 'stress', 'energy', 'balance', 'happy', 'fit']
    if urgency_terms is None:
        urgency_terms = ['tomorrow', 'today', 'now', 'need', 'must', 'should', 'due', 'immediate', 'urgent', 'soon']
    
    # Convert to lowercase for case-insensitive matching
    work_terms = [term.lower() for term in work_terms]
    wellbeing_terms = [term.lower() for term in wellbeing_terms]
    urgency_terms = [term.lower() for term in urgency_terms]
    
    # Create token positions
    token_positions = {}
    for i, token in enumerate(key_tokens):
        x, y = key_vectors[i, 0], key_vectors[i, 1]
        token_positions[token] = (x, y)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot key tokens with size based on importance
    for i, token in enumerate(key_tokens):
        x, y = token_positions[token]
        size = 100 + key_scores[i] * 150  # Size proportional to importance
        token_lower = token.lower()
        
        # Color code based on semantic patterns
        if any(term in token_lower for term in work_terms):
            color = '#FF6B6B'  # Red for work
        elif any(term in token_lower for term in wellbeing_terms):
            color = '#4ECDC4'  # Teal for wellbeing
        elif any(term in token_lower for term in urgency_terms):
            color = '#FFD166'  # Yellow for urgency
        else:
            color = '#808080'  # Gray for neutral
        
        plt.scatter(x, y, s=size, color=color, edgecolor='k', linewidth=1.5, alpha=0.8)
        plt.annotate(f"{token}\n({x:.2f}, {y:.2f})", 
                    (x, y),
                    xytext=(8, 8),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold')

    plt.xlabel('Vector Dimension 1', fontweight='bold')
    plt.ylabel('Vector Dimension 2', fontweight='bold')
    plt.title('How TaskFriend "Sees" Your Most Important Task Elements', 
              fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', 
               markersize=10, label='WORK'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', 
               markersize=10, label='WELLBEING'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD166', 
               markersize=10, label='URGENCY')
    ]
    plt.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.show()

    
# vector search visualization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def plot_vector_search(embed_model, docs, query):
    """
    Visualizes vector search in 2D:
    - Projects embeddings via PCA
    - Plots arrows from origin
    - Labels with truncated text (≤10 chars)
    - Highlights angle between query and top match

    Args:
        embed_model: LlamaIndex-compatible embedder with get_text_embedding()
        docs: List of document strings
        query: Query string
    """
    # Step 1: Generate embeddings
    print("🧠 Generating embeddings using DashScope...")
    try:
        # Get embeddings for all texts
        all_texts = docs + [query]
        embeddings = np.array([embed_model.get_text_embedding(text) for text in all_texts])
    except Exception as e:
        raise RuntimeError(f"Embedding generation failed: {e}")

    # Step 2: Reduce to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    query_vec_2d = embeddings_2d[-1]
    doc_vecs_2d = embeddings_2d[:-1]

    # Step 3: Compute cosine similarities
    query_orig = embeddings[-1].reshape(1, -1)
    doc_orig = embeddings[:-1]
    similarities = cosine_similarity(query_orig, doc_orig)[0]
    top_idx = np.argmax(similarities)

    # Step 4: Create short labels (max 10 characters)
    def shorten(text, max_len=10):
        clean = text.strip().strip(".,!?\"'")
        if len(clean) <= max_len:
            return clean
        else:
            return clean[:max_len] + "..."

    labels = [shorten(doc) for doc in docs] + ["query"]

    # Step 5: Plot
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Normalize arrow sizes based on vector length (optional, for visual balance)
    max_norm = np.max(np.linalg.norm(embeddings_2d, axis=1)) + 1e-6
    scale_factor = 0.8  # Overall control for arrow size

    # Draw arrows for documents
    cmap = plt.cm.viridis
    norm_sim = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-8)
    colors = cmap(norm_sim)

    for i, vec in enumerate(doc_vecs_2d):
        # Scale arrow size based on vector magnitude to avoid distortion
        norm = np.linalg.norm(vec)
        head_width = 0.03 * (norm / max_norm) * scale_factor
        head_length = 0.04 * (norm / max_norm) * scale_factor
        linewidth = 1.2 * scale_factor

        ax.arrow(0, 0, vec[0], vec[1],
                 head_width=head_width, head_length=head_length,
                 fc=colors[i], ec='k', linewidth=linewidth, alpha=0.8, zorder=3)

        # Label near tip
        label_pos = vec * 1.12
        plt.text(label_pos[0], label_pos[1], labels[i],
                 fontsize=9, ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[i], edgecolor='k', alpha=0.9, linewidth=0.5))

    # Draw arrow for query (slightly more prominent but still modest)
    norm_query = np.linalg.norm(query_vec_2d)
    head_width_q = 0.035 * (norm_query / max_norm) * scale_factor
    head_length_q = 0.05 * (norm_query / max_norm) * scale_factor
    linewidth_q = 1.5 * scale_factor

    ax.arrow(0, 0, query_vec_2d[0], query_vec_2d[1],
             head_width=head_width_q, head_length=head_length_q,
             fc='gold', ec='orange', linewidth=linewidth_q, alpha=0.9, zorder=10)

    plt.text(query_vec_2d[0]*1.2, query_vec_2d[1]*1.2, "Q",
             fontsize=10, fontweight='bold', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", edgecolor="orange", alpha=0.9))

    # Optional: Draw angle arc between query and top match
    top_doc_vec = doc_vecs_2d[top_idx]
    angle1 = np.arctan2(query_vec_2d[1], query_vec_2d[0])
    angle2 = np.arctan2(top_doc_vec[1], top_doc_vec[0])
    min_angle = min(angle1, angle2)
    max_angle = max(angle1, angle2)

    from matplotlib.patches import Arc
    arc_radius = min(np.linalg.norm(query_vec_2d), np.linalg.norm(top_doc_vec)) * 0.4
    arc = Arc((0, 0), arc_radius*2, arc_radius*2,
              theta1=np.degrees(min_angle), theta2=np.degrees(max_angle),
              color='gray', lw=1.5, linestyle='--', alpha=0.7, zorder=5)
    ax.add_patch(arc)

    # Label angle
    mid_angle = (min_angle + max_angle) / 2
    label_x = np.cos(mid_angle) * arc_radius * 1.3
    label_y = np.sin(mid_angle) * arc_radius * 1.3
    cos_sim = similarities[top_idx]
    angle_deg = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
    ax.annotate(f"{angle_deg:.0f}°", (label_x, label_y),
                fontsize=10, color='gray',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='gray', linewidth=0.5))

    # Styling
    margin = 0.4
    plt.xlim(embeddings_2d[:, 0].min() - margin, embeddings_2d[:, 0].max() + margin)
    plt.ylim(embeddings_2d[:, 1].min() - margin, embeddings_2d[:, 1].max() + margin)
    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5, zorder=1)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5, zorder=1)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel("Semantic Dimension 1 (PCA)", fontweight='bold', fontsize=11)
    plt.ylabel("Semantic Dimension 2 (PCA)", fontweight='bold', fontsize=11)
    plt.title("Cosine Similarity in Action", fontsize=14, fontweight='bold', pad=20)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=similarities.min(), vmax=similarities.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.03)
    cbar.set_label('Cosine Similarity to Query', rotation=270, labelpad=20, fontsize=10)

    plt.tight_layout()
    plt.show()

    # Print results
    print("\n🔍 Top Matches:")
    ranked = sorted(enumerate(similarities), key=lambda x: -x[1])
    for rank, (i, sim) in enumerate(ranked, 1):
        print(f"  {rank}. [{sim:.3f}] {docs[i]}")