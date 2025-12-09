# visualize_attention.py
import torch
import numpy
from transformers import AutoTokenizer, AutoModel
from bertviz import head_view, model_view, neuron_view

def visualize_attention(query):
    """
    Visualize attention patterns for a TaskFriend query using BertViz
    
    Args:
        query: User query to analyze
    """
    print(f"Analyzing attention patterns for: '{query}'")
    print("="*60)
    
    # Load model and tokenizer
    print("Loading model for attention analysis...", end="", flush=True)
    start_time = time.time()
    
    model_name = "bert-base-uncased"  # BertViz works best with BERT models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    
    load_time = time.time() - start_time
    print(f" Done ✓ ({load_time:.1f} seconds)")
    
    # Tokenize input with special tokens
    print("Tokenizing query...", end="", flush=True)
    start_time = time.time()
    
    inputs = tokenizer(query, return_tensors="pt", add_special_tokens=True)
    token_ids = inputs['input_ids'][0].numpy()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")
    print("-"*60)
    
    load_time = time.time() - start_time
    print(f" Done ✓ ({load_time:.1f} seconds)")
    
    # Get model outputs with attention
    print("Getting model outputs with attention...", end="", flush=True)
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    load_time = time.time() - start_time
    print(f" Done ✓ ({load_time:.1f} seconds)")
    
    # Extract attention weights
    print("Extracting attention weights...", end="", flush=True)
    start_time = time.time()
    
    attentions = outputs.attentions  # Tuple of attention tensors (layers)
    
    load_time = time.time() - start_time
    print(f" Done ✓ ({load_time:.1f} seconds)")
    
    # Convert to numpy for BertViz
    print("Converting to numpy for BertViz...", end="", flush=True)
    start_time = time.time()
    
    attention_data = {
        'all_attentions': [att.detach().numpy() for att in attentions],
        'tokens': tokens
    }
    
    load_time = time.time() - start_time
    print(f" Done ✓ ({load_time:.1f} seconds)")
    
    print("\nGenerating attention visualizations...")
    
    # 1. Head View - focus on a specific layer and head
    print("\n" + "="*60)
    print("HEAD VIEW: Detailed attention for specific head (Layer 8, Head 5)")
    print("Shows how a single attention head connects tokens")
    print("="*60)
    head_view(attention_data['all_attentions'], tokens, layer=8, head=5)
    
    # 2. Model View - overview of attention across all layers
    print("\n" + "="*60)
    print("MODEL VIEW: Attention patterns across all layers and heads")
    print("Shows the complete attention landscape of the model")
    print("="*60)
    model_view(attention_data['all_attentions'], tokens)
    
    return {
        'attention_data': attention_data,
        'tokens': tokens,
        'model': model,
        'tokenizer': tokenizer
    }