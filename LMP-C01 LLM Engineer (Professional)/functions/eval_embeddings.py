# evaluate_embeddings.py
import os
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness, context_recall, context_precision, faithfulness
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.dashscope import DashScopeEmbedding
from langchain_community.embeddings import DashScopeEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

def evaluate_embedding_models(persist_path, embedding_models, question, ground_truth, llm, ragas_llm):
    """
    Evaluate multiple DashScope embedding models on a single QA pair.
    
    Args:
        persist_path (str): Path to LlamaIndex storage
        embedding_models (List[str]): e.g., ["text-embedding-v3", "text-embedding-v4"]
        question (str): User question
        ground_truth (str): Expected answer
        llm: LlamaIndex-compatible LLM (e.g., Settings.llm)
        ragas_llm: Ragas-compatible LLM (e.g., LangChain-wrapped LLM)
        
    Returns:
        List[dict]: Evaluation results
    """
    # Load nodes — your original code
    storage_context = StorageContext.from_defaults(persist_dir=persist_path)
    node_ids = list(storage_context.docstore.docs.keys())
    nodes = storage_context.docstore.get_nodes(node_ids)

    test_cases = [{"question": question, "ground_truth": ground_truth}]
    evaluation_results = []

    for model_name in embedding_models:
        clean_model = model_name.strip()
        print(f"🧪 Evaluating with: {clean_model}")
        
        try:
            # Embedding model for retrieval
            embed_model = DashScopeEmbedding(
                model_name=clean_model,
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                encoding_format="float"
            )
            
            # Rebuild index
            index = VectorStoreIndex(nodes, embed_model=embed_model)
            query_engine = index.as_query_engine(
                streaming=False,  # safer for evaluation
                llm=llm,
                similarity_top_k=3
            )
            
            # Simple query (no external run_test_cases needed)
            response = query_engine.query(question)
            answer = str(response)
            contexts = [node.get_content() for node in response.source_nodes] if hasattr(response, 'source_nodes') else [""]

            # Print sample
            print(f"\n📝 Output for '{clean_model}'")
            print("-" * 50)
            print(f"Q: {question}")
            print(f"A: {answer}")
            print(f"GT: {ground_truth}")
            print("-" * 50)

            # Build dataset
            eval_dataset = Dataset.from_dict({
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth]
            })

            # Ragas-compatible DashScope embeddings
            ragas_embeddings = LangchainEmbeddingsWrapper(
                DashScopeEmbeddings(
                    model=clean_model,
                    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
                )
            )

            # Evaluate
            results = evaluate(
                dataset=eval_dataset,
                metrics=[answer_correctness, context_recall, context_precision, faithfulness],
                llm=ragas_llm,
                embeddings=ragas_embeddings
            )
            
            result_dict = results.to_pandas().to_dict('records')[0]
            result_dict.update({
                'embedding_model': clean_model,
                'question': question,
                'answer': answer,
                'ground_truth': ground_truth
            })
            evaluation_results.append(result_dict)
            print(f"✅ Done\n")
            
        except Exception as e:
            print(f"❌ Error with {clean_model}: {e}\n")
            evaluation_results.append({
                'embedding_model': clean_model,
                'question': question,
                'answer': str(e),
                'ground_truth': ground_truth,
                'answer_correctness': None,
                'context_recall': None,
                'context_precision': None,
                'faithfulness': None
            })

    return evaluation_results
