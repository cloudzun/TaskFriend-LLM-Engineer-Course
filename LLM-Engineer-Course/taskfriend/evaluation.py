from typing import Any, Dict, List, Optional
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    context_recall,
    context_precision,
    faithfulness,
    answer_relevancy
)
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
import pandas as pd

class Evaluator:
    """Class for evaluating RAG system performance"""

    def __init__(
        self,
        llm_model: str = "qwen-plus",
        embedding_model: str = "text-embedding-v3",
        api_base: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key: Optional[str] = None
    ):
        from .config import DASHSCOPE_API_KEY
        self.api_key = api_key or DASHSCOPE_API_KEY

        self.llm = OpenAILike(
            model=llm_model,
            api_base=api_base,
            api_key=self.api_key,
            is_chat_model=True
        )

        self.embeddings = DashScopeEmbedding(
            model_name=embedding_model
        )

        self.default_metrics = [
            answer_correctness,
            context_recall,
            context_precision,
            faithfulness,
            answer_relevancy
        ]

    def evaluate_result(
        self,
        question: str,
        response: Any,
        ground_truth: str,
        metrics: Optional[List] = None
    ) -> pd.DataFrame:
        try:
            answer = response.response if hasattr(response, "response") else str(response)
            context = []
            if hasattr(response, "source_nodes"):
                context = [node.get_content() for node in response.source_nodes]

            data_samples = {
                'question': [question],
                'answer': [answer],
                'ground_truth': [ground_truth],
                'contexts': [context],
            }

            dataset = Dataset.from_dict(data_samples)
            eval_metrics = metrics if metrics else self.default_metrics

            score = evaluate(
                dataset=dataset,
                metrics=eval_metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )

            return score.to_pandas()

        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {str(e)}")

    def compare_models(
        self,
        documents: List[Any],
        questions: List[str],
        ground_truths: List[str],
        models_to_compare: Dict[str, Dict[str, Any]],
        splitter: Optional[Any] = None
    ) -> Dict[str, pd.DataFrame]:
        results = {}

        for model_name, model_config in models_to_compare.items():
            print(f"\n{'='*50}")
            print(f"🔍 Evaluating model: {model_name}")
            print(f"{'='*50}")

            if splitter:
                nodes = splitter.get_nodes_from_documents(documents)
                index = VectorStoreIndex(nodes, embed_model=model_config["embed_model"])
            else:
                index = VectorStoreIndex.from_documents(documents, embed_model=model_config["embed_model"])

            query_engine = index.as_query_engine(
                streaming=True,
                llm=model_config["llm"],
                similarity_top_k=model_config.get("similarity_top_k", 2)
            )

            model_results = []
            for i, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
                print(f"\nQuestion {i+1}/{len(questions)}: {question}")

                response = query_engine.query(question)
                scores = self.evaluate_result(question, response, ground_truth)
                model_results.append(scores)

                print("Evaluation scores:")
                print(scores)

            results[model_name] = pd.concat(model_results, ignore_index=True)

        return results

    def compare_embeddings(
        self,
        query: str,
        chunks: List[str],
        embedding_models: Dict[str, DashScopeEmbedding]
    ) -> pd.DataFrame:
        from numpy import dot
        from numpy.linalg import norm

        results = []

        for model_name, model in embedding_models.items():
            print(f"\n{'='*20} {model_name} {'='*20}")

            try:
                query_embedding = model.get_query_embedding(query)
            except:
                query_embedding = model.get_text_embedding(query)

            for i, chunk in enumerate(chunks):
                try:
                    chunk_embedding = model.get_text_embedding(chunk)
                    similarity = dot(query_embedding, chunk_embedding) / (norm(query_embedding) * norm(chunk_embedding))
                    results.append({
                        "model": model_name,
                        "chunk": i+1,
                        "similarity": float(similarity)
                    })
                except Exception as e:
                    print(f"Error calculating similarity for chunk {i+1}: {str(e)}")

        df = pd.DataFrame(results)
        return df.pivot(index="model", columns="chunk", values="similarity")
