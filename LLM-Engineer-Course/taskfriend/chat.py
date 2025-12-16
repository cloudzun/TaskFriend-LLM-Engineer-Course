#chat.py

import time
from typing import List, Dict, Any, Generator, Callable, Optional
from IPython.display import clear_output

# Import local modules
from . import config, context, utils

# Define a unified function signature for all LLM/RAG functions
LLMFunction = Callable[
    [str, str, float, float, Optional[List[Dict[str, str]]]],  # query, system_prompt, temp, top_p, history
    Generator[str, None, None]
]


def chat_interface(
    full_conversation: List[Dict[str, str]],
    client: Any = None,
    query_engine: Any = None,
    use_context_window: bool = False,
    context_window: int = 2000,
    show_truncated: bool = False,
    summarize_dropped: bool = False,
    summarizer_fn: Optional[Callable[[str, Any], str]] = None,
    show_context_preview: bool = False,
    call_llm_fn: Optional[LLMFunction] = None,
    system_prompt_override: Optional[str] = None
) -> None:
    """
    Start an interactive chat session with TaskFriend.
    
    Args:
        full_conversation: List of message dicts with 'role' and 'content'
        client: LLM client for API calls
        query_engine: RAG query engine
        use_context_window: Whether to truncate conversation
        context_window: Max tokens to send to LLM
        show_truncated: Show dropped messages
        summarize_dropped: Summarize dropped messages
        summarizer_fn: Custom summarizer
        show_context_preview: Show what's sent to LLM
        call_llm_fn: Custom function to call LLM or RAG
        system_prompt_override: Custom system prompt
    """
    base_system_prompt = "You are TaskFriend, a helpful AI assistant for task management and productivity."
    system_prompt = system_prompt_override or base_system_prompt
    
    # Save original config to restore later
    old_config = {
        "USE_CONTEXT_WINDOW": config.USE_CONTEXT_WINDOW,
        "CONTEXT_WINDOW": config.CONTEXT_WINDOW,
        "SHOW_TRUNCATED": config.SHOW_TRUNCATED,
        "SUMMARIZE_DROPPED": config.SUMMARIZE_DROPPED,
        "SHOW_CONTEXT_PREVIEW": config.SHOW_CONTEXT_PREVIEW,
    }

    # Apply config overrides
    config.USE_CONTEXT_WINDOW = use_context_window
    config.CONTEXT_WINDOW = context_window
    config.SHOW_TRUNCATED = show_truncated
    config.SUMMARIZE_DROPPED = summarize_dropped
    config.SHOW_CONTEXT_PREVIEW = show_context_preview

    if not full_conversation:
        full_conversation.append({"role": "system", "content": system_prompt})

    try:
        print("🚀 Welcome to TaskFriend - Your AI Productivity Assistant")
        print("=" * 60)

        while True:
            user_input = input("📝 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "bye"]:
                print("\n👋 Ending conversation... See you next time!")
                break

            full_conversation.append({"role": "user", "content": user_input})

            try:
                clear_output(wait=True)

                # Show conversation history
                print("🚀 TaskFriend Conversation", flush=True)
                print("-" * 60, flush=True)
                for msg in full_conversation:
                    if msg["role"] == "user" and msg["content"] != system_prompt:
                        print(f"👤 You: {msg['content']}", flush=True)
                    elif msg["role"] == "assistant":
                        print(f"🤖 TaskFriend: {msg['content']}", flush=True)

                # Truncate conversation if needed
                truncated_context = context.truncate_conversation(
                    full_conversation,
                    client=client,
                    summarizer_fn=summarizer_fn
                )

                # Show context preview
                utils.print_context_preview(truncated_context)

                print("\n🤖 TaskFriend is thinking", end='', flush=True)
                for _ in range(3):
                    time.sleep(0.3)
                    print(".", end='', flush=True)
                print()

                # Extract system prompt
                system_msg = next(
                    (m["content"] for m in truncated_context if m["role"] == "system"),
                    system_prompt
                )

                # Call LLM function
                print("🤖 TaskFriend: ", end='', flush=True)
                full_response = ""

                if call_llm_fn:
                    try:
                        for content in call_llm_fn(
                            user_input,
                            system_msg,
                            config.TEMPERATURE,
                            config.TOP_P,
                            truncated_context
                        ):
                            full_response += content
                            print(content, end='', flush=True)
                        print()
                    except Exception as e:
                        print(f"\n❌ Streaming error: {e}")
                else:
                    # Fallback: Direct API call
                    try:
                        response = client.chat.completions.create(
                            model=config.MODEL,
                            messages=truncated_context,
                            temperature=config.TEMPERATURE,
                            top_p=config.TOP_P,
                            stream=True
                        )
                        for chunk in response:
                            content = chunk.choices[0].delta.content
                            if content:
                                full_response += content
                                print(content, end='', flush=True)
                        print()
                    except Exception as e:
                        print(f"\n❌ API Error: {e}")
                        full_response = "[API call failed]"

                # Save assistant response
                full_conversation.append({"role": "assistant", "content": full_response})

                # Show RAG sources if available
                if query_engine:
                    print("\n📚 Source References:")
                    print("-" * 60)
                    try:
                        raw_response = query_engine.query(user_input)
                        for i, source_node in enumerate(raw_response.source_nodes, 1):
                            print(f"Chunk {i}:")
                            print(source_node)
                            print()
                    except Exception as e:
                        print(f"[Error retrieving source: {e}]")

                print("\n" + "-" * 60)
                print("💡 Type your next message (or 'exit', 'bye' to end):")

            except Exception as e:
                print(f"\n❌ Error during interaction: {type(e).__name__}: {e}")

    finally:
        pass  # No config restore needed


# 🧰 Helper: Wrap streaming LLM functions
def wrap_streaming_for_chat(fn: Callable) -> LLMFunction:
    def wrapper(
        query: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        full_conversation: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, None]:
        # Build full query with history
        if full_conversation:
            history = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in full_conversation if msg["role"] != "system"
            ])
            query = f"{history}\nUser: {query}\nAssistant:"

        # Call original function
        yield from fn(query, system_prompt, temperature, top_p)

    return wrapper


def wrap_rag_for_chat(fn: Callable, query_engine: Any) -> LLMFunction:
    def wrapper(
        query: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        full_conversation: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, None]:
        # Build full context
        context = system_prompt
        if full_conversation:
            context += "\n\n" + "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in full_conversation if msg["role"] != "system"
            ])

        # Final prompt to send to RAG engine
        full_prompt = f"{context}\nUser: {query}\nAssistant:"

        # Call RAG function with full prompt
        answer = fn(full_prompt, query_engine)

        # Stream the answer back
        for token in answer.split(" "):
            yield token + " "
            time.sleep(0.05)  # Optional: simulate streaming

    return wrapper