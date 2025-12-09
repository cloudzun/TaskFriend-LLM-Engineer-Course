# ./functions/llm_utils.py

import time
import os
from openai import OpenAI

# Initialize client
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

def get_qwen_stream_response(query, system_prompt="", temperature=0.5, top_p=0.9):
    print("📡 Calling LLM (Qwen)... ", flush=True)
    start = time.time()

    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=temperature,
            top_p=top_p,
            stream=True
        )

        content = ""

        for chunk in response:
            token = chunk.choices[0].delta.content
            if token:
                content += token
                print(token, end="", flush=True)  # ←←← Print each token immediately

        elapsed = time.time() - start
        print(f"\n⏱️  Total time: {elapsed:.2f}s\n")
        return content, elapsed

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ LLM Error: {e}")
        print(f"⏱️  Time: {elapsed:.2f}s\n")
        return "Error", elapsed


def get_hardcoded_response(query, hardcoded_responses):
    """
    Returns a hardcoded response if query matches.
    `hardcoded_responses`: dict mapping lowercase queries to responses.
    """
    query = query.strip().lower()
    if query in hardcoded_responses:
        print("⚡ Using hardcoded response... ")
        start = time.time()
        time.sleep(0.0001)  # Simulate minimal overhead
        elapsed = time.time() - start
        response = hardcoded_responses[query]
        print(f"💬 Hardcoded: {response}")
        print(f"⏱️  Time: {elapsed:.6f} seconds\n")
        return response, elapsed
    else:
        raise KeyError(f"No hardcoded response for '{query}'")


def get_precomputed_response(query, precomputed_knowledge_base):
    """
    Looks up a precomputed response from a knowledge base.
    `precomputed_knowledge_base`: dict of precomputed answers.
    """
    query = query.strip().lower()
    print("🗄️  Looking up precomputed response... ", end="", flush=True)
    start = time.time()
    time.sleep(0.005)  # Simulate DB/cache lookup

    if query in precomputed_knowledge_base:
        elapsed = time.time() - start
        response = precomputed_knowledge_base[query]
        print(f"\r💬 Precomputed: {response[:80]}...")
        print(f"⏱️  Time: {elapsed:.4f} seconds\n")
        return response, elapsed
    else:
        raise KeyError(f"No precomputed answer for '{query}'")


def benchmark_responses(hardcoded_responses, precomputed_knowledge_base, queries):
    """
    Runs performance comparison between hardcoded, precomputed, and LLM responses.
    All inputs are passed as parameters.
    """
    print("🚀 PERFORMANCE COMPARISON: LLM vs Hardcoded vs Precomputed\n")
    print("-" * 80)

    results = []

    # 1. Hardcoded
    print("✅ 1. Hardcoded Response")
    try:
        resp, t = get_hardcoded_response(queries["hardcoded"], hardcoded_responses)
        results.append({
            "Method": "Hardcoded",
            "Time (s)": t,
            "Query": queries["hardcoded"],
            "Response": resp
        })
    except Exception as e:
        print(f"❌ Failed: {e}\n")
        results.append({
            "Method": "Hardcoded",
            "Time (s)": 0.0,
            "Query": queries["hardcoded"],
            "Response": f"Error: {e}"
        })
    print("-" * 80)

    # 2. Precomputed
    print("✅ 2. Precomputed Response")
    try:
        resp, t = get_precomputed_response(queries["precomputed"], precomputed_knowledge_base)
        results.append({
            "Method": "Precomputed",
            "Time (s)": t,
            "Query": queries["precomputed"],
            "Response": resp
        })
    except Exception as e:
        print(f"❌ Failed: {e}\n")
        results.append({
            "Method": "Precomputed",
            "Time (s)": 0.0,
            "Query": queries["precomputed"],
            "Response": f"Error: {e}"
        })
    print("-" * 80)

    # 3. LLM (Streaming)
    print("✅ 3. LLM-Generated Response")
    system_prompt = "You are a helpful assistant. Keep answers clear and concise."
    try:
        resp, t = get_qwen_stream_response(queries["llm"], system_prompt, temperature=0.5, top_p=0.9)
        results.append({
            "Method": "LLM",
            "Time (s)": t,
            "Query": queries["llm"],
            "Response": resp
        })
    except Exception as e:
        print(f"❌ LLM call failed: {e}\n")
        results.append({
            "Method": "LLM",
            "Time (s)": 0.0,
            "Query": queries["llm"],
            "Response": f"Error: {e}"
        })
    print("-" * 80)

    # --- Summary Table ---
    if not results:
        print("❌ No results to display.")
        return

    print("\n📊 SUMMARY: Response Times & Answers\n")
    print(f"{'Method':<12} {'Time (s)':<10} {'Response':<50} {'Query'}")
    print("-" * 100)

    for r in results:
        raw_resp = r.get("Response", "Unknown")
        truncated_resp = (str(raw_resp)[:45] + "...") if len(str(raw_resp)) > 45 else raw_resp
        print(f"{r['Method']:<12} {r['Time (s)']:<10.6f} {truncated_resp:<50} {r['Query']}")

    # Speedup relative to LLM
    llm_entry = next((r for r in results if r["Method"] == "LLM"), None)
    if llm_entry and llm_entry["Time (s)"] > 0:
        llm_time = llm_entry["Time (s)"]
        for r in results:
            if r["Method"] != "LLM" and r["Time (s)"] > 0:
                speedup = llm_time / r["Time (s)"]
                print(f"⚡ {r['Method']} was {speedup:.1f}x faster than LLM")
    else:
        print("⚠️  LLM response failed or took 0 seconds — cannot calculate speedup.")