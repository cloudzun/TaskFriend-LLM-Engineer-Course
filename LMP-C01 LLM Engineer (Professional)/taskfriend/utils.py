import sys
import logging
from . import config
from .config import CONTEXT_WINDOW, TOKEN_FACTOR, LOGGING_LEVEL

logging.basicConfig(level=LOGGING_LEVEL)

def estimate_tokens(text):
    """Estimate token count for text"""
    return max(1, len(text) // TOKEN_FACTOR)

def print_context_preview(messages):
    """Print a preview of the context being sent to the LLM"""
    from .config import SHOW_CONTEXT_PREVIEW

    if not SHOW_CONTEXT_PREVIEW:
        return

    print("=" * 50)
    print("ACTUAL CONTEXT SENT TO LLM")
    print("=" * 50)
    total = 0
    for i, msg in enumerate(messages):
        tokens = estimate_tokens(msg["content"])
        total += tokens
        content_preview = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
        role = msg["role"].capitalize()
        print(f"  [{i:2d}] {role:8} ({tokens:3d}t) → {content_preview}")

    print(f"Total tokens: {total}/{config.CONTEXT_WINDOW}")
    print("=" * 50)
    
def summarize_text(text, client, max_tokens=80):
    """Default fallback summarizer using the LLM"""
    from .config import MODEL, TEMPERATURE, TOP_P

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Summarize the following in one short paragraph"},
                {"role": "user", "content": text}
            ],
            temperature=TEMPERATURE,
            max_tokens=max_tokens,
            top_p=TOP_P,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Summary failed: {e}]"

def colorize(text, style="yellow"):
    """Add ANSI color codes to text for terminal output"""
    styles = {
        "bg_yellow": "\033[30;48;5;228m",
        "bg_gray":   "\033[30;48;5;245m",
        "bg_red":    "\033[37;48;5;196m",
        "bg_green":  "\033[30;48;5;119m",
        "bg_blue":   "\033[37;48;5;69m",
        "red":    "\033[38;5;196m",
        "green":  "\033[38;5;40m",
        "yellow": "\033[38;5;228m",
        "blue":   "\033[38;5;69m",
        "gray":   "\033[38;5;245m",
        "bold":   "\033[1m",
        "reset":  "\033[0m",
    }
    code = styles.get(style.lower(), "")
    return f"{code}{text}\033[0m"

def log_yellow(text):
    print(colorize(text, "bg_yellow"), file=sys.stderr, flush=True)

def log_gray(text):
    print(colorize(text, "bg_gray"), file=sys.stderr, flush=True)

def log_red(text):
    print(colorize(text, "bg_red"), file=sys.stderr, flush=True)

def log_green(text):
    print(colorize(text, "bg_green"), file=sys.stderr, flush=True)
