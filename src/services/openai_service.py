# openai_service.py
# ─────────────────────────────────────────────
# Handles all communication with local Llama model
# hosted on the ASUS DGX via Docker/NIM.
# Used by all core agents and portal agents.
# Includes retry logic for rate limit errors.
# ─────────────────────────────────────────────

import os
import time
import requests
import re
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

# Point directly to the DGX local host port exposed by your Docker container
INVOKE_URL     = os.getenv("DGX_BASE_URL", "http://localhost:8000/v1") + "/chat/completions"

# Local endpoints ignore the key, but the HTTP standard still expects a Bearer token string
DGX_API_KEY    = os.getenv("DGX_API_KEY", "dummy-local-key")

# The exact model namespace we loaded into the NIM container
MODEL          = os.getenv("DGX_MODEL", "meta/llama-3.1-8b-instruct")

# ── Core functions ────────────────────────────────────────────────────────────

def ask_ai(system_prompt: str, user_message: str, max_tokens: int = 1000) -> str:
    messages = [
        {
            "role":    "system",
            "content": (
                "RULE: Output the final answer only. "
                "No reasoning. No thinking. Start with the answer. "
                "\n\n" + system_prompt
            )
        },
        {
            "role":    "user",
            "content": user_message
        },
    ]
    raw = _call_dgx(messages, max_tokens)
    return clean_thinking(raw)


def ask_ai_with_history(messages: list, max_tokens: int = 1000) -> str:
    """
    Send a full conversation history to Llama via local DGX.
    Automatically strips thinking/reasoning from response.
    """
    # Inject hard stop instruction into system message
    enforced = []
    injected = False

    for msg in messages:
        if msg["role"] == "system" and not injected:
            enforced.append({
                "role":    "system",
                "content": (
                    "RULE: Output the final answer only. "
                    "No reasoning steps. No thinking. No self-talk. "
                    "Start immediately with the answer. "
                    "\n\n" + msg["content"]
                )
            })
            injected = True
        else:
            enforced.append(msg)

    raw = _call_dgx(enforced, max_tokens)
    return clean_thinking(raw)


def clean_thinking(text: str) -> str:
    """
    Remove all reasoning/thinking from LLM responses.
    Cuts the response at the first line that looks like a real answer.
    """
    # Remove XML think blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    lines = text.strip().split("\n")

    # Phrases that indicate reasoning not answer
    reasoning_phrases = [
        "the user is asking", "the user wants", "i need to find",
        "looking at the data", "i can see", "i see that", "let me",
        "based on the data", "from the data", "the data shows",
        "i need to check", "i should", "i will", "first i",
        "looking at", "the question is", "they want", "they've",
        "the system has", "the question asks", "the answer is",
        "so the answer", "wait i need", "but i need", "but the user",
        "however i", "so i need", "the area would be",
        "from the ward", "from the ttc", "from the neighbourhood",
        "let me check", "let me find", "let me look",
        "i need to identify", "i need to be",
        "looking at the ward", "looking at the ttc",
        "looking at the neighbourhood", "looking at the all",
    ]

    result      = []
    found_start = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if found_start:
                result.append("")
            continue

        lower = stripped.lower()

        # Skip reasoning lines
        if any(lower.startswith(p) for p in reasoning_phrases):
            continue

        # Skip lines that look like data field listings
        if re.match(r'^[-*]\s+(score|stops|rating|has_|total_|ward|ttc)', lower):
            continue

        # Skip lines with colons that look like raw data
        if re.match(r'^(score|stops|rating|parks|businesses|bike):', lower):
            continue

        # Skip numbered reasoning lists
        if re.match(r'^\d+\.\s+(there|the|i|looking|based|from)', lower):
            continue

        found_start = True
        result.append(stripped)

    cleaned = "\n".join(result).strip()

    # Last resort -- if still starts with reasoning, take last paragraph
    if cleaned and any(
        cleaned.lower().startswith(p) for p in reasoning_phrases
    ):
        paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        if paragraphs:
            cleaned = paragraphs[-1]

    return cleaned if cleaned else text.strip()

# ── Internal helper ───────────────────────────────────────────────────────────

def _call_dgx(messages: list, max_tokens: int, force_json: bool = False) -> str:
    """
    Makes the actual HTTP call to the local DGX API.
    Handles retries on 429 rate limit errors.
    """
    MAX_RETRIES  = 5
    WAIT_SECONDS = 20

    headers = {
        "Authorization": f"Bearer {DGX_API_KEY}",
        "Accept":        "application/json",
        "Content-Type":  "application/json",
    }

    payload = {
        "model":       MODEL,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": 0.10 if force_json else 0.70,
        "top_p":       1.00,
        "stream":      False,
    }

    if force_json:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                INVOKE_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )

            if response.status_code == 429:
                if attempt < MAX_RETRIES:
                    print(f"  Rate limited. Waiting {WAIT_SECONDS}s "
                          f"(attempt {attempt}/{MAX_RETRIES})...")
                    time.sleep(WAIT_SECONDS)
                    continue
                else:
                    raise Exception("Rate limit: max retries exceeded.")

            if response.status_code == 401:
                raise Exception(
                    "Authentication failed. The local container is rejecting the dummy key."
                )

            response.raise_for_status()

            data    = response.json()
            content = data["choices"][0]["message"]["content"]

            if content is None or content == "":
                content = data["choices"][0]["message"].get("reasoning", "")

            # Strip thinking text before returning
            return clean_thinking(content)

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                print(f"  Timeout. Retrying ({attempt}/{MAX_RETRIES})...")
                time.sleep(10)
                continue
            raise Exception("Local DGX API timed out after max retries. Check if the Docker container is running.")

        except Exception as e:
            if "Rate limit" in str(e) or "429" in str(e):
                if attempt < MAX_RETRIES:
                    time.sleep(WAIT_SECONDS)
                    continue
            raise

# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🤖 Testing Local DGX + Llama connection...\n")

    print("Test 1 — Single question:")
    answer = ask_ai(
        system_prompt="You are a Toronto city planning assistant.",
        user_message=(
            "In one sentence, what is the biggest challenge "
            "for urban planning in Toronto?"
        ),
        max_tokens=100,
    )
    print(f"✅ Response: {answer}\n")

    print("Test 2 — Chat with history:")
    history = [
        {
            "role": "system",
            "content": "You are a helpful Toronto neighbourhood assistant.",
        },
        {
            "role": "user",
            "content": "How is transit in Humber Bay?",
        },
        {
            "role": "assistant",
            "content": (
                "Humber Bay has decent TTC bus coverage "
                "but limited subway access."
            ),
        },
        {
            "role": "user",
            "content": "What about parks nearby?",
        },
    ]
    answer2 = ask_ai_with_history(history, max_tokens=100)
    print(f" Response: {answer2}\n")

    print("✅ Local DGX connection working!")