# openai_service.py
# ─────────────────────────────────────────────
# Handles all communication with Mistral model
# hosted on NVIDIA API (integrate.api.nvidia.com)
# Used by all 3 core agents and 3 portal agents.
# Includes retry logic for rate limit errors.
# ─────────────────────────────────────────────

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
INVOKE_URL     = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1") + "/chat/completions"
MODEL          = os.getenv("NVIDIA_MODEL", "mistralai/mistral-medium-3.5-128b")

# ── Core functions ────────────────────────────────────────────────────────────

def ask_ai(system_prompt: str, user_message: str, max_tokens: int = 1000, force_json: bool = False) -> str:
    """Send a single question to Mistral via NVIDIA API."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]
    return _call_nvidia(messages, max_tokens, force_json=force_json)


def ask_ai_with_history(messages: list, max_tokens: int = 1000) -> str:
    """
    Send a full conversation history to Mistral via NVIDIA API.
    Used by the user chatbot (user_agent.py) so the AI
    remembers previous messages in the conversation.

    Args:
        messages: Full chat history as list of dicts:
            [
                {"role": "system",    "content": "You are..."},
                {"role": "user",      "content": "Is transit good?"},
                {"role": "assistant", "content": "Yes, there are..."},
                {"role": "user",      "content": "What about parks?"},
            ]
        max_tokens: Max length of the response

    Returns:
        The AI response as a plain string
    """
    return _call_nvidia(messages, max_tokens)


# ── Internal helper ───────────────────────────────────────────────────────────

def _call_nvidia(messages: list, max_tokens: int, force_json: bool = False) -> str:
    """
    Makes the actual HTTP call to NVIDIA API.
    Handles retries on 429 rate limit errors.
    """
    MAX_RETRIES  = 5
    WAIT_SECONDS = 20

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept":        "application/json",
        "Content-Type":  "application/json",
    }

    payload = {
        "model":            MODEL,
        "messages":         messages,
        "max_tokens":       max_tokens,
        # Drop temperature to 0.1 if forcing JSON structures to prevent analytical drifts
        "temperature":      0.10 if force_json else 0.70,
        "top_p":            1.00,
        "reasoning_effort": "high",
        "stream":           False,
    }

    # If the calling agent specifically flagged force_json, explicitly inject the parameter
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
                    print(f"  ⏳ Rate limited. Waiting {WAIT_SECONDS}s (attempt {attempt}/{MAX_RETRIES})...")
                    time.sleep(WAIT_SECONDS)
                    continue
                else:
                    raise Exception("Rate limit: max retries exceeded.")

            if response.status_code == 401:
                raise Exception("Authentication failed. Check your NVIDIA_API_KEY in .env")

            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # NVIDIA reasoning model fallback validation
            if content is None or content == "":
                content = data["choices"][0]["message"].get("reasoning", "")
                
            return content

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                print(f"  ⏳ Timeout. Retrying ({attempt}/{MAX_RETRIES})...")
                time.sleep(10)
                continue
            raise Exception("NVIDIA API timed out after max retries.")

        except Exception as e:
            if "Rate limit" in str(e) or "429" in str(e):
                if attempt < MAX_RETRIES:
                    time.sleep(WAIT_SECONDS)
                    continue
            raise

# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🤖 Testing NVIDIA + Mistral connection...\n")

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

    print(" NVIDIA + Mistral connection working!")