"""
LLM Router: Strictly alternates between Groq, OpenRouter, and Gemini on EVERY request.
Strategy:
  - Request % 3 == 1 → Groq (llama-3.3-70b-versatile)
  - Request % 3 == 2 → OpenRouter (meta-llama/llama-3.3-70b-instruct)
  - Request % 3 == 0 → Gemini (gemini-1.5-flash)
  - On any error from primary → immediately use next in chain
  - Minimum inter-call delay enforced per provider
  - Long documents: chunk + summarize, rotating provider per chunk

ZERO domain-specific keywords anywhere in this file.
"""
import os
import time
import threading
import logging
import json
from typing import Optional

import httpx
from groq import Groq
from groq import RateLimitError as GroqRateLimitError
import google.generativeai as genai

logger = logging.getLogger(__name__)

# Thread-safe request counter for strict rotation
_request_counter = 0
_counter_lock = threading.Lock()

# Per-provider rate limit tracking
_groq_last_call_time = 0.0
_openrouter_last_call_time = 0.0
_gemini_last_call_time = 0.0

GROQ_MIN_INTERVAL = 2.1        # seconds between Groq calls
OPENROUTER_MIN_INTERVAL = 1.0  # seconds between OpenRouter calls
GEMINI_MIN_INTERVAL = 0.6      # seconds between Gemini calls

# Chunk size for long document summarization
LONG_DOC_CHUNK_CHARS = 10000
LONG_DOC_THRESHOLD_CHARS = 12000


def _get_next_provider() -> str:
    """Strictly rotate between three providers"""
    global _request_counter
    with _counter_lock:
        _request_counter += 1
        rem = _request_counter % 3
        if rem == 1: return "groq"
        if rem == 2: return "openrouter"
        return "gemini"


def _wait_for_provider(provider: str):
    """Enforce minimum interval between calls to each provider"""
    global _groq_last_call_time, _openrouter_last_call_time, _gemini_last_call_time
    now = time.time()
    if provider == "groq":
        elapsed = now - _groq_last_call_time
        if elapsed < GROQ_MIN_INTERVAL:
            time.sleep(GROQ_MIN_INTERVAL - elapsed)
        _groq_last_call_time = time.time()
    elif provider == "openrouter":
        elapsed = now - _openrouter_last_call_time
        if elapsed < OPENROUTER_MIN_INTERVAL:
            time.sleep(OPENROUTER_MIN_INTERVAL - elapsed)
        _openrouter_last_call_time = time.time()
    else:
        elapsed = now - _gemini_last_call_time
        if elapsed < GEMINI_MIN_INTERVAL:
            time.sleep(GEMINI_MIN_INTERVAL - elapsed)
        _gemini_last_call_time = time.time()


def _call_groq(client: Groq, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=min(max_tokens, 8000),
        temperature=temperature,
    )
    return response.choices[0].message.content


def _call_openrouter(api_key: str, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
    """Call OpenRouter via httpx (OpenAI-compatible)"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "Antigravity RAG"
    }
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    with httpx.Client(timeout=45.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _call_gemini(model, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
    # Gemini 1.5/2.0 supports system instruction directly in model init or prompt
    # Using the combined direct prompt approach for compatibility here
    full = f"{system}\n\n{prompt}" if system else prompt
    resp = model.generate_content(
        full,
        generation_config=genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
    )
    return resp.text


class LLMRouter:
    def __init__(self):
        self._groq_key = os.getenv("GROQ_API_KEY")
        self._openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self._gemini_key = os.getenv("GEMINI_API_KEY")

        self._groq_client = Groq(api_key=self._groq_key)
        genai.configure(api_key=self._gemini_key)
        self._gemini_model = genai.GenerativeModel("gemini-2.5-flash")

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
        force_provider: Optional[str] = None,
    ) -> dict:
        """
        Complete a prompt using 3-way rotation.
        Priority chain for current session: Primary -> (Remaining available)
        """
        primary = force_provider or _get_next_provider()
        
        # Define chain based on primary
        if primary == "groq":
            chain = ["groq", "openrouter", "gemini"]
        elif primary == "openrouter":
            chain = ["openrouter", "gemini", "groq"]
        else:
            chain = ["gemini", "groq", "openrouter"]

        last_error = None
        for provider in chain:
            try:
                _wait_for_provider(provider)
                if provider == "groq":
                    if not self._groq_key or "your_groq" in self._groq_key:
                        raise ValueError("Groq API key not configured")
                    text = _call_groq(self._groq_client, prompt, system, max_tokens, temperature)
                elif provider == "openrouter":
                    if not self._openrouter_key or "your_openrouter" in self._openrouter_key:
                        raise ValueError("OpenRouter API key not configured")
                    text = _call_openrouter(self._openrouter_key, prompt, system, max_tokens, temperature)
                else:
                    if not self._gemini_key or "your_gemini" in self._gemini_key:
                        raise ValueError("Gemini API key not configured")
                    text = _call_gemini(self._gemini_model, prompt, system, max_tokens, temperature)
                
                logger.debug(f"LLM call succeeded via {provider}")
                return {"text": text, "provider": provider}
            except (GroqRateLimitError, httpx.HTTPStatusError) as e:
                # Specific rate limit or status errors - move to next
                status = getattr(e, 'status_code', 'rate_limit')
                logger.warning(f"{provider} error ({status}) — falling back")
                last_error = str(e)
                continue
            except Exception as e:
                logger.warning(f"Provider {provider} failed ({str(e)[:80]}) — trying fallback")
                last_error = str(e)
                continue

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def summarize_for_rag(self, raw_text: str, file_name: str) -> str:
        if len(raw_text) <= LONG_DOC_THRESHOLD_CHARS:
            return raw_text

        logger.info(f"Long document ({len(raw_text)} chars) — running chunked summarization: {file_name}")

        chunks = []
        overlap = 1000
        start = 0
        while start < len(raw_text):
            end = min(start + LONG_DOC_CHUNK_CHARS, len(raw_text))
            chunks.append(raw_text[start:end])
            start = end - overlap

        summaries = []
        for i, chunk in enumerate(chunks):
            # Rotate provider per chunk for load balancing
            providers = ["groq", "openrouter", "gemini"]
            provider = providers[i % 3]
            
            result = self.complete(
                prompt=f"""Extract ALL specific facts, figures, and details from this segment of {file_name}.
Preserve exact numbers, names, and values. Output dense factual content for retrieval.

Document Segment {i + 1}/{len(chunks)}:
---
{chunk}
---""",
                system="You are a factual extractor. Preserve every detail exactly. No filler.",
                max_tokens=2000,
                temperature=0.1,
                force_provider=provider,
            )
            summaries.append(f"[Segment {i + 1}/{len(chunks)} of {file_name}]\n{result['text']}")
            time.sleep(0.5)

        return "\n\n".join(summaries)


llm_router = LLMRouter()
