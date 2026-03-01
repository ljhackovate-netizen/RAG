import os
import sys
from core.llm_router import _get_next_provider

def test_rotation():
    print("Testing 3-way rotation:")
    providers = []
    for i in range(12):
        providers.append(_get_next_provider())
    print(f"Rotation sequence: {providers}")
    
    # Expected: ['groq', 'openrouter', 'gemini', 'groq', 'openrouter', 'gemini', ...]
    # Note: _request_counter might not be 0 if imported in a live app, but sequence should be consistent.
    for i in range(0, 9, 3):
        chunk = providers[i:i+3]
        if len(set(chunk)) != 3:
            print(f"FAIL at index {i}: {chunk}")
            return
    print("PASS: Rotation is consistent across 3 providers.")

if __name__ == "__main__":
    test_rotation()
