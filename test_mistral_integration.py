#!/usr/bin/env python3
"""
Test script to verify Mistral API integration with .env configuration.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import from the app
from app.core.config import config  # noqa: E402
from app.llm.mistral_client import MistralClient  # noqa: E402


def test_env_loading():
    """Test that environment variables are loaded correctly."""
    print("Testing environment variable loading...")

    print(f"MISTRAL_API_KEY exists: {'Yes' if config.MISTRAL_API_KEY else 'No'}")
    if config.MISTRAL_API_KEY:
        print(f"MISTRAL_API_KEY length: {len(config.MISTRAL_API_KEY)} characters")
        print(f"MISTRAL_API_KEY starts with: {config.MISTRAL_API_KEY[:10]}...")

    print(f"MISTRAL_EMBED_MODEL: {config.MISTRAL_EMBED_MODEL}")
    print(f"MISTRAL_CHAT_MODEL: {config.MISTRAL_CHAT_MODEL}")
    print()


def test_mistral_embedding():
    """Test Mistral embedding API call."""
    print("Testing Mistral embedding API...")

    try:
        if not config.MISTRAL_API_KEY:
            print("❌ No Mistral API key found in environment")
            return False

        client = MistralClient()

        # Test with a simple text
        test_text = "This is a test document for embedding."
        print(f"Testing embedding for: '{test_text}'")

        embedding = client.get_single_embedding(test_text)

        if embedding is not None and len(embedding) > 0:
            print(f"✅ Embedding successful! Dimension: {len(embedding)}")
            print(f"First 5 values: {embedding[:5]}")
            return True
        print("❌ Embedding returned None or empty")
        return False

    except Exception as e:
        print(f"❌ Embedding failed with error: {e}")
        return False


def test_mistral_chat():
    """Test Mistral chat API call."""
    print("Testing Mistral chat API...")

    try:
        if not config.MISTRAL_API_KEY:
            print("❌ No Mistral API key found in environment")
            return False

        client = MistralClient()

        # Test with a simple query
        test_query = "What is the capital of France?"
        print(f"Testing chat with query: '{test_query}'")

        messages = [{"role": "user", "content": test_query}]
        response = client.chat_completion(messages)

        if response:
            print(f"✅ Chat successful! Response length: {len(response)} characters")
            print(f"Response preview: {response[:100]}...")
            return True
        print("❌ Chat returned empty response")
        return False

    except Exception as e:
        print(f"❌ Chat failed with error: {e}")
        return False


def main():
    """Run all tests."""
    print("=== Mistral API Integration Test ===\n")

    # Test environment loading
    test_env_loading()

    # Test embedding
    embedding_success = test_mistral_embedding()
    print()

    # Test chat
    chat_success = test_mistral_chat()
    print()

    # Summary
    print("=== Test Summary ===")
    print(f"Embedding API: {'✅ PASS' if embedding_success else '❌ FAIL'}")
    print(f"Chat API: {'✅ PASS' if chat_success else '❌ FAIL'}")

    if embedding_success and chat_success:
        print("\n🎉 All tests passed! Mistral API integration is working correctly.")
        return 0
    print("\n❌ Some tests failed. Check your Mistral API key and network connection.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
