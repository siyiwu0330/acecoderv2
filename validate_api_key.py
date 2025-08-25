#!/usr/bin/env python3
"""
Validate OpenAI API key and check remaining quota
"""
import openai
import sys
from openai import OpenAI

def check_api_key(api_key: str):
    """Check if API key is valid and has remaining quota"""
    
    if not api_key or api_key == "your-api-key-here":
        print("❌ No valid API key provided")
        return False
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Try a minimal request to test the key
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1
        )
        
        print("✅ API key is valid and has quota")
        return True
        
    except openai.RateLimitError as e:
        print(f"❌ API quota exceeded: {e}")
        print("💡 Please check your billing and usage at: https://platform.openai.com/usage")
        return False
        
    except openai.AuthenticationError as e:
        print(f"❌ Invalid API key: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Error testing API key: {e}")
        return False

def get_usage_info(api_key: str):
    """Try to get usage information (this might not work with all API keys)"""
    try:
        # Note: Usage endpoint might not be available for all users
        print("💡 For detailed usage information, visit: https://platform.openai.com/usage")
        
    except Exception as e:
        print(f"ℹ️  Cannot retrieve usage info automatically: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = input("Enter your OpenAI API key: ").strip()
    
    print("🔍 Validating OpenAI API key...")
    
    if check_api_key(api_key):
        get_usage_info(api_key)
    else:
        print("\n📋 What to do:")
        print("1. Check your billing at: https://platform.openai.com/account/billing")
        print("2. Add credits to your account if needed")
        print("3. Wait if you hit rate limits (they reset over time)")
        print("4. Consider using a different model (gpt-3.5-turbo is cheaper)")