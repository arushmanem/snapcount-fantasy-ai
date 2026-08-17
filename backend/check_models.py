import google.generativeai as genai
import os

# 1. Paste your API Key here directly to test
os.environ["GOOGLE_API_KEY"] = "PASTE_YOUR_KEY_HERE"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("--- Checking Available Models ---")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Found: {m.name}")
except Exception as e:
    print(f"❌ Error: {e}")