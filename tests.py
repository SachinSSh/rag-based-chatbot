import google.generativeai as genai
import os

# Enter your API key here or set GEMINI_API_KEY environment variable
API_KEY = "AIzaSyCiZYjUbg4rVNM2P35KI6Lo50iZzZ75YoA" # or os.getenv('GEMINI_API_KEY')

genai.configure(api_key=API_KEY)

print("Available Gemini Models:")
print("-" * 60)

free_models = []

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        model_name = model.name.replace('models/', '')
        free_models.append(model_name)
        print(f"✅ {model_name}")
        print(f"   Display Name: {model.display_name}")
        print(f"   Input Limit: {model.input_token_limit}")
        print(f"   Output Limit: {model.output_token_limit}")
        print()

print("-" * 60)
print(f"\nTotal usable models: {len(free_models)}")

if free_models:
    print(f"\n💡 Recommended for RAG: {free_models[0]}")