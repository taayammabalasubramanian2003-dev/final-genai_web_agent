import os
from openai import OpenAI, AuthenticationError, RateLimitError

# --- 1. SETUP YOUR KEY ---
# Replace this string with your actual sk-... key if not using env variables
# or set it in your terminal: export OPENAI_API_KEY="sk-..."
api_key = "sk-YOUR_ACTUAL_KEY_HERE" 

# Better practice: Try to get from environment
if os.getenv("OPENAI_API_KEY"):
    api_key = os.getenv("OPENAI_API_KEY")

print("--- 🕵️ OpenAI Model Access Checker ---\n")

if api_key == "sk-YOUR_ACTUAL_KEY_HERE":
    print("❌ Error: You didn't set your API Key in the code.")
    exit()

try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    print(f"❌ Initialization Error: {e}")
    exit()

# --- 2. DEFINE MODELS TO TEST ---
# These are the most common model names people need.
# We test these specifically because listing ALL models returns 100+ junk items (like babbage, davinci, etc.)
candidates = [
    "gpt-4o",              # Newest, fastest Omni model
    "gpt-4o-mini",         # Cheaper, fast Omni model
    "gpt-4-turbo",         # Previous flagship
    "gpt-4",               # Standard GPT-4
    "gpt-3.5-turbo",       # Standard fast model
    "gpt-3.5-turbo-16k",   # Larger context (legacy)
]

print(f"🔍 Testing {len(candidates)} common models with your key...\n")

working_models = []

for model_name in candidates:
    print(f"Testing {model_name}...", end=" ")
    try:
        # We try a tiny request to see if it allows access
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5  # Keep it cheap
        )
        print("✅ WORKING")
        working_models.append(model_name)
        
    except AuthenticationError:
        print("❌ FAILED (Invalid API Key)")
        break # No point testing others
    except RateLimitError:
        print("⚠️ FAILED (Quota Exceeded / Billing setup required)")
    except Exception as e:
        # Often "model_not_found" or "404"
        if "does not exist" in str(e):
            print("🚫 ACCESS DENIED (Not available on your plan)")
        else:
            print(f"❌ ERROR: {e}")

# --- 3. SUMMARY ---
print("\n" + "="*30)
if working_models:
    print("🚀 SUCCESS! Use one of these in your code:")
    for m in working_models:
        print(f'model="{m}"')
else:
    print("⚠️ No working models found. Check your OpenAI billing/credits.")
