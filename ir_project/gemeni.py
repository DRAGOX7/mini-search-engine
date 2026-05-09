from google import genai

# Initialize the client. 
# It will automatically pick up your GEMINI_API_KEY environment variable.
client = genai.Client(api_key="AIzaSyB4K5UWKQseNWT9jHMhXUf5xoZvhxiy1Zg")

print("Models available for content generation:")
for model in client.models.list():
    # Filter specifically for models that support text/content generation
    if "generateContent" in model.supported_actions:
        print(f"- {model.name}")