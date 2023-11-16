from transformers import pipeline

# Load the sentiment analysis pipeline
# This will download and cache the model the first time you run it
sentiment_pipeline = pipeline("sentiment-analysis")

# Example text
text = "I love using transformer models for natural language processing."

# Perform sentiment analysis
result = sentiment_pipeline(text)

print(result)
