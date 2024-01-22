from test_library.myfunctions import analyze_sentiment
input_text = "I love using Hugging Face Transformers library!"
predicted_sentiment, confidence = analyze_sentiment(input_text)
print(f"Sentiment: {predicted_sentiment}, Confidence: {confidence}")
