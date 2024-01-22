from transformers import AutoTokenizer
from transformers import pipeline
def tokenize_text(text, model_name="bert-base-uncased"):
    """
    Tokenize a given text using the Hugging Face Transformers library.

    Parameters:
    - text (str): The input text to tokenize.
    - model_name (str): The name of the pre-trained model to use for tokenization.
                       Default is "bert-base-uncased".

    Returns:
    - tokens (list): List of tokens obtained by tokenizing the input text.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokens = tokenizer.tokenize(text)

    return tokens

def analyze_sentiment(text, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
    """
    Analyze sentiment of a given text using a pre-trained sentiment analysis model.

    Parameters:
    - text (str): The input text for sentiment analysis.
    - model_name (str): The name of the pre-trained sentiment analysis model.
                       Default is "nlptown/bert-base-multilingual-uncased-sentiment".

    Returns:
    - sentiment (str): The predicted sentiment (e.g., "POSITIVE", "NEGATIVE", "NEUTRAL").
    - confidence (float): The confidence score associated with the predicted sentiment.
    """

    sentiment_analyzer = pipeline('sentiment-analysis', model=model_name)


    result = sentiment_analyzer(text)


    sentiment = result[0]['label']
    confidence = result[0]['score']

    return sentiment, confidence

