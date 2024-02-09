from transformers import AutoTokenizer
from transformers import pipeline
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

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

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

# Helper function to convert NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_words(words):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    
    # Get POS tag for each word
    pos_tagged = pos_tag(words)
    
    for word, tag in pos_tagged:
        wordnet_pos = get_wordnet_pos(tag) or wordnet.NOUN
        lemmatized_word = lemmatizer.lemmatize(word, pos=wordnet_pos)
        lemmatized_words.append(lemmatized_word)
        
    return lemmatized_words




import nltk
from nltk.stem.porter import PorterStemmer

# Download the necessary NLTK data (if not already done)
nltk.download('punkt')

def stem_words(words):
    # Initialize the Porter Stemmer
    stemmer = PorterStemmer()
    
    # Stem each word in the list
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words


