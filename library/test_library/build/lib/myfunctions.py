from transformers import AutoTokenizer
from transformers import pipeline
import nltk


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
    """
    Lemmatize a list of words.

    This function takes a list of words, determines the part of speech for each word, 
    and then lemmatizes it (converts it to its base or dictionary form) according 
    to its part of speech. It utilizes the NLTK library's WordNetLemmatizer 
    and the part-of-speech tagging to accurately lemmatize each word.

    Parameters:
    - words: A list of words (strings) that you want to lemmatize.

    Returns:
    - A list of lemmatized words.

    Note: This function requires nltk's WordNetLemmatizer and pos_tag to be imported, 
    along with the wordnet corpus and a function get_wordnet_pos(tag) that converts 
    the part-of-speech tagging conventions between nltk and wordnet.
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    
    # Get POS tag for each word
    pos_tagged = pos_tag(words)
    
    for word, tag in pos_tagged:
        wordnet_pos = get_wordnet_pos(tag) or wordnet.NOUN
        lemmatized_word = lemmatizer.lemmatize(word, pos=wordnet_pos)
        lemmatized_words.append(lemmatized_word)
        
    return lemmatized_words

from nltk.sentiment.vader import SentimentIntensityAnalyzer


def analyze_sentiment_vader(text):
    nltk.download('vader_lexicon')
    """
    Analyzes the sentiment of a given text using VADER sentiment analysis.

    Parameters:
    - text: A string containing the text to analyze.

    Returns:
    - A dictionary containing the scores for negative, neutral, positive, and compound sentiments.
    """
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores




import nltk
from nltk.stem.porter import PorterStemmer


def stem_words(words):
    """
    Stems a list of words.

    This function applies the Porter Stemming algorithm to a list of words, 
    reducing each word to its root or stem form. It's particularly useful in 
    natural language processing and search applications where the exact form of 
    a word is less important than its root meaning.

    Parameters:
    - words: A list of words (strings) to be stemmed.

    Returns:
    - A list containing the stemmed version of each input word.

    Example:
    >>> stem_words(["running", "jumps", "easily"])
    ['run', 'jump', 'easili']
    
    Note: This function requires the nltk's PorterStemmer to be imported.
    """
    # Initialize the Porter Stemmer
    stemmer = PorterStemmer()
    
    # Stem each word in the list
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words



