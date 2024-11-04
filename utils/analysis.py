from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.text_processor import preprocess_text
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import cosine_similarity
from polyglot.detect import Detector
import thulac

# Load THULAC for Chinese segmentation
thu = thulac.thulac(seg_only=True)

# Load Chinese stopwords from file
def load_chinese_stopwords(filepath):
    with open(filepath, encoding='utf-8') as f:
        return set(line.strip() for line in f)

chinese_stopwords = load_chinese_stopwords('stopwords-zh.txt')

def is_chinese(text):
    """
    Detect if text is primarily in Chinese.
    """
    try:
        lang = Detector(text, quiet=True).language.code
        return lang == 'zh'
    except:
        return False

def preprocess_chinese(text):
    """
    Preprocess Chinese text by tokenizing with THULAC and removing stopwords.
    """
    segmented_text = thu.cut(text, text=True)
    words = segmented_text.split()
    words = [word for word in words if word not in chinese_stopwords]
    return ' '.join(words)

def analyze_vocabulary(texts, min_freq=2):
    """
    Analyze vocabulary distribution in a corpus.
    Returns word frequencies and vocabulary statistics.
    """
    # Detect language and set appropriate processing
    if texts and is_chinese(texts[0]):
        texts = [preprocess_chinese(text) for text in texts]
        stop_words = chinese_stopwords
    else:
        texts = [preprocess_text(text) for text in texts]
        stop_words = set(stopwords.words('english'))

    # Tokenize all texts with appropriate vectorizer for language
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=None, min_df=min_freq)
    vectorizer.fit(texts)
    words = vectorizer.get_feature_names_out()
    
    # Count word frequencies
    word_freq = Counter(words)
    
    # Calculate vocabulary statistics
    total_words = len(words)
    unique_words = len(word_freq)
    
    # Create frequency distribution DataFrame
    freq_df = pd.DataFrame(list(word_freq.items()), columns=['word', 'frequency'])
    freq_df['percentage'] = freq_df['frequency'] / total_words * 100
    freq_df = freq_df.sort_values('frequency', ascending=False)
    
    # Calculate cumulative coverage
    freq_df['cumulative_percentage'] = freq_df['percentage'].cumsum()
    
    stats = {
        'total_words': total_words,
        'unique_words': unique_words,
        'words_min_freq': sum(1 for freq in word_freq.values() if freq >= min_freq),
        'coverage_top_1000': freq_df.iloc[:1000]['frequency'].sum() / total_words * 100 if len(freq_df) >= 1000 else 100
    }
    
    return freq_df, stats

def tfidf_analyze_subreddit(posts, max_terms=1000, min_doc_freq=2, include_selftext=False):
    """
    Analyze a single subreddit's posts independently.
    """
    # Combine title and optionally selftext
    texts = [
        (preprocess_chinese(post.get('title', '')) + ' ' + preprocess_chinese(post.get('selftext', ''))
        if include_selftext else preprocess_chinese(post.get('title', '')))
        if is_chinese(post.get('title', '')) else
        (preprocess_text(post.get('title', '')) + ' ' + preprocess_text(post.get('selftext', ''))
        if include_selftext else preprocess_text(post.get('title', '')))
        for post in posts
    ]

    # Analyze vocabulary first
    freq_df, vocab_stats = analyze_vocabulary(texts, min_freq=min_doc_freq)
    
    # Generate TF-IDF matrix and feature names
    tfidf_matrix, feature_names = generate_tfidf_matrix(texts, max_terms, min_doc_freq)
    
    # Create results object from the matrix and feature names
    results = {
        "tfidf_matrix": tfidf_matrix, 
        "feature_names": feature_names, 
        "freq_df": freq_df, 
        "vocab_stats": vocab_stats
    }
    
    return results

def generate_tfidf_matrix(texts, max_terms=1000, min_doc_freq=2):
    """
    Generate TF-IDF matrix and feature names from texts.
    """
    if texts and is_chinese(texts[0]):
        stop_words = chinese_stopwords
    else:
        stop_words = list(set(stopwords.words('english')))

    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_features=max_terms,
        min_df=min_doc_freq
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    return tfidf_matrix, feature_names



