# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:48:27 2024

@author: venus
"""

import os
import gzip
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
from nltk.stem import WordNetLemmatizer
import langid
import warnings
import contractions
import fasttext
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')
import ahocorasick
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.svm import SVC
import contractions
import emoji
from transformers import pipeline
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sentence_transformers import SentenceTransformer, util
import spacy
from collections import Counter
import torch


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_word_negative = {
'no',
'not',
'none',
'neither',
'never',
'nobody',
'nothing',
'nowhere',
"doesn't",
"isn't",
"wasn't",
"shouldn't",
"won't",
"can't",
"couldn't",
"don't",
"haven't",
"hasn't",
"hadn't",
"aren't",
"weren't",
"wouldn't",
"daren't",
"needn't",
"didn't",
"without",
"against",
'negative',
'deny',
'reject',
'refuse',
'decline',
'unhappy',
'sad',
'miserable',
'hopeless',
'worthless',
'useless',
'futile',
'disagree',
'oppose',
'contrary',
'contradict',
'disapprove',
'dissatisfied',
'objection',
'unsatisfactory',
'unpleasant',
'regret',
'resent',
'lament',
'mourn',
'grieve',
'bemoan',
'despise',
'loathe',
'detract',
'abhor',
'dread',
'fear',
'worry',
'anxiety',
'sorrow',
'gloom',
'melancholy',
'dismay',
'disheartened',
'despair',
'dislike',
'aversion',
'antipathy',
'hate',
'disdain'}

custom_stopwords = stop_words - stop_word_negative
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# C:\Users\Ale\CENTENNIAL\FALL_2024\COMP262_NLP\NLP-Project
# os.chdir('C:/Users/venus/Documents/03 Study/01 AI/Fall2024/NLP/Project')
os.chdir('C:/Users/Ale/CENTENNIAL/FALL_2024/COMP262_NLP/NLP-Project')
df = pd.read_json('AMAZON_FASHION.json', lines=True)
df.head()
df.info()
print(f'Total reviews for full datset: {df.shape[0]}')

#convert time
df['reviewTime'] = pd.to_datetime(df['reviewTime'], format='%m %d, %Y')
df['reviewTime'] = df['reviewTime'].dt.strftime('%Y-%m-%d')

#exclude 1000 from 800k
df_test = pd.read_csv('test_part2.csv')
df_test.info()
df_test.head()
print(f'Total review of test dataset: {df_test.shape[0]}')

columns_to_compare =['reviewTime', 'asin', 'reviewerID', 'reviewText', 'overall']
df_filtered = df.merge(df_test[columns_to_compare], on=columns_to_compare, how='left',indicator=True)

#keep only rows not in df_test
df = df_filtered[df_filtered['_merge'] == 'left_only']

#drop the _merge
df = df.drop(columns = ['_merge'])
print(f'Total review after excluding test dataset: {df.shape[0]}')
#---------------------------------------Step 11---------------------------------------
# a. Select a subset of the original data minimum 2000 reviews
# convert rating to positive, neutral and negative
df['sentiment'] = df['overall'].apply(lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral'))
rating_distribution = df['sentiment'].value_counts().sort_index()

plt.figure(figsize=(8, 6))
colors = ['skyblue', 'coral', 'lightpink']
rating_distribution.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

# sampling for 2000 reviews for each sentiment
# Process each group separately and concatenate
df_subset = pd.concat([group.sample(2000, random_state=42) for _, group in df.groupby('sentiment')]).reset_index(drop=True)

# b. Carry out data exploration on the subset and pre-processing and justify each step of preprocessing

# b1. number of reviews per rating
rating_distribution_sample = df_subset['sentiment'].value_counts().sort_index()

plt.figure(figsize=(8, 6))
colors = ['skyblue', 'coral', 'lightpink']
rating_distribution_sample.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

# b2 total review
total_reviews = len(df_subset)
unique_products = df_subset['asin'].nunique()
unique_users = df_subset['reviewerID'].nunique()
verified_purchases = df_subset['verified'].sum()
average_rating = df_subset['overall'].mean()

print(df_subset.info())
print("Total Reviews:", total_reviews)
print("Unique Products:", unique_products)
print("Unique Users:", unique_users)
print("Verified Purchases:", verified_purchases)
print("Average Rating:", average_rating)

# b3 top 20 products
min_reviews = 1

reviews_per_product = df_subset['asin'].value_counts()
filtered_reviews_per_product = reviews_per_product[reviews_per_product >= min_reviews]

top_n = 20
top_reviews_per_product = filtered_reviews_per_product.head(top_n)

plt.figure(figsize=(10, 6))
top_reviews_per_product.plot(kind='bar', color='skyblue', edgecolor='black')

plt.title(f'Top {top_n} Products with at least {min_reviews} Reviews')
plt.xlabel('Product (ASIN)')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# b4. outliers
df_subset['review_length'] = df_subset['reviewText'].apply(lambda x: len(str(x).split()))
outliers = df_subset[(df_subset['review_length'] > df_subset['review_length'].quantile(0.99))]

print("Number of Outliers:", len(outliers))

print(outliers['review_length'].head())

max_word_count = df_subset['reviewText'].apply(lambda x: len(str(x).split())).max()
print(f'Maximum review word count: {max_word_count}')

min_word_count = df_subset['reviewText'].apply(lambda x: len(str(x).split())).min()
print(f'Minimum review word count: {min_word_count}')

first_percentile_value = df_subset['review_length'].quantile(0.01)
print(f'1st percential word count: {first_percentile_value}')

max_percentile_value = df_subset['review_length'].quantile(0.99)
print(f'99th percential word count: {max_percentile_value}')

avg_word_count = df_subset['review_length'].mean()
print(f'Average word count: {avg_word_count}')

#review len
plt.figure(figsize=(10,6))
df_subset['review_length'].plot(kind='hist', bins=30,edgecolor='black', color='skyblue')
plt.title('Distribution of Review Word Counts')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()

# b5. duplicate
duplicate_reviews = df_subset.duplicated(subset=['reviewTime', 'asin', 'reviewerID', 'reviewText']).sum()

print("Duplicate Reviews:", duplicate_reviews)

# b6 HTML check
def contains_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return len(soup.find_all()) > 0

html_tagged_reviews = df_subset['reviewText'].apply(lambda x: contains_html(str(x)))

html_tag_count = html_tagged_reviews.sum()

print(f'Reviews with HTML tags: {html_tag_count}')
print(df_subset[html_tagged_reviews == True]['reviewText'].head())

# b7 URL check
def contains_url(text):
    url_pattern = r'http[s]?://'
    return re.search(url_pattern, text) is not None

utl_tagged_reviews = df_subset['reviewText'].apply(lambda x: contains_url(str(x)))

url_tag_count = utl_tagged_reviews.sum()
print(f'Reviews with URLs: {url_tag_count}')
print(df_subset[utl_tagged_reviews == True]['reviewText'].head())

# b8. null check
null_reviews = df_subset['reviewText'].isnull().sum()
empty_reviews = (df_subset['reviewText'].str.strip() == '').sum()

print(f'Null reviews: {null_reviews}')
print(f'Empty reviews: {empty_reviews}')

print(df_subset[df_subset['reviewText'].isnull()][['reviewText']].head())
print(df_subset[df_subset['reviewText'].str.strip() == ''][['reviewText']].head())

# b9. email check
def contains_email(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return bool(re.search(email_pattern, text))

email_mentions = df_subset['reviewText'].apply(lambda x: contains_email(str(x)) if isinstance(x, str) else False)

email_count = email_mentions.sum()

print(f'Reviews with email addresses: {email_count}')
print(df_subset[email_mentions == True]['reviewText'])

# b10. user mention
def contains_user_mentions(text):
    return bool(re.search(r'@\w+', text))

user_mentions = df_subset['reviewText'].apply(lambda x: contains_user_mentions(str(x)) if isinstance(x, str) else False)

mention_count = user_mentions.sum()

print(f'Reviews with user mentions: {mention_count}')
print(df_subset[user_mentions == True]['reviewText'])

# b11. verified vs non-verified
plt.figure(figsize=(8, 6))
sns.countplot(x='verified', data=df_subset, palette='Set2')
plt.title('Verified vs Non-Verified Reviews')
plt.xlabel('Verified Purchase')
plt.ylabel('Count')
plt.show()

# b12. review time
df_subset['reviewTime'] = pd.to_datetime(df_subset['reviewTime'], errors='coerce')
df_subset['year'] = df_subset['reviewTime'].dt.year
plt.figure(figsize=(10, 6))
sns.countplot(x='year', data=df_subset, palette='Set2')
plt.title('Reviews Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# b13. check non-english
model = fasttext.load_model('lid.176.bin')

def detect_language_fasttext(text):
    try:
        predictions = model.predict(text, k=1)
        lang = predictions[0][0].replace('__label__', '')
        return lang
    except:
        return 'unknown'

df_subset['language'] = df_subset['reviewText'].apply(detect_language_fasttext)

non_english_reviews = df_subset[df_subset['language'] != 'en']

print(f'Non-English reviews: {non_english_reviews.shape[0]}')

if not non_english_reviews.empty:
    print("Example non-English reviews:")
    print(non_english_reviews[['reviewText', 'language']].head(50))
else:
    print("All reviews are in English.")
    
language_counts = df_subset['language'].value_counts()

print(language_counts)

plt.figure(figsize=(10, 6))
sns.barplot(x=language_counts.index, y=language_counts.values, palette='Set2')

plt.title('Distribution of Languages in Reviews', fontsize=15)
plt.xlabel('Language', fontsize=12)
plt.ylabel('Number of Reviews', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

special_character = df_subset['reviewText'].str.contains(r'[^\w\s]',regex=True)
print(f'Reviews with special characters: {special_character.sum()}')

#check emoji
emoji_pattern = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
    "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed Characters
    "]+", 
    flags=re.UNICODE
)

def contains_emoji(text):
    return bool(emoji_pattern.search(text))
df_subset['contains_emoji'] = df_subset['reviewText'].apply(lambda x: contains_emoji(str(x)))

emoji_count = df_subset['contains_emoji'].sum()
print(f"Number of reviews with emojis: {emoji_count}")


# ------------ Pre-processing ------------------
def common_preprocessing(df_original):
    
    print(f'Dataframe before preprocessing: {df_original.shape[0]}')
    #4a. Remove outliers (words more than 99 percentile)
    df_clean = df_original[(df_original['review_length'] <= df_original['review_length'].quantile(0.99))]
    print(f"Dataframe size after removing outliers: {df_clean.shape[0]}")

    # remove null and empty review
    
    df_clean = df_clean.dropna(subset=['reviewText'])

    df_clean = df_clean[df_clean['reviewText'].str.strip() != '']

    print(f'Remaining reviews after removing null and empty reviews: {df_clean.shape[0]}')

    
    
    #remove non-english
    df_clean = df_clean[df_clean['language'] == 'en']
    print(f'Remaining reviews after removing non-English reviews: {df_clean.shape[0]}')
    
    
    #remove mention form the text
    def remove_mentions(text):
        return re.sub(r'@\w+', '', text)
    
    df_clean['clean_text'] = df_clean['reviewText'].apply(remove_mentions)
    
    
    #expand contractions in the text
    df_clean['clean_text'] = df_clean['clean_text'].apply(contractions.fix)  


    # remove stopword
    def remove_stopwords(text):
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in custom_stopwords]
        return ' '.join(words)
    
    df_clean['clean_text'] = df_clean['clean_text'].apply(remove_stopwords)
    
    # remove special character
    df_clean['clean_text'] = df_clean['clean_text'].str.replace(r'[^\w\s]','',regex=True)
    
    df_clean['clean_text'] = df_clean['clean_text'].str.lower()
    
    return df_clean

df_preprocessed = common_preprocessing(df_subset)
df_preprocessed.head()

df_preprocessed['review_length'] = df_preprocessed['clean_text'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10,6))
df_preprocessed['review_length'].plot(kind='hist', bins=30,edgecolor='black', color='skyblue')
plt.title('Distribution of Review Word Counts (After Preprocessing)')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()

avg_word_count_preprocessed = df_preprocessed['review_length'].mean()
print(f'Average Word Count (After Preprocessing): {avg_word_count_preprocessed}')

# c. Represent your text 
# use tf-idf
# d. Split the data into 70% for training and 30% for testing,—Use stratified splitting
X_train, X_test, y_train, y_test = train_test_split(df_preprocessed['clean_text'], df_preprocessed['sentiment'], test_size=0.3, stratify=df_preprocessed['sentiment'], random_state=42)

#check splitting
plt.figure(figsize=(10,5))
sns.countplot(x=y_train, palette="Set2")
plt.title('Training Set Distribution')
plt.xlabel('Sentiment Type')
plt.ylabel('Number of Reviews')
plt.show()
training_count = y_train.value_counts()
print(f'Training Sentiment Counts: {training_count}')

plt.figure(figsize=(10,5))
sns.countplot(x=y_test, palette="Set2")
plt.title('Testing Set Distribution')
plt.xlabel('Sentiment Type')
plt.ylabel('Number of Reviews')
plt.show()
testing_count = y_test.value_counts()
print(f'Testing Sentiment Counts: {testing_count}')

#check unique word
all_reviews_text = ' '.join(df_preprocessed['clean_text'].dropna().astype(str).tolist())
all_words = word_tokenize(all_reviews_text.lower())
word_counts = Counter(all_words)
unique_words_count = len(word_counts)
duplicate_words = {word: count for word, count in word_counts.items() if count > 1}

print(f"Total number of words in all reviews: {len(all_words)}")
print(f"Total number of unique words: {unique_words_count}")
print(f"Number of duplicate words: {len(duplicate_words)}")
print(f"Some duplicate words: {list(duplicate_words.items())[:10]}")

# apply tf-idf
tfidf_vectorizer = TfidfVectorizer(
    max_features = 5000,
    ngram_range=(1,2))

#fit and transform
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

#check tf idf
print(f'Training TF-IDF shape: {X_train_tfidf.shape}')
print(f'Testing TF-IDF shape: {X_test_tfidf.shape}')

print(f'Top 10 features: {tfidf_vectorizer.get_feature_names_out()[:10]}')

# e. Build two model using 70% of the data
# logistic regression
#define param
param_grid_lr = {
    'C': [0.01, 0.1, 1],
    'penalty':['l1', 'l2'],
    'solver':['liblinear'],
    'max_iter':[1000,2000,5000]}

lr_model = LogisticRegression(max_iter=1000, random_state=42)

#train the model
grid_search_lr = GridSearchCV(
    estimator=lr_model,
    param_grid=param_grid_lr,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1)

grid_search_lr.fit(X_train_tfidf, y_train)

print(f'Best Params: {grid_search_lr.best_params_}')

best_lr_model = grid_search_lr.best_estimator_
best_lr_model.fit(X_train_tfidf, y_train)
y_train_pred_lr = best_lr_model.predict(X_train_tfidf)

train_accuracy_lr = accuracy_score(y_train, y_train_pred_lr)
train_f1_lr = f1_score(y_train, y_train_pred_lr, average='weighted')
train_recall_lr = recall_score(y_train, y_train_pred_lr, average='weighted')
train_precision_lr = precision_score(y_train, y_train_pred_lr, average='weighted')

print('Training Results for Logistic Regression')
print(f'Training Accuracy: {train_accuracy_lr}')
print(f'Training F1: {train_f1_lr}')
print(f'Training Precision: {train_precision_lr}')
print(f'Training Recall: {train_recall_lr}')

joblib.dump(best_lr_model, 'best_lr_model.pkl')

#SVM
param_grid_svm = {
    'C':[0.01, 0.1, 1, 10],
    'kernel':['linear', 'rbf'],
    'gamma':[0.01, 0.1, 1, 10]}

svm = SVC(random_state=42)

grid_search_svm = GridSearchCV(
    estimator=svm,
    param_grid=param_grid_svm,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1)

grid_search_svm.fit(X_train_tfidf, y_train)

print(f'Best param SVM: {grid_search_svm.best_params_}')
best_svm_model = grid_search_svm.best_estimator_
joblib.dump(best_svm_model, 'best_svm_model.pkl')

y_train_pred_svm = best_svm_model.predict(X_train_tfidf)

train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)
train_f1_svm = f1_score(y_train, y_train_pred_svm, average='weighted')
train_recall_svm = recall_score(y_train, y_train_pred_svm, average='weighted')
train_precision_svm = precision_score(y_train, y_train_pred_svm, average='weighted')

print('Training Results for SVM')
print(f'Training Accuracy: {train_accuracy_svm}')
print(f'Training F1: {train_f1_svm}')
print(f'Training Precision: {train_precision_svm}')
print(f'Training Recall: {train_recall_svm}')

# ---------------------------------- Step 13 -------------------------------
# Test out the two model
def test_model(model, X_test_tfidf, y_test):
    y_test_pred = model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_test, y_test_pred, average='weighted')
    train_recall = recall_score(y_test, y_test_pred, average='weighted')
    train_precision = precision_score(y_test, y_test_pred, average='weighted')
    print(f'Classification Report')
    print(classification_report(y_test, y_test_pred))
    cm = confusion_matrix(y_test, y_test_pred)
    print(f'Testing Accuracy: {test_accuracy}')
    print(f'Testing F1: {train_f1}')
    print(f'Testing Precision: {train_precision}')
    print(f'Testing Recall: {train_recall}')
    print(cm)

    
print('Logistic Regression')
test_model(best_lr_model, X_test_tfidf, y_test)

print('SVM')
test_model(best_svm_model, X_test_tfidf, y_test)

# ---------------------------------- Step 14 -------------------------------
# use results from phase 1 after selected to run the model
# group sentiment and preprocess
# this is the dataset that was used for lexicon
df_test = pd.read_csv('test_part2.csv')
df_test.info()

#preprocessing
df_test_preprocessed = common_preprocessing(df_test)

df_test_preprocessed.info()

X_test_ml = df_test_preprocessed['clean_text'].str.lower()
y_test_ml = df_test_preprocessed['sentiment'].str.lower()


X_test_ml_tfidf = tfidf_vectorizer.transform(X_test_ml)



print('Test with Lexicon Dataset')
print('Logistic Regression')
test_model(best_lr_model, X_test_ml_tfidf, y_test_ml)

print('SVM')
test_model(best_svm_model, X_test_ml_tfidf, y_test_ml)


# ---------------------------------- Step 15 -------------------------------
# ============================ Enhance using emotion ========================
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
emotion_classifier("I love this!")

def get_emotion(text):
    emotions = emotion_classifier(text)[0]
    highest_emotion = max(emotions, key=lambda x: x['score'])['label']
    return highest_emotion

df_preprocessed['emotion'] = df_preprocessed['clean_text'].apply(get_emotion)

# weights for each label
emotion_weights = {
    'anger': -0.25,
    'disgust': -0.25,
    'fear': -0.1,
    'joy': 0.25,
    'neutral': 0,
    'sadness': -0.1,
    'surprise': 0
    }

def get_enhanced_rating_emotion(row):
    original_rating = row['overall']
    emotion = row['emotion']
    
    enhanced_rating = original_rating + emotion_weights[emotion]
    
    return enhanced_rating

df_preprocessed['enhanced_rating_emotion'] = df_preprocessed.apply(get_enhanced_rating_emotion, axis=1)

# ===========================================================================

# ============================ Enhance using context ========================
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # For embeddings
nlp = spacy.load("en_core_web_sm")  # For dependency parsing

# Step 1: Extract Feature-Modifier Pairs
def extract_context_mapping(review):
    """
    Extract feature-modifier pairs dynamically from review text.
    """
    doc = nlp(review)
    context_pairs = []
    for token in doc:
        # Look for nouns and adjectives as features
        if token.pos_ in {"NOUN", "ADJ"}:
            # Find modifiers (adverbs, prepositions, objects) linked to the feature
            for child in token.children:
                if child.dep_ in {"advmod", "prep", "pobj", "amod"}:
                    context_pairs.append((token.text, child.text))
    return context_pairs

# Apply the context mapping to all reviews
print("Extracting context mappings...")
df_preprocessed['context_mappings'] = df_preprocessed['clean_text'].apply(extract_context_mapping)

# Step 2: Encode Reviews and Contexts with Sentence Embeddings
def encode_text(text):
    """
    Encode text into a dense vector using SentenceTransformer.
    """
    return sentence_model.encode(text)

# Encode the full reviews
print("Encoding review embeddings...")
df_preprocessed['review_embedding'] = df_preprocessed['clean_text'].apply(encode_text)

# Encode extracted context mappings as a single string
df_preprocessed['context_string'] = df_preprocessed['context_mappings'].apply(
    lambda x: " ".join([f"{feature} {modifier}" for feature, modifier in x])
)
print("Encoding context embeddings...")
df_preprocessed['context_embedding'] = df_preprocessed['context_string'].apply(encode_text)

# Step 3: Compute Contextual Diversity
def calculate_context_diversity(contexts):
    """
    Calculate diversity score based on pairwise cosine similarity.
    """
    embeddings = np.array([sentence_model.encode(context) for context in contexts])
    if embeddings.shape[0] > 1:
        diversity = 1 - pairwise_distances(embeddings, metric="cosine").mean()
        return diversity
    else:
        return 0.0

print("Calculating contextual diversity...")
df_preprocessed['diversity_score'] = df_preprocessed['context_mappings'].apply(
    lambda x: calculate_context_diversity([f"{feature} {modifier}" for feature, modifier in x])
)

# Step 4: Enhance Ratings Based on Contextual Diversity and Sentiment
alpha = 0.7  # Weight for original rating
beta = 0.3   # Weight for diversity score

# Compute enhanced rating
df_preprocessed['enhanced_rating_context'] = (
    alpha * df_preprocessed['overall'] + beta * df_preprocessed['diversity_score'] * 5
)

# ============================================================================

# ============================ Combining enhancements ========================
def get_enhanced_rating_combined(row):
    enhanced_rating_emotion = row['enhanced_rating_emotion']
    enhanced_rating_context = row['enhanced_rating_context']
    
    enhanced_rating_combined = (enhanced_rating_emotion + enhanced_rating_context) / 2
    
    return enhanced_rating_combined

df_preprocessed['enhanced_rating_combined'] = df_preprocessed.apply(get_enhanced_rating_combined, axis=1)

# ============================================================================

# ========================= Final steps for enhancements =====================
df_enhanced_columns = [
    'clean_text',
    'overall',
    'enhanced_rating_emotion',
    'enhanced_rating_context',
    'enhanced_rating_combined',
    'emotion',
    'context_mappings',
    'review_embedding', 
    'context_string', 
    'context_embedding',
    'diversity_score'
    ]

df_enhanced = df_preprocessed[df_enhanced_columns]

# Rank and Save Results
# Sort reviews by enhanced rating
df_enhanced = df_enhanced.sort_values(by='enhanced_rating_combined', ascending=False)

# Save the enhanced recommendations to a file
output_file = "enhanced_ratings.csv"
df_enhanced.to_csv(output_file, index=False)
print(f"Enhanced ratings saved to '{output_file}'.")

# Visualize enhanced ratings
plt.figure(figsize=(10, 6))

# Plot each enhanced rating
plt.scatter(df_enhanced['overall'], df_enhanced['enhanced_rating_emotion'], 
            alpha=0.5, label="Enhanced Rating (Emotion)", color='blue')
plt.scatter(df_enhanced['overall'], df_enhanced['enhanced_rating_context'], 
            alpha=0.5, label="Enhanced Rating (Context)", color='green')
plt.scatter(df_enhanced['overall'], df_enhanced['enhanced_rating_combined'], 
            alpha=0.5, label="Enhanced Rating (Combined)", color='orange')

# Plot the ideal line
plt.plot([1, 5], [1, 5], color='red', linestyle='--', label="Baseline (No Change)")

# Add titles and labels
plt.title("Original vs Enhanced Ratings")
plt.xlabel("Original Ratings")
plt.ylabel("Enhanced Ratings")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------- Step 16 --------------------------------------
# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Select long reviews
long_reviews = df_preprocessed[df_preprocessed['review_length'] > 100].head(10)

# Generate summaries
long_reviews['summary'] = long_reviews['reviewText'].apply(
    lambda x: summarizer(x, max_length=50, min_length=30, do_sample=False)[0]['summary_text']
)

# Take the original and summarized text
df_summarized = long_reviews[['reviewText', 'summary']]

# Record the first two summaries
print("Summaries for Long Reviews:")
for _, row in df_summarized.head(2).iterrows():
    print(f"reviewText:\n{row['reviewText']}\n")
    print(f"summary:\n{row['summary']}\n")
    
# ----------------------------- Step 17 --------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
# Review with question nature
question_review = df_preprocessed[df_preprocessed['reviewText'].str.contains(r'\?')].iloc[0][['reviewText', 'clean_text']]
query_text = question_review['reviewText']

# Load conversational model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Load a text similarity model
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define a knowledge base
knowledge_base = {
    "refund": "We're sorry to hear about your issue. Please contact our support team with your order details to process your refund.",
    "sizing": "If the size doesn't fit, you can exchange it for a different size or request a refund.",
    "general": "Thank you for reaching out. How can we assist you today?"
}

# Function to retrieve the best response from the knowledge base
def retrieve_response(query, knowledge_base, retriever_model):
    queries = list(knowledge_base.values())
    query_embedding = retriever_model.encode(query, convert_to_tensor=True)
    knowledge_embeddings = retriever_model.encode(queries, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, knowledge_embeddings)
    best_match_idx = torch.argmax(similarities)
    return queries[best_match_idx]

# Function to generate a conversational response
def generate_response(review, retrieved_response):
    # Combine the retrieved response with the review as input
    input_text = f"Customer query: {review}\nHelpful response: {retrieved_response}\nReply:"
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    response_ids = model.generate(
        input_ids,
        max_length= 300,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    return tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

# Retrieve the most relevant response
retrieved_response = retrieve_response(query_text, knowledge_base, retriever_model)

print(f"Question Review: {query_text}")
print(f"Retrieved Response: {retrieved_response}")

final_response = generate_response(query_text, retrieved_response)

# Validate if the response is complete
if len(final_response.strip()) < 10 or "refund" not in final_response.lower():
    final_response = f"{retrieved_response} Let us know if you need further assistance."

print(f"Final Automated Response: {final_response}")

