# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 21:00:38 2024

Group 1 NLP
Pratheepan
Alejandro
Jungyu
Sirada
Zahin

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
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')


os.chdir('C:/Users/venus/Documents/03 Study/01 AI/Fall2024/NLP/Project')


#load the data
df = pd.read_json('AMAZON_FASHION_5.json', lines=True)
df.head()

'''
Task 1a: Counts, Averages
'''
total_reviews = len(df)
unique_products = df['asin'].nunique()
unique_users = df['reviewerID'].nunique()
verified_purchases = df['verified'].sum()
average_rating = df['overall'].mean()

print(df.info())
print("Total Reviews:", total_reviews)
print("Unique Products:", unique_products)
print("Unique Users:", unique_users)
print("Verified Purchases:", verified_purchases)
print("Average Rating:", average_rating)


'''
Task 1b: Distribution of the number of reviews across products
'''
reviews_per_product = df['asin'].value_counts()
print(reviews_per_product)

# Distribution of the number of reviews across products
reviews_per_product_review = df.groupby('asin')
reviews_per_product_review.size().plot.hist(figsize=(10,4), bins=25, title="Distributions of reviews across products")
plt.xlabel('Distributions of reviews per product')

'''
Task 1b: ASIN with exactly 1 review
'''
asin_with_one_review = reviews_per_product[reviews_per_product == 1]

count_asin_with_one_review = asin_with_one_review.count()

print(f"Number of ASINs with exactly 1 review: {count_asin_with_one_review}")

'''
Task 1b: Top 20 products with at least 50 reviews
'''
min_reviews = 50

reviews_per_product = df['asin'].value_counts()
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

'''
Task 1c: Distribution of reviews per product (histogram)

Each product has about 4.75 reviews.
'''
reviews_per_product_hist = reviews_per_product.describe()
print(reviews_per_product_hist)

reviews_per_product_review.size().plot.bar(figsize=(10,4), title="Number of reviews per product")
plt.ylabel('Count')

'''
Task 1d: Distribution of reviews per user (super reviewers)
'''
reviews_per_user = df['reviewerID'].value_counts()
print(reviews_per_user)

plt.figure(figsize=(10, 6))
plt.hist(reviews_per_user, bins=50, color='coral')
plt.xlim(0, 10)
plt.title("Distribution of Reviews per User", fontsize=15)
plt.xlabel("Number of Reviews per User", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(False)
plt.show()

'''
Task 1e: Review lengths and outliers

- Mean word count 28
- Max word count 2088
'''
df['review_length'] = df['reviewText'].apply(lambda x: len(str(x).split()))
review_length_stats = df['review_length'].describe()

print(review_length_stats)

'''
Task 1g: Check for duplicates
'''
duplicate_reviews = df.duplicated(subset=['reviewTime', 'asin', 'reviewerID', 'reviewText']).sum()

print("Duplicate Reviews:", duplicate_reviews)


'''
Task 1h: Check for HTML tags
'''
def contains_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return len(soup.find_all()) > 0

html_tagged_reviews = df['reviewText'].apply(lambda x: contains_html(str(x)))

html_tag_count = html_tagged_reviews.sum()

print(f'Reviews with HTML tags: {html_tag_count}')
print(df[html_tagged_reviews == True]['reviewText'].head())

#check URL
def contains_url(text):
    url_pattern = r'http[s]?://'
    return re.search(url_pattern, text) is not None

html_tagged_reviews = df['reviewText'].apply(lambda x: contains_url(str(x)))

html_tag_count = html_tagged_reviews.sum()
print(f'Reviews with HTML tags or URLs: {html_tag_count}')
print(df[html_tagged_reviews == True]['reviewText'].head())


'''
Task 1i: Check for null or empty reviewText
'''
null_reviews = df['reviewText'].isnull().sum()
empty_reviews = (df['reviewText'].str.strip() == '').sum()

print(f'Null reviews: {null_reviews}')
print(f'Empty reviews: {empty_reviews}')

print(df[df['reviewText'].isnull()][['reviewText']].head())
print(df[df['reviewText'].str.strip() == ''][['reviewText']].head())


'''
Task 1j: Check for emails
'''
def contains_email(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return bool(re.search(email_pattern, text))

email_mentions = df['reviewText'].apply(lambda x: contains_email(str(x)) if isinstance(x, str) else False)

email_count = email_mentions.sum()

print(f'Reviews with email addresses: {email_count}')
print(df[email_mentions == True]['reviewText'])

'''
Task 1k: Check for user mentions
'''
def contains_user_mentions(text):
    return bool(re.search(r'@\w+', text))

user_mentions = df['reviewText'].apply(lambda x: contains_user_mentions(str(x)) if isinstance(x, str) else False)

mention_count = user_mentions.sum()

print(f'Reviews with user mentions: {mention_count}')
print(df[user_mentions == True]['reviewText'])

'''
Task 1l: Distribution of Ratings
'''
rating_distribution = df['overall'].value_counts().sort_index()

plt.figure(figsize=(8, 6))
colors = ['skyblue', 'coral', 'lightgreen', 'gold', 'lightpink']
rating_distribution.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

'''
Task 1m: Distribution of Verified vs Non-Verified Reviews
'''
plt.figure(figsize=(8, 6))
sns.countplot(x='verified', data=df, palette='Set2')
plt.title('Verified vs Non-Verified Reviews')
plt.xlabel('Verified Purchase')
plt.ylabel('Count')
plt.show()

'''
Task 1n: Distribution of Review Times
'''
df['reviewTime'] = pd.to_datetime(df['reviewTime'], errors='coerce')
df['year'] = df['reviewTime'].dt.year
plt.figure(figsize=(10, 6))
sns.countplot(x='year', data=df, palette='Set2')
plt.title('Reviews Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

'''
Task 1p. Check non-english
'''
model = fasttext.load_model('lid.176.bin')

def detect_language_fasttext(text):
    try:
        predictions = model.predict(text, k=1)
        lang = predictions[0][0].replace('__label__', '')
        return lang
    except:
        return 'unknown'

df['language'] = df['reviewText'].apply(detect_language_fasttext)

non_english_reviews = df[df['language'] != 'en']

print(f'Non-English reviews: {non_english_reviews.shape[0]}')

if not non_english_reviews.empty:
    print("Example non-English reviews:")
    print(non_english_reviews[['reviewText', 'language']].head(50))
else:
    print("All reviews are in English.")
    
language_counts = df['language'].value_counts()

print(language_counts)

plt.figure(figsize=(10, 6))
sns.barplot(x=language_counts.index, y=language_counts.values, palette='Set2')

plt.title('Distribution of Languages in Reviews', fontsize=15)
plt.xlabel('Language', fontsize=12)
plt.ylabel('Number of Reviews', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

'''
Task 2a: Label the data based on product rating
'''
def sentiment_label(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

df['sentiment'] = df['overall'].apply(sentiment_label)

def plot_sentiment_count_distribution(df, title):
    plt.figure(figsize=(10, 6))

    sns.countplot(x='sentiment', data=df, order=df['sentiment'].value_counts().index, palette="rainbow", dodge=False)

    plt.title(title)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    sentiment_counts = df['sentiment'].value_counts()
    for i in range(len(sentiment_counts)):
        plt.text(i, sentiment_counts.values[i] + 0.5, f'{sentiment_counts.values[i]}', ha='center')

    plt.show()

plot_sentiment_count_distribution(df, "Sentiment Count Distribution")

def plot_sentiment_percentage_distribution(df, title):
    sentiment_counts = df['sentiment'].value_counts()
    sentiment_percentages = 100 * sentiment_counts / len(df)

    colors = sns.color_palette("rainbow", len(sentiment_counts))

    plt.figure(figsize=(10, 6))
    sns.barplot(x=sentiment_percentages.index, y=sentiment_percentages.values, palette=colors, dodge=False)

    plt.title(title)
    plt.xlabel('Sentiment')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)

    for i in range(len(sentiment_percentages)):
        plt.text(i, sentiment_percentages.values[i] + 0.5, f'{sentiment_percentages.values[i]:.2f}%', ha='center')

    plt.show()

plot_sentiment_percentage_distribution(df, "Sentiment Percentage Distribution")

'''
Task 2b: Choose the appropriate columns for sentiment analysis

- 'reviewText': This contains the full text of the review, which is crucial for sentiment analysis.
- 'sentiment': The sentiment label we just created based on the rating.
'''
selected_columns = df[['reviewText', 'sentiment']]

'''
Task 2c: Check for outliers in terms of number of word in the review
'''
df['review_length'] = df['reviewText'].apply(lambda x: len(str(x).split()))
outliers = df[(df['review_length'] > df['review_length'].quantile(0.99)) | (df['review_length'] < df['review_length'].quantile(0.01))]

print("Number of Outliers:", len(outliers))

print(outliers['review_length'].head())

max_word_count = df['reviewText'].apply(lambda x: len(str(x).split())).max()
print(f'Maximum review word count: {max_word_count}')


def plot_word_count_distribution(df, title):
    df['word_count'] = df['reviewText'].apply(lambda x: len(str(x).split()))

    percentile_50 = df['word_count'].quantile(0.50)
    percentile_75 = df['word_count'].quantile(0.75)
    percentile_90 = df['word_count'].quantile(0.90)

    plt.figure(figsize=(10, 6))
    sns.histplot(df['word_count'], bins=50, kde=True)

    plt.axvline(percentile_50, color='red', linestyle='--', label=f'50th percentile: {int(percentile_50)} words')
    plt.axvline(percentile_75, color='green', linestyle='--', label=f'75th percentile: {int(percentile_75)} words')
    plt.axvline(percentile_90, color='blue', linestyle='--', label=f'90th percentile: {int(percentile_90)} words')

    plt.xlim(0, 300)
    plt.title(title)
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    print(f"50th percentile: {int(percentile_50)} words")
    print(f"75th percentile: {int(percentile_75)} words")
    print(f"90th percentile: {int(percentile_90)} words")

plot_word_count_distribution(df, "Review Word Count Distribution")


def plot_word_count_per_label(df, title):
    df['word_count'] = df['reviewText'].apply(lambda x: len(str(x).split()))

    plt.figure(figsize=(14, 8))

    colors = sns.color_palette("rainbow", len(df['sentiment'].unique()))

    sns.boxplot(x='sentiment', y='word_count', data=df, order=df['sentiment'].value_counts().index, hue='sentiment', palette=colors)

    plt.title(title)
    plt.xlabel('Sentiment Label')
    plt.ylabel('Review Length (Number of Words)')
    plt.xticks(rotation=45)

    plt.legend([],[], frameon=False)

    plt.show()

plot_word_count_per_label(df, "Review Length Distribution by Sentiment Label")

'''
Common Preprocessing
'''

def common_preprocessing(df_original):
    #4a. Remove outliers (less than 1% of reviews)
    df_clean = df_original[(df_original['review_length'] <= df_original['review_length'].quantile(0.99)) &
                (df_original['review_length'] >= df_original['review_length'].quantile(0.01))]
    print(f"Dataframe size after removing outliers: {df_clean.shape[0]}")

    '''
    4b. Remove duplicates based on 'reviewTime', 'asin, 'reviewerID' and 'reviewText'
    '''
    df_clean = df_clean.drop_duplicates(subset=['reviewTime', 'asin', 'reviewerID', 'reviewText'], keep='first')

    print(f'Remaining reviews after removing duplicates: {df_clean.shape[0]}')

    '''
    4c. Remove null and empty reviews
    '''
    df_clean = df_clean.dropna(subset=['reviewText'])

    df_clean = df_clean[df_clean['reviewText'].str.strip() != '']

    print(f'Remaining reviews after removing null and empty reviews: {df_clean.shape[0]}')

    '''
    4d. Remove non-English reviews
    '''
    df_clean = df_clean[df_clean['language'] == 'en']
    print(f'Remaining reviews after removing non-English reviews: {df_clean.shape[0]}')
    
    return df_clean

df = common_preprocessing(df)

'''
3a-4 . Vader Preprocessing

VADER Preprocessing Steps:
- Keep punctuation, capitalization, and emojis: VADER considers these elements important for sentiment analysis, so we won’t modify them.
- No need for stopword removal or lemmatization: VADER performs well without these steps.
'''
df_vader = df.copy()
df_vader = df_vader[['reviewText', 'sentiment']]

'''
3b-4. TextBlob Preprocessing

TextBlob Preprocessing Steps:
- Remove contractions
- Lowercase conversion
- Remove stopwords & lemmatize
- Keep negation words
'''
df_textblob = df.copy()
df_textblob = df_textblob[['reviewText', 'sentiment']]

lemmatizer = WordNetLemmatizer()

def preprocess_textblob(text):
    expanded_text = contractions.fix(text)

    expanded_text = expanded_text.lower()

    words = word_tokenize(expanded_text)

    stop_words = set(stopwords.words('english'))

    negation_words = {"not", "no", "never", "none", "nor"}

    filtered_words = [lemmatizer.lemmatize(word) for word in words if (word not in stop_words or word in negation_words) and word not in string.punctuation]

    return ' '.join(filtered_words)

df_textblob['reviewText'] = df_textblob['reviewText'].apply(preprocess_textblob)

print('Vader example: ', df_vader['reviewText'].iloc[10])
print('Textblob example: ',df_textblob['reviewText'].iloc[10])

'''
5. Randomly select 1000 reviews from your dataset
(Keep track of sample_indices)
'''
sample_indices = df_vader.sample(n=1000, random_state=42).index

df_vader_sample = df_vader.loc[sample_indices]

df_textblob_sample = df_textblob.loc[sample_indices]

'''
5. Sentiment distribution in the sample
(As vader and textblob uses same sample_indices, the distribution is equal)
'''
vader_sentiment_counts = df_vader_sample['sentiment'].value_counts()
print("Sentiment distribution in VADER sample:")
print(vader_sentiment_counts)

textblob_sentiment_counts = df_textblob_sample['sentiment'].value_counts()
print("\nSentiment distribution in TextBlob sample:")
print(textblob_sentiment_counts)


'''
6a - Build VADER Sentiment Analysis Model
'''

vader_analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    scores = vader_analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df_vader_sample['predicted_sentiment'] = df_vader_sample['reviewText'].apply(get_vader_sentiment)

'''
6b - Build TextBlob Sentiment Analysis Model
'''
def get_textblob_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df_textblob_sample['predicted_sentiment'] = df_textblob_sample['reviewText'].apply(get_textblob_sentiment)

'''
7a. Evaluate VADER Model
'''

vader_accuracy = accuracy_score(df_vader_sample['sentiment'], df_vader_sample['predicted_sentiment'])

vader_report = classification_report(df_vader_sample['sentiment'], df_vader_sample['predicted_sentiment'])

print(f"VADER Accuracy: {vader_accuracy}")
print("VADER Classification Report:")
print(vader_report)

'''
7b. Evaluate TextBlob Model
'''
textblob_accuracy = accuracy_score(df_textblob_sample['sentiment'], df_textblob_sample['predicted_sentiment'])

textblob_report = classification_report(df_textblob_sample['sentiment'], df_textblob_sample['predicted_sentiment'])

print(f"TextBlob Accuracy: {textblob_accuracy}")
print("TextBlob Classification Report:")
print(textblob_report)
