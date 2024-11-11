# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:08:26 2024

@author: 8778t
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup

def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def clean_common_jungyu(text):
    text = remove_html(text)

    text = re.sub(r'(?<!\d)[^\w\s](?!\d)', '', text)
        
    text = text.lower()
    
    return text

def clean_jungyu_text11(text):
    text = clean_common_jungyu(text)
    return text

def clean_jungyu_text12(text):
    text = clean_common_jungyu(text)
    return text

text11 = """The variety packs taste great!<br /><br /> I have them every morning. At $0.30 cents per meal, I don't understand why everyone on earth isn't buying this stuff up.<br /><br /> Maple and brown sugar is terrific, followed by apples and cinnamon, followed by regular."""
text12 = """You don't get tired of the same ole thing, Maple and brown sugar and they taste great. <br /><br />I just boil water from a small pot, empty the packet or 2 of Maple in a bowl, pour in boiling water, and watch it expand to 2x its size! <br /><br />Taste really good and takes minutes to prepare. <br /><br />Not sure why everyone on earth isn't this. Convenient, healthy, very quick, excellent quality, and extremely cheap."""

cleaned_text11 = clean_jungyu_text11(text11)
cleaned_text12 = clean_jungyu_text12(text12)

print("Before cleaning text11:\n", text11)
print("\nAfter cleaning text11:\n", cleaned_text11)
print("\nBefore cleaning text12:\n", text12)
print("\nAfter cleaning text12:\n", cleaned_text12)

def tfidf_jungyu(texts):
    vectorizer = TfidfVectorizer()
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    idf_values = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    
    print("IDF for 'sugar':", idf_values.get('sugar', 'Not found'))
    print("IDF for 'maple':", idf_values.get('maple', 'Not found'))

tfidf_jungyu([cleaned_text11, cleaned_text12])
