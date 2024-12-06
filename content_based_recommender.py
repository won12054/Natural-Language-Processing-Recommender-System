import pandas as pd
import contractions
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

'''
1. Load the dataset named "meta_Digital_Music.json.gz", 
attached to this assignment into a dataframe, name it songs_firstname. 
'''

file_path = 'C:/Users/Public/6th/NLP and Recommender Systems/assignment3/meta_Digital_Music.json.gz'

songs_jungyu = pd.read_json(file_path, lines=True, compression='gzip')

songs_jungyu.head()

songs_jungyu.info()


'''
2. Data exploration:
  a. Carry out a thorough exploration and note the results into your analysis report. 
     Make sure to check for empty data, not just null.
  b. Also, in your analysis report, 
     suggest which columns you will take into consideration for the recommender system and why. 
     (There are 19 columns in total)
  c. Based on the output of points a & b suggest any filtering steps, 
     for example, if you need to drop any columns or filter out any rows and explain why. 
     Write your explanation in the analysis report.
'''

# a
empty_counts = songs_jungyu.apply(lambda col: sum(col == "") if col.dtype == "object" else 0)
songs_jungyu.isnull().sum()

# b
filtered_jungyu = songs_jungyu[['title', 'description']]


# c
filtered_jungyu = filtered_jungyu[filtered_jungyu['title'] != '']

def is_invalid_list(value):
    return isinstance(value, list) and all(item.strip() in ['', '.', ','] for item in value)

filtered_jungyu = filtered_jungyu[~filtered_jungyu['description'].apply(is_invalid_list)]

filtered_jungyu.reset_index(drop=True, inplace=True)

filtered_jungyu.head()

'''
3. Feature engineering:
  a. Clean your data and prepare your feature space based on the results in point#2 above. 
     You might combine columns, transform…etc. 
     Make sure you follow all the recommendations you noted in your analysis report.
  b. If your feature space has text data, which most likely is the case:
    i. Pre-process the data and note the steps in your analysis report.
    ii. Create TF-IDF vectors for the textual description (or overview) of every song
  c. Compute the pairwise cosine similarity score of every song title.
  d. Store the recommendations into a separate file that your simple app will access.
'''
# a
filtered_jungyu['description'] = filtered_jungyu['description'].apply(
    lambda x: ''.join(x) if isinstance(x, list) else x
)

filtered_jungyu['combined_description'] = filtered_jungyu['title'] + " " + filtered_jungyu['description']

filtered_jungyu['combined_description'] = filtered_jungyu['combined_description'].str.lower()

filtered_jungyu[['title', 'description', 'combined_description']].head()

# b
# Removes punctuation and digits
filtered_jungyu['combined_description'] = filtered_jungyu['combined_description'].apply(
    lambda x: re.sub(r'[^\w\s]|[\d]', '', x)  
)

# Remove contractions
filtered_jungyu['combined_description'] = filtered_jungyu['combined_description'].apply(
    lambda x: contractions.fix(x) if isinstance(x, str) else x
)

# Remove stop words including negation words
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(filtered_jungyu['combined_description'])
tfidf_matrix.shape

# c
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim_df = pd.DataFrame(
    cosine_sim, 
    index=filtered_jungyu['title'], 
    columns=filtered_jungyu['title']
)

cosine_sim_df.head()

# d
cosine_sim_df.to_csv('C:/Users/Public/6th/NLP and Recommender Systems/assignment3/song_similarity_matrix.csv', index=True)



'''
4. Recommender function: Write the recommender function that takes in a song title as an argument 
   and outputs the top ten 'song titles' most similar to it.
  a. Your app should receive a 'song title' as input from the user, 
     for example, 'Long Legends', 'There can be miracles'…etc. 
     if the 'song title' is not available, 
     then reply to the user 'We don’t have recommendations for XXX' where XXX is the inputted song title. 
     Then prompt the user to enter a different 'song title'. (hint: use python input())
  b. If the 'song title' is available, present the top-10 most similar song titles back to the user as a recommendation.
  c. Continue accepting input from the user and responding until the user enters an “exit” text.
'''

def recommend_songs_with_df():
    print("\nWelcome to the Song Recommender System!")
    
    while True:
        user_input = input("\nEnter a song title (or type 'exit' to quit): ").strip()

        if user_input.lower() == 'exit':
            print("Exiting the recommender system. Have a great day!")
            break

        if user_input not in cosine_sim_df.index:
            print(f"We don’t have recommendations for '{user_input}'. Please try another title.")
            continue

        similar_scores = cosine_sim_df[user_input].sort_values(ascending=False)[1:11]

        print(f"\nTop 10 song recommendations for '{user_input}':")
        for i, (song, score) in enumerate(similar_scores.items(), start=1):
            print(f"{i}. {song} (Similarity: {score:.2f})")

recommend_songs_with_df()




