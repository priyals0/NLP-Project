# DATA EXPLORATION - LDA + KMEANS

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load datasets
fake_df = pd.read_csv("Fake.csv", nrows = 1000)
real_df = pd.read_csv("True.csv", nrows = 1000)

# Combine datasets and add labels
fake_df['label'] = 'fake'
real_df['label'] = 'real'
df = pd.concat([fake_df, real_df])

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters & numbers
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# Vectorization using CountVectorizer for LDA
count_vectorizer = CountVectorizer(max_features=5000)
count_matrix = count_vectorizer.fit_transform(df['clean_text'])

# Apply LDA for topic modeling
num_topics = 10  # Adjust as needed
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(count_matrix)

# Display topics
def display_topics(model, feature_names, num_top_words):
    print(f"\n▶️ Top words per topic from LDA topic modeling ({num_topics} topics):")
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}: ", " ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))
    print("\n")

# shows the top 10 words in each topic to give a sense of what each topic is about
display_topics(lda, count_vectorizer.get_feature_names_out(), 10)

# Vectorization using TF-IDF for clustering
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
newsvectors = tfidf_vectorizer.fit_transform(df['clean_text'])

# K-Means clustering
n_clusters = 10 # Adjust as needed, group the articles into 25 clusters
kmnews = KMeans(n_clusters, random_state=0) 
newsclusters = kmnews.fit_predict(newsvectors)

# shows the top TF-IDF terms closest to each cluster center:
def print_top_terms_per_cluster(tfidf_vectorizer, kmeans_model, n_terms=10):
    terms = tfidf_vectorizer.get_feature_names_out()
    centroids = kmeans_model.cluster_centers_

    print(f"▶️ Top terms per cluster from K-Means clustering of news articles ({n_clusters} clusters):")
    for i, centroid in enumerate(centroids):
        print(f"Cluster {i}: ", end='')
        top_indices = centroid.argsort()[-n_terms:][::-1]
        top_terms = [terms[ind] for ind in top_indices]
        print(" ".join(top_terms))
    print("\n")

print_top_terms_per_cluster(tfidf_vectorizer, kmnews)

# Reduce dimensions to 2D for visualization
svd = TruncatedSVD(n_components=2, random_state=0)
reduced_vectors = svd.fit_transform(newsvectors)

# Plot the clustered data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1], hue=newsclusters, palette='tab20', legend=False)
plt.title('K-Means Clustering of News Articles (2D Projection)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()