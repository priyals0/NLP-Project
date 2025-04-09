import nltk
nltk.download('stopwords')
from nltk import FreqDist
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re

from sklearn import metrics

stops = stopwords.words('english')
stops.extend([",", ".", "!", "?", "'", '"', "I", "i", "n't", "'ve", "'d", "'s"])

allwords = []

# Process each true news article
true_df = pd.read_csv("True.csv")
true_articles = []

for idx, row in true_df.iterrows():
    # Combine 'title' and 'text'
    full_text = str(row['title']) + " " + str(row['text'])
    # Split into words and remove stopwords
    words = full_text.lower().split()
    filtered_words = list(set([w for w in words if w not in stops]))
    true_articles.append(filtered_words)
    allwords.extend(filtered_words)
    
    # Limit to 2000 articles
    if idx >= 2000:
        break

print(f"Processed {len(true_articles)} true news articles")

# Process fake news articles
fake_df = pd.read_csv("Fake.csv")
fake_articles = []

for idx, row in fake_df.iterrows():
    full_text = str(row['title']) + " " + str(row['text'])
    words = full_text.lower().split()
    filtered_words = list(set([w for w in words if w not in stops]))
    fake_articles.append(filtered_words)
    allwords.extend(filtered_words)
    
    if idx >= 2000:
        break

print(f"Processed {len(fake_articles)} fake news articles")

## Get the 1000 most frequent words
## These will be your features
wfreq = FreqDist(allwords)
top1000 = wfreq.most_common(1000)

training = []
traininglabel = []

# Process true articles for training
for article in true_articles:
    vec = []
    for t in top1000:
        if t[0] in article:
            vec.append(1)
        else:
            vec.append(0)
    training.append(vec)
    traininglabel.append(1)  # 1 for true news

# Process fake articles for training
for article in fake_articles:
    vec = []
    for t in top1000:
        if t[0] in article:
            vec.append(1)
        else:
            vec.append(0)
    training.append(vec)
    traininglabel.append(0)  # 0 for fake news

print(len(traininglabel)) # total training examples
print(len(training[0])) #feature vector length

from sklearn.model_selection import train_test_split

X = np.array(training)
y = np.array(traininglabel)

# Randomly split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# formatting the data to CNN shape
X_train = X_train.reshape(-1, 1000, 1)
X_test = X_test.reshape(-1, 1000, 1)

print('Shape of training data: ')
print(X_train.shape)
print(y_train.shape)
print('Shape of test data: ')
print(X_test.shape)
print(y_test.shape)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, Flatten

# Initialize the model
# 4 layers 2 convolutional, 1 flatten, 1 dense layer
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(1000, 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),

    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),

    Dense(1, activation='sigmoid')  # Binary classification (true/fake)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))