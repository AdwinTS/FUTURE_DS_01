import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset (Replace 'your_dataset.csv' with your actual filename)
df = pd.read_csv('/kaggle/input/adwinfutureinterstask1/sentimentdataset.csv')  # Update with the correct path

# Display the first few rows
print(df.head())

# Assume dataset has a column named 'text' with tweets/posts
if 'Text' not in df.columns:
    raise ValueError("Dataset must have a 'text' column.")

# Function to clean text data
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = ' '.join(word for word in text.split() if word.isalnum())  # Remove special characters
    return text

df['cleaned_text'] = df['Text'].apply(clean_text)

# Sentiment Analysis using TextBlob
df['sentiment'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['sentiment_label'] = df['sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

# Trending Topics (Word Frequency)
vectorizer = CountVectorizer(stop_words='english', max_features=20)
X = vectorizer.fit_transform(df['cleaned_text'])
word_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))

# Word Cloud Visualization
plt.figure(figsize=(10, 5))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Trending Words")
plt.show()

# Sentiment Distribution
plt.figure(figsize=(6,4))
df['sentiment_label'].value_counts().plot(kind='bar', color=['green', 'gray', 'red'])
plt.title("Sentiment Analysis Results")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Save processed dataset
df.to_csv('/kaggle/working/processed_data.csv', index=False)
print("Processed data saved as 'processed_data.csv'.")