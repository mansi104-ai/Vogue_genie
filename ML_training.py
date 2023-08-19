import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download the necessary resources (only needs to be done once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load CSV data into a DataFrame
df = pd.read_csv("flipkart_com-ecommerce_sample.csv")

# Preprocess and tokenize the "product_category_tree" column
product_categories = df['product_category_tree']
preprocessed_categories = []

for category in product_categories:
    category = category.lower()
    category = re.sub(r'[^\w\s]', '', category)
    tokens = word_tokenize(category)
    preprocessed_categories.append(tokens)

preprocessed_categories_df = pd.DataFrame({'Preprocessed_Category': preprocessed_categories})

# Preprocess and tokenize the "description" column
descriptions = df['description']
preprocessed_descriptions = []

# Load the list of stopwords
stop_words = set(stopwords.words('english'))

# Initialize the stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Initialize vocabulary dictionary and keyword set
vocabulary = defaultdict(int)
keyword_set = set()

for description in descriptions:
    if isinstance(description, str):
        description = description.lower()
        description = re.sub(r'[^\w\s]', '', description)
        tokens = word_tokenize(description)
        
        # Apply stemming and lemmatization to each token
        stemmed_tokens = [stemmer.stem(word) for word in tokens]
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Remove stopwords from both stemmed and lemmatized tokens
        filtered_stemmed_tokens = [word for word in stemmed_tokens if word not in stop_words]
        filtered_lemmatized_tokens = [word for word in lemmatized_tokens if word not in stop_words]
        
        preprocessed_descriptions.append(filtered_stemmed_tokens)  # You can also use filtered_lemmatized_tokens here
        
        # Update vocabulary dictionary and keyword set
        for word in filtered_stemmed_tokens:
            vocabulary[word] += 1
            keyword_set.add(word)
    else:
        preprocessed_descriptions.append([])

preprocessed_descriptions_df = pd.DataFrame({'Preprocessed_Description': preprocessed_descriptions})

# Concatenate the preprocessed data columns with the original DataFrame
df = pd.concat([df, preprocessed_categories_df, preprocessed_descriptions_df], axis=1)

# Save the modified DataFrame to a new CSV file
df.to_csv('preprocessed_data.csv', index=False)

# Print the total number of unique keywords
print("Total number of unique keywords:", len(keyword_set))

# Matching and Ranking
# Combine preprocessed categories and descriptions into a single column for matching
df['Combined'] = df['Preprocessed_Category'] + df['Preprocessed_Description']

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Combined'].apply(lambda x: ' '.join(x)))

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Define a function to get top matching products
def get_top_matches(product_id, num_matches=5):
    idx = df.index[df['pid'] == product_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_matches+1]
    product_indices = [i[0] for i in sim_scores]
    return df.iloc[product_indices]

# Example: Get top 5 matches for the product at index 0
top_matches = get_top_matches(df['pid'][0], num_matches=5)
print(top_matches[['product_name', 'pid']])

print("Preprocessed data with stemming and stopwords removed saved to 'preprocessed_data.csv'")
