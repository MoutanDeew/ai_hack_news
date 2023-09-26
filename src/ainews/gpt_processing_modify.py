import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

# Text preprocessing
def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('russian'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = Mystem()
    tokens = [lemmatizer.lemmatize(word)[0] for word in tokens]
    processed_text = ' '.join(tokens)
    return processed_text

# Text vectors
def text_to_vector(text):
    if isinstance(text, str):
        vector = np.zeros(model.vector_size)
        word_count = 0
        for word in text.split():
                if word in model.wv:
                    vector += model.wv[word]
                    word_count += 1
                if word_count > 0:
                    vector /= word_count
                    return vector
                else:
                    return np.zeros(model.vector_size)

def process_text_data(input_file, output_file):
    # Load data
    df = pd.read_excel(input_file)

    # Text preprocessing
    nltk.download('stopwords')
    nltk.download('punkt')


    texts = df['text'].to_numpy()
    preprocessed_texts = np.vectorize(text_preprocessing)(texts)
    df['cleaned_text'] = preprocessed_texts

    # Word2Vec model
    model = Word2Vec(sentences=preprocessed_texts, vector_size=100, window=5, min_count=1, sg=0)


    text_vectors = [text_to_vector(text) for text in df['cleaned_text']]

    # Similarity matrix
    similarity_matrix = cosine_similarity(text_vectors, text_vectors)

    # Similarity threshold
    similarity_threshold = 0.9

    # Select most informative texts
    selected_indices = []

    for i in range(len(df)):
        max_word_count = 0
        most_informative_index = i

        for j in range(len(df_1)):
            if i != j and similarity_matrix[i][j] > similarity_threshold:
                word_count_i = len(df['cleaned_text'][i].split())
                word_count_j = len(df['cleaned_text'][j].split())

                if word_count_j > max_word_count:
                    max_word_count = word_count_j
                    most_informative_index = j

        if most_informative_index not in selected_indices:
            selected_indices.append(most_informative_index)

    selected_df = df.iloc[selected_indices]

    # Text vectors for selected texts
    text_vectors = [text_to_vector(text) for text in selected_df['cleaned_text']]

    # DBSCAN clustering
    eps = 0.2
    min_samples = 1
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(text_vectors)

    selected_df['cluster_label'] = cluster_labels

    # Save the result
    selected_df.to_excel(output_file, index=False)