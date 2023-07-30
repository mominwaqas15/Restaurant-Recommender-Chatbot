import pandas as pd
import nltk
import spacy
from spacy.matcher import Matcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from questionary import text, select, confirm

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

df = pd.read_csv("restaurants_data_analysis.csv")

# Filter out data where the country is Pakistan
df = df[df["country"] == "Pakistan"]
df = df[df['is_active'] == 1]

df = df[['budget', 'latitude', 'longitude', 'name', 'rating','review_number', 'city', 'dine_in', 'main_cuisine','country']]


df = df.drop_duplicates()
df = df.dropna()
df.reset_index(drop=True, inplace=True)

df[['main_cuisine', 'city']] = df[['main_cuisine', 'city']].apply(lambda row: row.astype(str).str.lower())

df.to_csv("Restaurants_Cleaned_Dataset")

nltk.download('wordnet')

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Tokenize the text and lemmatize the words
    doc = nlp(text)
    lemmatized_text = " ".join(token.lemma_ for token in doc)
    return lemmatized_text

def extract_city(text):
    # Use spaCy NER to extract location entities (cities) from the text
    doc = nlp(text)
    cities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    if cities:
        return cities[0]  # Return the first city mentioned in the text
    else:
        return None

def get_user_input():
    name = text("Chatbot: Hi! What's your name?").ask()

    text(f"Chatbot: Nice to meet you, {name}!")

    budget = int(text("Chatbot: What is your budget for the restaurant (in PKR)?").ask())

    rating = float(text("Chatbot: What rating of restaurant are you looking for?").ask())

    available_cities = df['city'].unique()
    city = select("Chatbot: Which city are you in? Please select from the available options:", choices=available_cities).ask()

    dine_in = confirm("Chatbot: Do you prefer dine-in?").ask()

    available_cuisines = df['main_cuisine'].unique()
    cuisines = select("Chatbot: Which cuisines are you interested in? Please select from the available options:", choices=available_cuisines, qmark='>').ask()

    return name, budget, rating, city, dine_in, cuisines

def recommend_restaurants(name, budget, rating, city, dine_in, cuisines, df, num_options=10, similarity_threshold=0.3):
    # Preprocess the user input
    user_input = f"I want to have {cuisines} under {budget} PKR in {city}"
    preprocessed_input = preprocess_text(user_input)

    # Filter the DataFrame to include only the restaurants in the specified city
    df_city = df[df["city"].str.lower() == city.lower()]

    # Create a TF-IDF vectorizer to convert text into numerical features
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_city["name"] + " " + df_city["main_cuisine"])

    # Calculate the similarity score between the user input and each restaurant name and cuisine
    input_tfidf = tfidf_vectorizer.transform([preprocessed_input])
    cosine_similarities = linear_kernel(input_tfidf, tfidf_matrix).flatten()

    # Sort the restaurants by similarity score in descending order
    sorted_indices = cosine_similarities.argsort()[::-1]

    # Select restaurants with similarity scores above the threshold
    recommended_restaurants = []
    for idx in sorted_indices:
        if cosine_similarities[idx] >= similarity_threshold:
            restaurant = df_city.iloc[idx]
            recommended_restaurants.append(restaurant)
            if len(recommended_restaurants) >= num_options:
                break

    return recommended_restaurants

# User input
name, budget, rating, city, dine_in, cuisines = get_user_input()

# Get the restaurant recommendations
restaurants = recommend_restaurants(name, budget, rating, city, dine_in, cuisines, df, num_options=10, similarity_threshold=0.3)

# Display the recommendations
if restaurants:
    print(f"Chatbot: Here are some restaurant options for you in {restaurants[0]['city']}, {name}:")
    for idx, restaurant in enumerate(restaurants, 1):
        print(f"{idx}. '{restaurant['name']}' for {restaurant['main_cuisine']} cuisine.")
else:
    print(f"Chatbot: I'm sorry, but I couldn't find any suitable restaurants based on your preferences, {name}.")