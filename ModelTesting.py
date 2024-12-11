import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction
from chromadb import Client
import requests

nltk.download('punkt')

GROQ_API_KEY = "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"

print("Using Groq API for inference.")

us_speeches_path = "speeches.xlsx"
russian_speeches_path = "speeches_russian_PM.xlsx"
us_speeches = pd.read_excel(us_speeches_path)
russian_speeches = pd.read_excel(russian_speeches_path)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token.isalnum()]
    return " ".join(tokens)

def process_dataset(speeches, column_names):
    speeches = speeches.dropna(subset=[column_names['transcript'], column_names['president']])
    speeches['transcript'] = speeches[column_names['transcript']].apply(preprocess_text)
    speeches['President'] = speeches[column_names['president']]
    return speeches[['date', 'title', 'transcript', 'President']]

russian_columns = {'transcript': 'transcript_filtered', 'president': 'President'}
us_columns = {'transcript': 'transcript', 'president': 'president'}

russian_speeches = process_dataset(russian_speeches, russian_columns)
us_speeches = process_dataset(us_speeches, us_columns)

combined_speeches = pd.concat([russian_speeches, us_speeches], ignore_index=True)
combined_speeches.to_excel("combined_presidents_speeches.xlsx", index=False)
print("Combined Dataset Saved")

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model):
        self.model = model

    def __call__(self, inputs):
        return self.model.encode(inputs).tolist()
    
# ---- Paths and Configuration ----
import os
PERSIST_DIRECTORY = "./chroma_db"  # Local directory for persistent storage
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)  # Ensure directory exists

chroma_client = Client(Settings(persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False))
retriever_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedding_function = SentenceTransformerEmbeddingFunction(retriever_model)

collection_name = "presidential_speeches"
collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

# ---- Add Data to Collection if Not Already Added ----
existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()
print(f"Existing IDs in collection: {len(existing_ids)}")

new_data = []
for index, row in combined_speeches.iterrows():
    doc_id = str(index)
    if doc_id not in existing_ids:  # Only add new data
        metadata = {
            'date': str(row['date']) if pd.notna(row['date']) else "Unknown Date",
            'title': row['title'] if pd.notna(row['title']) else "Untitled Speech",
            'President': row['President'] if pd.notna(row['President']) else "Unknown President"
        }
        collection.add(
            documents=[row['transcript']],
            metadatas=[metadata],
            ids=[doc_id]
        )
        new_data.append(doc_id)

print(f"Added {len(new_data)} new items to the collection.")
print(f"Total Items in Collection: {collection.count()}")


print("ChromaDB Collection is ready for use.")
import requests

# ---- Groq API Key ----
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

def chat_completion(messages, model="gpt-4", temperature=0.7, max_tokens=150, top_p=1):
    """
    Sends a chat completion request to the Groq API.
    
    Args:
        messages (list): A list of message objects representing the conversation so far.
        model (str): Model ID to use (e.g., "gpt-4").
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum tokens for the response.
        top_p (float): Nucleus sampling parameter.
    
    Returns:
        dict: The API response containing the model's completion.
    """
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    
    response = requests.post(GROQ_BASE_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


# def summarize_speech(text):
#     try:
#         response = groq_request("summarize", text, {"max_length": 130, "min_length": 30})
#         return response["summary"] if response else "Error in summarization."
#     except Exception as e:
#         return "Error in summarization."

# def extract_entities(text):
#     try:
#         response = groq_request("ner", text)
#         return response["entities"] if response else []
#     except Exception:
#         return []

# def classify_sentiment(text):
#     try:
#         response = groq_request("sentiment", text)
#         return response["label"], response["score"] if response else ("Unknown", 0.0)
#     except Exception:
#         return "Error", 0.0

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = vectorizer.fit_transform(combined_speeches['transcript'])
nmf_model = NMF(n_components=5, random_state=42)
nmf_model.fit(tfidf_matrix)

def determine_topic(text):
    if isinstance(text, list):
        text = " ".join(text)  # Convert list to single string
    text = preprocess_text(text)  # Ensure consistent preprocessing
    text_tfidf = vectorizer.transform([text])  # Wrap in list for TF-IDF
    topic_idx = nmf_model.transform(text_tfidf).argmax()
    return f"Topic {topic_idx}"


def preprocess_query(query):
    tokens = nltk.word_tokenize(query)
    tokens = [token for token in tokens if token.isalnum()]
    return " ".join(tokens)


def retrieve_similar_speeches(query, k=5, similarity_threshold=0.98):
    preprocessed_query = preprocess_query(query)
    results = collection.query(query_texts=[preprocessed_query], n_results=k)

    if results and 'documents' in results and 'metadatas' in results and 'distances' in results:
        documents = results['documents']
        metadatas = results['metadatas']
        distances = results['distances']

        if isinstance(distances[0], list):
            distances = [d[0] for d in distances]

        print("[DEBUG] Distances:", distances)

        # Fix invalid metadata format
        if not all(isinstance(meta, dict) for meta in metadatas):
            print("[DEBUG] Metadatas format is invalid. Attempting to fix.")
            metadatas = [{'date': "Unknown Date", 'title': "Untitled Speech", 'President': "Unknown President"} for _ in documents]

        # Filter by similarity threshold
        filtered_data = [
            {**meta, 'transcript': doc, 'similarity': sim}
            for meta, doc, sim in zip(metadatas, documents, distances)
            if isinstance(sim, (int, float)) and sim <= similarity_threshold
        ]

        if filtered_data:
            return pd.DataFrame(filtered_data)

    return pd.DataFrame()


# ---- Chatbot Logic ----
def conversational_chatbot():
    print("Welcome to the Chatbot using Groq API!")
    print("You can ask about speeches or have a general conversation.")
    print("Type 'exit' to quit.")
    
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Thank you for chatting. Goodbye!")
            break
        
        # Add user input to the conversation history
        messages.append({"role": "user", "content": user_input})
        
        # Get chat completion from Groq API
        response = chat_completion(messages)
        if response and "choices" in response:
            reply = response["choices"][0]["message"]["content"]
            print(f"Bot: {reply}")
            
            # Add bot reply to the conversation history
            messages.append({"role": "assistant", "content": reply})
        else:
            print("Error in getting a response. Please try again.")

# ---- Run Chatbot ----
conversational_chatbot()
