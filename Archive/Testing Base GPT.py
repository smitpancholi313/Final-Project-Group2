import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
import pandas as pd

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# Load dataset
filepath = "/home/ubuntu/sairam/Project/speeches.xlsx"
data = pd.read_excel(filepath)
print(data.head())

# Load pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Create text generation pipeline for the pre-trained GPT-2 model
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Preprocess the dataset (break transcripts into chunks)
chunk_size = 100
chunks = []

for speech in data['transcript']:
    words = speech.split()
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(chunks)

# Function to retrieve the most relevant chunk based on cosine similarity
def get_most_relevant_chunk(question, top_n=2):
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, tfidf_matrix)
    top_n_indices = np.argsort(similarities[0])[-top_n:][::-1]
    combined_context = " ".join([chunks[idx] for idx in top_n_indices])
    return combined_context

# Function to answer the question using GPT-2 and prompt engineering
def answer_question(question):
    relevant_context = get_most_relevant_chunk(question)
    prompt = f"Answer the question as if you were a US president, given the context.\n\nContext: {relevant_context}\n\nQuestion: {question}\nAnswer:"
    response = text_generator(
        prompt,
        max_new_tokens=300,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        truncation=True
    )
    return response[0]['generated_text'].split('Answer:')[-1].strip()

# Interactive Q&A loop
print("Ask a question to the President (type 'exit' to quit):")
while True:
    input_text = input("You: ")
    if input_text.lower() == "exit":
        print("Exiting... Goodbye!")
        break
    answer = answer_question(input_text)
    print(f"President: {answer}")
