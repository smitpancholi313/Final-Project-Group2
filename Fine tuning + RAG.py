import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
import pandas as pd
# Hardcoded path
filepath = "/home/ubuntu/sairam/Project/speeches.xlsx"
data = pd.read_excel(filepath)
print(data.head())

#%%
# print(data['transcript'])

output_dir = "/home/ubuntu/sairam/Project/fine_tuned_president_20_epochs"
# Load your fine-tuned GPT-2 model and tokenizer
fine_tuned_model = GPT2LMHeadModel.from_pretrained(output_dir).to(device)
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

# Create text generation pipeline for the fine-tuned GPT-2 model
fine_tuned_generator = pipeline(
    "text-generation",
    model=fine_tuned_model,
    tokenizer=fine_tuned_tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Load and preprocess the dataset (break transcripts into chunks)
chunk_size = 100
chunks = []

# Break the transcript into smaller chunks of 300 words
for speech in data['transcript']:
    words = speech.split()
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(chunks)


# Function to retrieve the most relevant chunk based on cosine similarity
def get_most_relevant_chunk(question, top_n=2):
    """
    Retrieve the top `n` most relevant chunks for the given question based on TF-IDF cosine similarity.

    Args:
    - question (str): The input question to query.
    - top_n (int): The number of top relevant chunks to retrieve. Default is 10.

    Returns:
    - combined_context (str): Combined text from the most relevant chunks.
    """
    # Convert question to TF-IDF vector
    question_vector = vectorizer.transform([question])

    # Compute cosine similarity between the question and the chunks
    similarities = cosine_similarity(question_vector, tfidf_matrix)

    # Get indices of the top `n` most similar chunks
    top_n_indices = np.argsort(similarities[0])[-top_n:][::-1]

    # Combine the top `n` most relevant chunks into one context
    combined_context = " ".join([chunks[idx] for idx in top_n_indices])

    return combined_context


# Function to answer the question using the fine-tuned GPT-2 model and prompt engineering
def answer_question(question):
    # Retrieve the most relevant context using cosine similarity
    relevant_context = get_most_relevant_chunk(question)

    # Create the prompt for the fine-tuned GPT-2 model
    prompt = f"Context: {relevant_context}\n\nQuestion: {question}\nAnswer:"

    # Generate the response from the fine-tuned GPT-2 model
    response = fine_tuned_generator(
        prompt,
        max_new_tokens=100,  # Limit the number of new tokens generated
        num_return_sequences=1,
        pad_token_id=fine_tuned_tokenizer.eos_token_id,
        truncation=True  # Ensure truncation if the prompt is too long
    )

    # Return only the generated answer (not the entire prompt)
    return response[0]['generated_text'].split('Answer:')[-1].strip()


# Interactive Q&A loop
print("Ask a question to the President (type 'exit' to quit):")

while True:
    input_text = input("You: ")
    if input_text.lower() == "exit":
        print("Exiting... Goodbye!")
        break

    # Get the answer to the user's question using the fine-tuned model
    answer = answer_question(input_text)

    # Display the answer
    print(f"President: {answer}")
