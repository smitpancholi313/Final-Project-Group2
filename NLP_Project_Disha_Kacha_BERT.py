import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Load the pre-trained models
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Use a smaller, faster model
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')  # BART model for text generation
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')  # Tokenizer for BART

# If a GPU is available, move the models to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bart_model.to(device)

# Load your dataset (replace with the correct file path)
df = pd.read_excel('speeches.xlsx')

# Extracting the 'transcript' column as the list of documents
documents = df['transcript'].tolist()

# Precompute the document embeddings (done only once)
document_embeddings = sentence_model.encode(documents, batch_size=32, show_progress_bar=True)

# Function to retrieve documents based on embedding similarity (using Sentence-BERT)
def retrieve_documents_with_embeddings(query, documents, top_n=5):
    # Encode the query
    query_embedding = sentence_model.encode([query])

    # Compute cosine similarity between the query and all document embeddings
    cosine_similarities = cosine_similarity(query_embedding, document_embeddings).flatten()

    # Get the top N most relevant documents (sorted by descending relevance)
    top_n_indices = np.argsort(cosine_similarities)[-top_n:][::-1]  # Sorted in descending order
    retrieved_docs = [documents[i] for i in top_n_indices]
    return retrieved_docs, cosine_similarities[top_n_indices]

# Function to generate answers using BART model
def generate_answer(question, context):
    # Prepare the input format for BART model
    input_text = f"Answer the question based on the context provided: question: {question} context: {context}"
    print("Input text: ", input_text)
    # Tokenize the input
    inputs = bart_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)

    # Generate the output answer with increased max_length and other parameters to get a longer response
    output = bart_model.generate(inputs,
                                 max_length=1000,  # Increase max length for longer answers
                                 num_beams=5,  # Use beam search for better quality
                                 no_repeat_ngram_size=3,  # Prevent repetition of n-grams
                                 early_stopping=True)  # Stop early when a complete answer is generated

    # Decode the generated output
    answer = bart_tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Main program
if __name__ == "__main__":
    # Step 1: Input question from user
    question = input("Enter a question: ")

    # Step 2: Retrieve the most relevant documents using precomputed embeddings
    retrieved_documents, scores = retrieve_documents_with_embeddings(question, documents, top_n=3)  # Reduced to 3 for faster results

    # Debug: Print the retrieved documents and their similarity scores
    print("\nRetrieved Documents:")
    for i, doc in enumerate(retrieved_documents):
        print(f"Doc {i + 1} (Score: {scores[i]:.4f}): {doc[:1000]}...")  # Preview first 1000 characters of each document

    # Step 3: Combine the top N most relevant documents into a context for answer generation
    combined_context = " ".join(retrieved_documents)  # Combine top retrieved documents

    # Step 4: Generate the answer using the BART model
    answer = generate_answer(question, combined_context)

    # Step 5: Output the generated answer
    print("\nGenerated Answer:", answer)
