import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
import pandas as pd
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# Hardcoded path
filepath = "/home/ubuntu/sairam/Project/speeches.xlsx"
data = pd.read_excel(filepath)
print(data.head())

# Load your fine-tuned GPT-2 model and tokenizer
output_dir = "/home/ubuntu/sairam/Project/fine_tuned_president_20_epochs"
fine_tuned_model = GPT2LMHeadModel.from_pretrained(output_dir).to(device)
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

# Create text generation pipeline for the fine-tuned GPT-2 model
fine_tuned_generator = pipeline(
    "text-generation",
    model=fine_tuned_model,
    tokenizer=fine_tuned_tokenizer,
    device=0 if torch.cuda.is_available() else -1
)


def preprocess_and_chunk(data, column="transcript", chunk_size=1000, chunk_overlap=20):
    """
    Preprocess and split the text data into chunks.

    Args:
    - data (pd.DataFrame): Input DataFrame containing transcripts.
    - column (str): Name of the column with text to chunk.
    - chunk_size (int): Size of each chunk in characters.
    - chunk_overlap (int): Overlap size between chunks.

    Returns:
    - chunks (list): List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for speech in data[column]:
        chunks = text_splitter.split_text(speech)
        all_chunks.extend(chunks)
    return all_chunks


# Chunk the data
chunks = preprocess_and_chunk(data)

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

# Initialize the SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define a wrapper for the SentenceTransformer embedding function
class SentenceTransformerWrapper:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=True).tolist()

# Wrap the SentenceTransformer model
embedding_function = SentenceTransformerWrapper(embedding_model)

# Initialize Chroma with the wrapper
vectorstore_dir = "/home/ubuntu/sairam/Project/chromadb_store"

chroma_db = Chroma(
    persist_directory=vectorstore_dir,
    embedding_function=embedding_function
)


#%%
ct=0
# Add chunks to ChromaDB
for idx, chunk in enumerate(chunks):

    print(f"New chunk {ct}out of ",len(chunks))
    ct=ct+1
    chroma_db.add_texts(texts=[chunk], metadatas=[{"chunk_id": idx}], ids=[str(idx)])
chroma_db.persist()


#%%
# Function to retrieve the most relevant chunks using ChromaDB
def get_relevant_context(question, top_n=5):
    """
    Retrieve the most relevant context from ChromaDB for a given question.

    Args:
    - question (str): The input question.
    - top_n (int): Number of top relevant chunks to retrieve.

    Returns:
    - combined_context (str): Combined text from the most relevant chunks.
    """
    docs = chroma_db.similarity_search(question, k=top_n)
    combined_context = " ".join([doc.page_content for doc in docs])
    return combined_context


# Function to answer the question using the fine-tuned GPT-2 model
def answer_question(question, max_new_tokens=300):
    """
    Answer a question using the fine-tuned GPT-2 model and LangChain's context retrieval.

    Args:
    - question (str): The input question.
    - max_new_tokens (int): Maximum number of tokens for the generated response.

    Returns:
    - answer (str): The model's answer to the question.
    """
    relevant_context = get_relevant_context(question)
    prompt = f"Context: {relevant_context}\n\nQuestion: {question}\nAnswer:"
    print("Context is",relevant_context)
    response = fine_tuned_generator(
        prompt,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        pad_token_id=fine_tuned_tokenizer.eos_token_id,
        truncation=True
    )
    return response[0]['generated_text'].split('Answer:')[-1].strip()

#%%
# Initialize Chroma with the wrapper
vectorstore_dir = "/home/ubuntu/sairam/Project/chromadb_store"

chroma_db = Chroma(
    persist_directory=vectorstore_dir,
    embedding_function=embedding_function
)
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

