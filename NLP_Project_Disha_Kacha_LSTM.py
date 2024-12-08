import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load the data
data = pd.read_excel('speeches.xlsx')
documents = data['transcript'].tolist()

# Define the retrieval function (e.g., TF-IDF)
def retrieve_documents(query, documents, top_n=3):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_n_indices = np.argsort(cosine_similarities)[-top_n:]
    return [documents[i] for i in top_n_indices]

# Define the Seq2Seq Model (simple LSTM-based model)
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # Adding padding_idx for <PAD> token
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_tensor, hidden):
        # Embedding the input sequence
        embedded = self.embedding(input_tensor)
        # Pass the embedded sequence through the LSTM
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded, hidden)
        return encoder_outputs, (hidden, cell)

    def generate(self, input_tensor, max_len=20):
        # Ensure input_tensor is a single token (for simplicity)
        batch_size = input_tensor.size(0)  # Get the batch size dynamically

        # If batch_size is 1 (unbatched input), we need 2-D hidden and cell states (not 3-D)
        if batch_size == 1:
            hidden = (torch.zeros(1, hidden_dim).to(input_tensor.device),  # (num_layers, hidden_dim)
                      torch.zeros(1, hidden_dim).to(input_tensor.device))  # (num_layers, hidden_dim)
        else:
            hidden = (torch.zeros(1, batch_size, hidden_dim).to(input_tensor.device),  # (num_layers, batch_size, hidden_dim)
                      torch.zeros(1, batch_size, hidden_dim).to(input_tensor.device))  # (num_layers, batch_size, hidden_dim)

        # Process the input sequence one token at a time
        predicted_words = []

        # Process the input sequence one token at a time
        for i in range(max_len):
            # Pass the input tensor through the encoder LSTM and get the hidden state
            output, (hidden, cell) = self(input_tensor, hidden)

            # Get the predicted word (highest probability) from the output
            predicted_word = torch.argmax(output[0, 0, :])  # Get the word with highest probability
            predicted_words.append(predicted_word.item())  # Append the predicted word index

            # Update the input tensor with the predicted word for the next step
            input_tensor = predicted_word.unsqueeze(0).unsqueeze(0)  # Add batch dimension for next input

        # Convert word indices to words
        generated_words = [idx2word[idx] for idx in predicted_words]

        # Join the generated words into a sentence
        generated_sentence = " ".join(generated_words)
        return generated_sentence


# Assuming you have the vocabulary mappings:
word2idx = {'What': 0, 'is': 1, 'the': 2, 'president\'s': 3, 'view': 4, 'on': 5, 'unity?': 6, '<UNK>': 7, '<PAD>': 8}
idx2word = {0: 'What', 1: 'is', 2: 'the', 3: 'president\'s', 4: 'view', 5: 'on', 6: 'unity?', 7: '<UNK>', 8: '<PAD>'}

# Define model parameters
vocab_size = len(word2idx)
hidden_dim = 64
embedding_dim = 128

# Initialize the model
model = Seq2Seq(vocab_size, hidden_dim, embedding_dim)

# Example input query (tokenized)
query = "What is the president's view on unity?"

# Convert query to tensor with out-of-vocabulary handling
input_tensor = torch.tensor([word2idx.get(word, word2idx['<UNK>']) for word in query.split()])

# Pad the input tensor to a fixed length (optional but recommended for handling variable lengths)
max_length = 20  # Define a max length for the sequences
input_tensor = torch.cat([input_tensor, torch.tensor([word2idx['<PAD>']] * (max_length - len(input_tensor)))], dim=0)

# Ensure the tensor is a batch (batch size of 1)
input_tensor = input_tensor.unsqueeze(0)

# Generate the answer
answer = model.generate(input_tensor.squeeze(0))  # Squeeze batch dimension before passing to generate
print("Generated Answer:", answer)
