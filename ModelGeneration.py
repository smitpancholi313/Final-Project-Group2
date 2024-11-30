import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_excel("speeches.xlsx")

def preprocess_text(text):
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

df['cleaned_speech'] = df['transcript'].apply(preprocess_text)


from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

df['sentiment'] = df['cleaned_speech'].apply(lambda x: TextBlob(x).sentiment.polarity)

vectorizer = CountVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(df['cleaned_speech'])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

topics = lda.components_
for idx, topic in enumerate(topics):
    print(f"Topic {idx + 1}: {[vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]}")


import torch
from torch.utils.data import DataLoader
from transformers import AdamW, GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token

train_encodings = tokenizer(
    list(df['cleaned_speech']), 
    truncation=True, 
    padding=True, 
    max_length=512, 
    return_tensors="pt"
)

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

train_dataset = SpeechDataset(train_encodings)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

epochs = 1
model.train()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        optimizer.step()

        print(f"Batch loss: {loss.item():.4f}")

model.save_pretrained('./fine_tuned_gpt2')
tokenizer.save_pretrained('./fine_tuned_gpt2')

