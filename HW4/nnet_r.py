import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
 
# Constants
GLOVE_PATH = "glove.6B.100d.txt"
EMBED_DIM = 100
CATEGORIES = ["comp.graphics", "rec.sport.baseball", "sci.med", "alt.atheism", "talk.politics.guns"]
 
# Model definition
class SimpleGloveModel(nn.Module):
    def __init__(self, embedding_matrix: Tensor, num_classes: int, freeze_embedding: bool = True) -> None:
        super(SimpleGloveModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_embedding)
        self.fc1 = nn.Linear(embedding_matrix.size(1), 256)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
 
    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = x.mean(dim=1)  # Average across the sequence dimension
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
 
# Utility functions
def load_data():
    """Load the 20 Newsgroups dataset"""
    train_data = fetch_20newsgroups(subset='train', categories=CATEGORIES, remove=('headers', 'footers', 'quotes'))
    test_data = fetch_20newsgroups(subset='test', categories=CATEGORIES, remove=('headers', 'footers', 'quotes'))
    return train_data, test_data
 
def load_glove(text_data):
    """Load GloVe embeddings for words in the dataset"""
    # Get all unique words from the dataset
    all_words = set()
    for doc in text_data:
        words = simple_preprocess(doc)
        all_words.update(words)
   
    print(f"Total unique words in corpus: {len(all_words)}")
   
    # Load embeddings for words in our dataset
    word_to_idx = {"<PAD>": 0, "<UNK>": 1}
    embeddings = [
        np.zeros(EMBED_DIM),  # PAD
        np.random.normal(scale=0.1, size=EMBED_DIM)  # UNK
    ]
   
    idx = 2
    found_words = 0
   
    with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split(' ')
            word = values[0]
           
            if word in all_words:
                vector = np.array(values[1:], dtype=np.float32)
                embeddings.append(vector)
                word_to_idx[word] = idx
                idx += 1
                found_words += 1
   
    print(f"Words found in GloVe: {found_words}")
    print(f"OOV ratio: {1 - (found_words / len(all_words)):.4f}")
   
    # Convert to tensor
    embeddings_matrix = torch.tensor(np.array(embeddings), dtype=torch.float)
    return embeddings_matrix, word_to_idx
 
def preprocess_data(docs, word_to_idx, max_length=None):
    """Convert documents to sequences of word indices and pad them"""
    sequences = []
    for doc in docs:
        words = simple_preprocess(doc)
        seq = [word_to_idx.get(word, 1) for word in words]  # Use 1 for unknown words
        sequences.append(seq)
   
    # Find max length if not provided
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
   
    # Pad sequences
    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_length:
            padded_sequences.append(seq[:max_length])
        else:
            padded_sequences.append(seq + [0] * (max_length - len(seq)))
   
    return torch.tensor(padded_sequences, dtype=torch.long), max_length
 
def train_model(model, train_loader, val_loader, epochs, device, is_frozen=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
   
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
   
    print(f"Training model with {'frozen' if is_frozen else 'fine-tuned'} embeddings")
   
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
       
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
           
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
           
            train_loss += loss.item()
       
        train_loss /= len(train_loader)
       
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
       
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
               
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
       
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
       
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
       
        # LR scheduling
        scheduler.step(val_loss)
       
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
           
            # Save the best model
            model_path = f"best_model_{'frozen' if is_frozen else 'finetuned'}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
   
    return best_val_acc
 
 
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
   
    # Load data
    train_data, test_data = load_data()
    all_docs = train_data.data + test_data.data
   
    # Load GloVe embeddings
    embedding_matrix, word_to_idx = load_glove(all_docs)
   
    # Preprocess train data
    X_train_padded, max_length = preprocess_data(train_data.data, word_to_idx)
    y_train = torch.tensor(train_data.target, dtype=torch.long)
   
    # Preprocess test data
    X_test_padded, _ = preprocess_data(test_data.data, word_to_idx, max_length)
    y_test = torch.tensor(test_data.target, dtype=torch.long)
   
    # Split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_padded, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
   
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test_padded, y_test)
   
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
   
    # Model with frozen embeddings
    num_classes = len(CATEGORIES)
    frozen_model = SimpleGloveModel(embedding_matrix, num_classes, freeze_embedding=True).to(device)
   
    # Train with frozen embeddings
    best_frozen_acc = train_model(frozen_model, train_loader, val_loader, epochs=15, device=device, is_frozen=True)
   
    # Evaluate on test set
    frozen_model.load_state_dict(torch.load("best_model_frozen.pt"))
    frozen_model.eval()
   
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = frozen_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
   
    frozen_test_acc = 100 * correct / total
    print(f"Frozen model test accuracy: {frozen_test_acc:.2f}%")
   
    # Visualize embeddings for frozen model
    # visualize_embeddings(frozen_model, test_loader, device, is_frozen=True)
   
    # Model with fine-tuned embeddings
    finetuned_model = SimpleGloveModel(embedding_matrix, num_classes, freeze_embedding=False).to(device)
   
    # Train with fine-tuned embeddings
    best_finetuned_acc = train_model(finetuned_model, train_loader, val_loader, epochs=35, device=device, is_frozen=False)
   
    # Evaluate on test set
    finetuned_model.load_state_dict(torch.load("best_model_finetuned.pt"))
    finetuned_model.eval()
   
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = finetuned_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
   
    finetuned_test_acc = 100 * correct / total
    print(f"Fine-tuned model test accuracy: {finetuned_test_acc:.2f}%")
   
    # Visualize embeddings for fine-tuned model
    # visualize_embeddings(finetuned_model, test_loader, device, is_frozen=False)
   
    # Compare results
    print("\n" + "="*40 + " RESULTS COMPARISON " + "="*40)
    print(f"Frozen Embeddings Accuracy: {frozen_test_acc:.2f}%")
    print(f"Fine-tuned Embeddings Accuracy: {finetuned_test_acc:.2f}%")
 
if __name__ == "__main__":
    main()