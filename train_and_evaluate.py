"""
Assignment 7: Sentiment Analysis on IMDB Dataset using Traditional ML and Recurrent Neural Networks (RNN & LSTM)
Author: NLP Course - Week 7
Models compared:
  1. Logistic Regression (TF-IDF)
  2. Support Vector Machine - LinearSVC (TF-IDF)
  3. Basic Neural Network (MLP / Global Average Pooling)
  4. Recurrent Neural Network (Simple RNN)
  5. Long Short-Term Memory (LSTM)
  6. Gated Recurrent Unit (GRU)
"""

import os
import re
import time
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Using device: {DEVICE}")

# -------------------------------------------------------------
# 1. DATA LOADING AND PREPROCESSING
# -------------------------------------------------------------
def clean_text(text: str) -> str:
    """Removes HTML tags, punctuation, and converts to lowercase."""
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str):
    """Simple whitespace-based word tokenization."""
    return text.split()

def load_imdb_data(data_dir: str):
    train_path = os.path.join(data_dir, "train.parquet")
    test_path = os.path.join(data_dir, "test.parquet")
    
    print("[*] Loading datasets...")
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    
    print(f"    Train size: {len(df_train)} (Positive: {(df_train['label']==1).sum()}, Negative: {(df_train['label']==0).sum()})")
    print(f"    Test size:  {len(df_test)} (Positive: {(df_test['label']==1).sum()}, Negative: {(df_test['label']==0).sum()})")
    
    print("[*] Cleaning text data...")
    df_train["clean_text"] = df_train["text"].apply(clean_text)
    df_test["clean_text"] = df_test["text"].apply(clean_text)
    
    return df_train, df_test

# -------------------------------------------------------------
# 2. VOCABULARY & PYTORCH DATASET
# -------------------------------------------------------------
class Vocabulary:
    def __init__(self, max_size=25000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_idx = 0
        self.unk_idx = 1
        
        self.word2idx = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}
        self.idx2word = {self.pad_idx: self.pad_token, self.unk_idx: self.unk_token}
        
    def build_vocab(self, texts):
        print("[*] Building vocabulary...")
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))
            
        most_common = counter.most_common(self.max_size - 2)
        idx = 2
        for word, count in most_common:
            if count >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        print(f"    Total vocabulary size: {len(self.word2idx)}")
        
    def encode(self, text, max_len=256):
        tokens = tokenize(text)
        indices = [self.word2idx.get(w, self.unk_idx) for w in tokens[:max_len]]
        if len(indices) < max_len:
            indices += [self.pad_idx] * (max_len - len(indices))
        return indices

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocabulary, max_len=256):
        self.labels = labels
        self.encoded_texts = [vocab.encode(t, max_len=max_len) for t in texts]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

# -------------------------------------------------------------
# 3. PYTORCH MODEL ARCHITECTURES
# -------------------------------------------------------------
class BasicMLP(nn.Module):
    """Basic Feedforward Neural Network with Embedding and Global Average Pooling."""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        embeds = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        # Mask out padding tokens for accurate average pooling
        mask = (x != 0).unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        out = self.fc1(pooled)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out).squeeze(-1)
        return logits

class RNNModel(nn.Module):
    """Bidirectional Recurrent Neural Network (RNN)."""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            nonlinearity='tanh'
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        embeds = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        out, h_n = self.rnn(embeds) # out: [batch_size, seq_len, hidden_dim * 2]
        # h_n: [num_layers * 2, batch_size, hidden_dim]
        last_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) # [batch_size, hidden_dim * 2]
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden).squeeze(-1)
        return logits

class LSTMModel(nn.Module):
    """Bidirectional Long Short-Term Memory (LSTM) Network."""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        embeds = self.embedding(x)
        out, (h_n, c_n) = self.lstm(embeds)
        last_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden).squeeze(-1)
        return logits

class GRUModel(nn.Module):
    """Bidirectional Gated Recurrent Unit (GRU) Network."""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        embeds = self.embedding(x)
        out, h_n = self.gru(embeds)
        last_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden).squeeze(-1)
        return logits

# -------------------------------------------------------------
# 4. TRAINING & EVALUATION FUNCTIONS FOR PYTORCH
# -------------------------------------------------------------
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)
        
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        total_loss += loss.item() * batch_x.size(0)
        
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(batch_y.cpu().numpy().astype(int))
        total += batch_x.size(0)
        
    avg_loss = total_loss / total
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "preds": all_preds,
        "probs": all_probs,
        "labels": all_labels
    }
    return metrics

def train_and_track(model_name, model, train_loader, test_loader, epochs=5, lr=1e-3):
    print(f"\n---> Training {model_name} on {DEVICE}...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate_model(model, test_loader, criterion)
        
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        
        print(f"    Epoch {epoch}/{epochs} | Train Loss: {tr_loss:.4f} - Acc: {tr_acc*100:.2f}% | Val Loss: {val_metrics['loss']:.4f} - Val Acc: {val_metrics['accuracy']*100:.2f}%")
        
    elapsed = time.time() - start_time
    final_metrics = evaluate_model(model, test_loader, criterion)
    final_metrics["train_time"] = elapsed
    final_metrics["history"] = history
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    final_metrics["num_params"] = num_params
    
    print(f"[*] Completed {model_name} in {elapsed:.1f}s | Final Test Acc: {final_metrics['accuracy']*100:.2f}% | F1: {final_metrics['f1']:.4f}")
    return final_metrics

# -------------------------------------------------------------
# 5. MAIN EXECUTION PIPELINE
# -------------------------------------------------------------
def run_all_experiments():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    # 1. Load Data
    df_train, df_test = load_imdb_data(data_dir)
    y_train = df_train["label"].values
    y_test = df_test["label"].values
    
    results = {}
    
    # 2. Traditional ML Baselines: TF-IDF + Logistic Regression & LinearSVC
    print("\n" + "="*60)
    print("PART 1: TRADITIONAL MACHINE LEARNING BASELINES")
    print("="*60)
    
    print("[*] Vectorizing with TF-IDF (unigrams + bigrams, max_features=25000)...")
    tfidf = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(df_train["clean_text"])
    X_test_tfidf = tfidf.transform(df_test["clean_text"])
    
    # 2.1 Logistic Regression
    print("\n---> Training Logistic Regression...")
    t0 = time.time()
    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)
    lr_time = time.time() - t0
    
    lr_preds = lr_model.predict(X_test_tfidf)
    lr_probs = lr_model.predict_proba(X_test_tfidf)[:, 1]
    
    results["Logistic Regression"] = {
        "accuracy": accuracy_score(y_test, lr_preds),
        "precision": precision_score(y_test, lr_preds),
        "recall": recall_score(y_test, lr_preds),
        "f1": f1_score(y_test, lr_preds),
        "roc_auc": roc_auc_score(y_test, lr_probs),
        "confusion_matrix": confusion_matrix(y_test, lr_preds),
        "train_time": lr_time,
        "num_params": X_train_tfidf.shape[1] + 1
    }
    print(f"[*] Logistic Regression Test Acc: {results['Logistic Regression']['accuracy']*100:.2f}% | F1: {results['Logistic Regression']['f1']:.4f} | Time: {lr_time:.1f}s")
    
    # 2.2 Support Vector Machine (LinearSVC)
    print("\n---> Training SVM (LinearSVC)...")
    t0 = time.time()
    svm_model = LinearSVC(C=0.5, max_iter=2000, random_state=42)
    svm_model.fit(X_train_tfidf, y_train)
    svm_time = time.time() - t0
    
    svm_preds = svm_model.predict(X_test_tfidf)
    svm_scores = svm_model.decision_function(X_test_tfidf)
    
    results["SVM (LinearSVC)"] = {
        "accuracy": accuracy_score(y_test, svm_preds),
        "precision": precision_score(y_test, svm_preds),
        "recall": recall_score(y_test, svm_preds),
        "f1": f1_score(y_test, svm_preds),
        "roc_auc": roc_auc_score(y_test, svm_scores),
        "confusion_matrix": confusion_matrix(y_test, svm_preds),
        "train_time": svm_time,
        "num_params": X_train_tfidf.shape[1] + 1
    }
    print(f"[*] SVM (LinearSVC) Test Acc: {results['SVM (LinearSVC)']['accuracy']*100:.2f}% | F1: {results['SVM (LinearSVC)']['f1']:.4f} | Time: {svm_time:.1f}s")
    
    # 3. Deep Learning Setup (Vocabulary & PyTorch DataLoaders)
    print("\n" + "="*60)
    print("PART 2: NEURAL NETWORKS (MLP, RNN, LSTM, GRU)")
    print("="*60)
    
    vocab = Vocabulary(max_size=25000, min_freq=2)
    vocab.build_vocab(df_train["clean_text"])
    
    max_len = 256
    batch_size = 64
    
    train_dataset = IMDBDataset(df_train["clean_text"], y_train, vocab, max_len=max_len)
    test_dataset = IMDBDataset(df_test["clean_text"], y_test, vocab, max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(DEVICE.type=="cuda"))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=(DEVICE.type=="cuda"))
    
    vocab_size = len(vocab.word2idx)
    epochs = 5
    
    # 3.1 Basic Neural Network (MLP)
    mlp_model = BasicMLP(vocab_size=vocab_size, embed_dim=128, hidden_dim=128, dropout=0.3).to(DEVICE)
    results["Basic Neural Network (MLP)"] = train_and_track("Basic MLP", mlp_model, train_loader, test_loader, epochs=epochs, lr=1e-3)
    
    # 3.2 Recurrent Neural Network (RNN)
    rnn_model = RNNModel(vocab_size=vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.3).to(DEVICE)
    results["Simple RNN"] = train_and_track("Simple RNN", rnn_model, train_loader, test_loader, epochs=epochs, lr=5e-4)
    
    # 3.3 Long Short-Term Memory (LSTM)
    lstm_model = LSTMModel(vocab_size=vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3).to(DEVICE)
    results["LSTM"] = train_and_track("LSTM", lstm_model, train_loader, test_loader, epochs=epochs, lr=1e-3)
    
    # 3.4 Gated Recurrent Unit (GRU)
    gru_model = GRUModel(vocab_size=vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3).to(DEVICE)
    results["GRU"] = train_and_track("GRU", gru_model, train_loader, test_loader, epochs=epochs, lr=1e-3)
    
    # 4. Summary Table and Visualization
    print("\n" + "="*60)
    print("PART 3: COMPREHENSIVE COMPARISON & METRICS SUMMARY")
    print("="*60)
    
    summary_data = []
    for model_name, m in results.items():
        summary_data.append({
            "Model": model_name,
            "Accuracy": round(m["accuracy"] * 100, 2),
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1-Score": round(m["f1"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
            "Train Time (s)": round(m["train_time"], 2),
            "Parameters": m["num_params"]
        })
        
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save CSV
    csv_out = os.path.join(base_dir, "results_summary.csv")
    summary_df.to_csv(csv_out, index=False)
    print(f"\n[*] Saved summary table to {csv_out}")
    
    # 5. Plotting Comparison Charts
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(data=summary_df, x="Accuracy", y="Model", palette="viridis")
    plt.title("Model Accuracy Comparison (%)", fontsize=14, fontweight="bold")
    plt.xlabel("Accuracy (%)", fontsize=12)
    plt.xlim(70, 95)
    for index, value in enumerate(summary_df["Accuracy"]):
        plt.text(value + 0.3, index, f"{value}%", va="center", fontweight="bold")
        
    plt.subplot(1, 2, 2)
    sns.barplot(data=summary_df, x="F1-Score", y="Model", palette="magma")
    plt.title("Model F1-Score Comparison", fontsize=14, fontweight="bold")
    plt.xlabel("F1-Score", fontsize=12)
    plt.xlim(0.7, 0.95)
    for index, value in enumerate(summary_df["F1-Score"]):
        plt.text(value + 0.003, index, f"{value}", va="center", fontweight="bold")
        
    plt.tight_layout()
    plot_file = os.path.join(base_dir, "metrics_comparison.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"[*] Saved metrics comparison bar plot to {plot_file}")
    
    # 6. Plotting Deep Learning Learning Curves
    plt.figure(figsize=(14, 5))
    dl_models = ["Basic Neural Network (MLP)", "Simple RNN", "LSTM", "GRU"]
    
    plt.subplot(1, 2, 1)
    for m_name in dl_models:
        if "history" in results[m_name]:
            plt.plot(range(1, epochs + 1), results[m_name]["history"]["val_loss"], marker="o", label=m_name)
    plt.title("Validation Loss across Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.subplot(1, 2, 2)
    for m_name in dl_models:
        if "history" in results[m_name]:
            plt.plot(range(1, epochs + 1), [acc * 100 for acc in results[m_name]["history"]["val_acc"]], marker="s", label=m_name)
    plt.title("Validation Accuracy across Epochs (%)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    curves_file = os.path.join(base_dir, "training_curves.png")
    plt.savefig(curves_file, dpi=300)
    plt.close()
    print(f"[*] Saved training curves to {curves_file}")
    
    # 7. Plotting Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for idx, (m_name, m) in enumerate(results.items()):
        cm = m["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
                    xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        axes[idx].set_title(f"{m_name}\nAcc: {m['accuracy']*100:.2f}% | F1: {m['f1']:.4f}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
        
    plt.tight_layout()
    cm_file = os.path.join(base_dir, "confusion_matrices.png")
    plt.savefig(cm_file, dpi=300)
    plt.close()
    print(f"[*] Saved confusion matrices to {cm_file}")
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    run_all_experiments()
