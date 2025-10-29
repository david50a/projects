import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from networkx.classes.filters import hide_nodes
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


dataset = pd.read_csv('fraud_email_.csv')
print(dataset.dtypes)
print(dataset.describe())

words = dataset['Text'].tolist()
words = [str(sentence).split() for sentence in words]

# build vocabulary
unique_words = {word for sent in words for word in sent}
word2idx = {word: idx + 1 for idx, word in enumerate(unique_words)}
vocab_size = len(word2idx) + 1

def pad_sequences(sequences, padding_value=0):
    max_len = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        padded.append(seq + [padding_value] * (max_len - len(seq)))
    return torch.tensor(padded, dtype=torch.long)

def create_sequences(data, tokens):
    texts, labels = [], []
    for sent, label in data:
        token_ids = [tokens[word] for word in sent if word in tokens]
        texts.append(token_ids)
        labels.append(label)
    return pad_sequences(texts), torch.tensor(labels, dtype=torch.long)

dataset_pairs = [(sentence, label) for sentence, label in zip(words, dataset['Class'].values)]
texts, classes = create_sequences(dataset_pairs, word2idx)
print(texts)
print(classes)
# split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    texts, classes, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        rnn_out, hidden = self.rnn(x)
        output = self.h2o(hidden[0])
        return self.softmax(output)

embedding_dim = 50
hidden_size = 128
output_size = dataset['Class'].nunique()

model = RNN(vocab_size, embedding_dim, hidden_size, output_size)
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def train(model, train_loader, n_epochs=10, report_every=1):
    all_losses = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0
        for batch_text, batch_labels in train_loader:
            optimizer.zero_grad()
            output = model(batch_text)
            loss = criterion(output, batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        all_losses.append(avg_loss)
        if epoch % report_every == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    return all_losses


def evaluate(model, test_loader, classes=[0,1]):
    confusion = torch.zeros(len(classes), len(classes))
    model.eval()
    with torch.no_grad():
        for batch_text, batch_labels in test_loader:
            output = model(batch_text)
            preds = torch.argmax(output, dim=1)
            for t, p in zip(batch_labels.view(-1), preds.view(-1)):
                confusion[t.long(), p.long()] += 1
    # normalize
    for i in range(len(classes)):
        if confusion[i].sum() > 0:
            confusion[i] /= confusion[i].sum()

    # plot confusion matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.cpu().numpy())
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.colorbar(cax)
    plt.show()
    acc = (confusion.diag().sum() / confusion.sum()).item()
    print(f"Test Accuracy: {acc*100:.2f}%")
def predict(model,text,tokens,max_len=50):
    words=text.split()
    token_ids=[tokens[w] for w in words if w in tokens]
    if len(token_ids)<max_len:
        token_ids+=[0]*(max_len-len(token_ids))
    else:
        token_ids=token_ids[:max_len]
    input_tensor=torch.tensor([token_ids],dtype=torch.long)
    with torch.no_grad():
        output=model(input_tensor)
        predicted_class=torch.argmax(output,dim=1).item()
    return predicted_class

start = time.time()
all_losses = train(model, train_loader, n_epochs=10, report_every=1)
end = time.time()
print(f"Training took {end-start:.2f} seconds")

plt.figure()
plt.plot(all_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

evaluate(model, test_loader, classes=[0,1])
torch.save(model.state_dict(),'fraud_model.pth')
torch.save(model,'full_fraud_detection_model.pth')
rnn=RNN(vocab_size,embedding_dim,hidden_size,output_size)
rnn.load_state_dict(torch.load('fraud_model.pth'))
rnn.eval()
model=torch.load('full_fraud_detection_model.pth')
model.eval()
test_text = "Congratulations! You have won a free lottery prize"
prediction = predict(model, test_text, word2idx)
print("Prediction:", prediction)
