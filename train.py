import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from model import SnoreNet
from dataset import KaggleSnoreDataset
import os

# Training settings
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001

# Dataset split
full_dataset = KaggleSnoreDataset(root_dir="snoring_dataset")
train_len = int(0.8 * len(full_dataset))
val_len = len(full_dataset) - train_len
train_set, val_set = random_split(full_dataset, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)

model = SnoreNet()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x_batch, y_batch = zip(*batch)
        x_batch = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True)
        y_batch = torch.nn.utils.rnn.pad_sequence(y_batch, batch_first=True)

        logits = model(x_batch).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {total_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x_batch, y_batch = zip(*batch)
            x_batch = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True)
            y_batch = torch.nn.utils.rnn.pad_sequence(y_batch, batch_first=True)

            logits = model(x_batch).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_batch)
            val_loss += loss.item()
    print(f"           Val Loss: {val_loss / len(val_loader):.4f}")

# Save model
os.makedirs("./checkpoints", exist_ok=True)
torch.save(model.state_dict(), "./checkpoints/snore_net.pth")
