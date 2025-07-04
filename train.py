import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from utils.download_dataset import download_dataset
from model import vit

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, loader, optimizer, criterion, epoch):
    # Set the mode of the model into training
    model.train()

    total_loss, correct = 0, 0

    for x, y in tqdm(loader, desc=f"Training Progress (Epoch {epoch+1}/{vit.ViTConfig.epochs})"):
        # Moving (Sending) our data into the target device
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # 1. Forward pass (model outputs raw logits)
        out = model(x)
        # 2. Calcualte loss (per batch)
        loss = criterion(out, y)
        # 3. Perform backpropgation
        loss.backward()
        # 4. Perforam Gradient Descent
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
    # You have to scale the loss (Normlization step to make the loss general across all batches)
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate(model, loader):
    model.eval() # Set the mode of the model into evlauation
    correct = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(dim=1) == y).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__':
    # Download and prepare the dataset
    train_dataloader, test_dataloader = download_dataset(root='./data', isDownload=True, batch_size=vit.ViTConfig.batch_size)
    
    # Initialize the ViT
    model = vit.VisionTransformer().to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=vit.ViTConfig.learning_rate)

    # Training
    train_accuracies, test_accuracies = [], [] 
    for epoch in range(vit.ViTConfig.epochs):
        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, epoch)
        test_acc = evaluate(model, test_dataloader)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        print(f"Epoch: {epoch+1}/{vit.ViTConfig.epochs}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}%, Test acc: {test_acc:.4f}")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f'./checkpoints/vit_epoch_{vit.ViTConfig.epochs}.pth')

    # Plot accuracy
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Test Accuracy")
    plt.show()