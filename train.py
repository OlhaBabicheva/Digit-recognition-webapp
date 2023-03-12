import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

# Parameters
in_dim = 1
num_classes = 10
l_rate = 1e-3
batch_size = 64
num_epochs = 7

# Load MNIST dataset
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor(), download = True)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

model = Net(in_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = optim.Adam(model.parameters(), lr = l_rate) # optimizer

def train(dataloader, model):
    size = len(dataloader.dataset)
    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}\n-------------------------------')
        losses = []
        accuracies = []
        for batch, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)
            # Compute predictions and loss
            pred = model(X)
            loss = criterion(pred, y)
            losses.append(loss.item()) 
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Running training accuracy
            _, prediction = pred.max(1)
            n_correct = (prediction == y).sum()
            running_training_acc = n_correct/X.shape[0]
            accuracies.append(running_training_acc)
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    torch.save(model.state_dict(), 'model.pt')

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Accuracy on training data")
    else:
        print("Accuracy on test data")
    model.eval()
    n_correct = 0
    n_samples = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            _, prediction = pred.max(1) # idx of max value for the second dim
            n_correct += (prediction == y).sum()
            n_samples += prediction.size(0)
        print(f'Accuracy: {n_correct/n_samples * 100:.3f}%')

if __name__ == '__main__':
    train(train_loader, model)
    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)