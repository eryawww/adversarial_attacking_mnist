from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os

class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class FCNet(nn.Module):
    def __init__(self, hidden_sizes=[784, 512, 256, 128]):
        super(FCNet, self).__init__()
        self.flatten = nn.Flatten()
        layers = []
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(hidden_sizes[-1], 10))
        self.fc_layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc_layers(x)
        return F.log_softmax(x, dim=1)

def __train_epoch(model, device, data, target, optimizer):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def __train_model(model, device, train_loader, optimizer, epochs, test_loader):
    model.train()
    
    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        __train_epoch(model, device, train_loader, optimizer, epoch)
        accuracy = __test_model(model, device, test_loader)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, 'models/base')

def __test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def load_mnist_data(batch_size=64):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
        batch_size=batch_size, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader

def save_model(model, path='models/base'):
    if isinstance(model, CnnNet):
        path += '_cnn.pth'
        torch.save(model.state_dict(), path)
    elif isinstance(model, FCNet):
        path += '_fc.pth'
        torch.save(model.state_dict(), path)

def load_pretrained_model(path='models/base_fc.pth'):
    if os.path.exists(path):
        if '_cnn' in path:
            model = CnnNet()
        elif '_fc' in path:
            model = FCNet()
        else:
            raise ValueError(f"Invalid path, should be *_fc* or *_cnn*")
        
        model.load_state_dict(torch.load(path))
        return model
    else:
        # Create and train model
        if '_cnn' in path:
            model = CnnNet().to(device)
        elif '_fc' in path:
            model = FCNet().to(device)
        else:
            raise ValueError(f"Invalid path, should be *_fc* or *_cnn*")

        train_loader, test_loader = load_mnist_data()   
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, epochs + 1):
            __train_model(model, device, train_loader, optimizer, epoch)
        print(f'Model saved at {path}')
        # Save model
        save_model(model, path)
        return model

def train_mnist(model_type: Literal['fc', 'cnn'], model_path: str):
    # Training settings
    batch_size = 64
    epochs = 10
    lr = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model and optimizer
    if model_type == 'fc':
        model = FCNet().to(device)
    elif model_type == 'cnn':
        model = CnnNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader, test_loader = load_mnist_data(batch_size)

    __train_model(model, device, train_loader, optimizer, epochs, test_loader)
    save_model(model, model_path)

def main():
    train_mnist('fc', 'models/base_fc.pth')
    train_mnist('cnn', 'models/base_cnn.pth')
    
if __name__ == "__main__":
    main()
