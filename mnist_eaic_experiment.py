import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
num_epochs = 100
batch_size = 128
learning_rate = 0.001
threshold_r = 0.1  # Threshold scaling factor for EAIC
regularization_lambda = 0.001  # Regularization parameter for L2 regularization (weight decay)

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transform,
                               download=True)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Model 1: Simple Neural Network (SNN)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

# Model 2: Deep Neural Network without Regularization (DNN)
class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

# Model 3: Deep Neural Network with L2 Regularization (DNN-Reg)
class DeepNNReg(nn.Module):
    def __init__(self):
        super(DeepNNReg, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

# Function to train the model
def train_model(model, train_loader, optimizer, criterion, num_epochs):
    model.train()
    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Print loss every epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/total_steps:.4f}')
    return model

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        avg_loss = total_loss / len(test_loader)
        accuracy = 100 * correct / total
    return accuracy, avg_loss

# Function to calculate AIC and EAIC
def calculate_aic_eaic(model, train_loader, threshold_r):
    model.eval()
    total_loss = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')  # Sum over all samples
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_samples += labels.size(0)
    
    nll = total_loss  # Negative log-likelihood
    k = sum(p.numel() for p in model.parameters())  # Total number of parameters
    
    # Get all parameters as a flat vector
    params = torch.cat([p.view(-1) for p in model.parameters()])
    sigma = params.std().item()
    h = threshold_r * sigma
    
    # Effective number of parameters
    k_eff = torch.sum(torch.abs(params) > h).item()
    
    AIC = 2 * k + 2 * nll
    EAIC = 2 * k_eff + 2 * nll
    
    return AIC, EAIC, k, k_eff, nll
# Define a function to perform the full pipeline for one iteration
def run_experiment():
    # Initialize models
    model_snn = SimpleNN().to(device)
    model_dnn = DeepNN().to(device)
    model_dnn_reg = DeepNNReg().to(device)

    # Define optimizers
    optimizer_snn = optim.SGD(model_snn.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_dnn = optim.SGD(model_dnn.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_dnn_reg = optim.SGD(model_dnn_reg.parameters(), lr=learning_rate, momentum=0.9, weight_decay=regularization_lambda)

    # Train models
    train_model(model_snn, train_loader, optimizer_snn, criterion, num_epochs)
    train_model(model_dnn, train_loader, optimizer_dnn, criterion, num_epochs)
    train_model(model_dnn_reg, train_loader, optimizer_dnn_reg, criterion, num_epochs)

    # Evaluate models
    accuracy_snn, _ = evaluate_model(model_snn, test_loader)
    accuracy_dnn, _ = evaluate_model(model_dnn, test_loader)
    accuracy_dnn_reg, _ = evaluate_model(model_dnn_reg, test_loader)

    # Calculate AIC and EAIC
    AIC_snn, EAIC_snn, _, _, _ = calculate_aic_eaic(model_snn, train_loader, threshold_r)
    AIC_dnn, EAIC_dnn, _, _, _ = calculate_aic_eaic(model_dnn, train_loader, threshold_r)
    AIC_dnn_reg, EAIC_dnn_reg, _, _, _ = calculate_aic_eaic(model_dnn_reg, train_loader, threshold_r)

    # Correlation analysis
    aic_values = np.array([AIC_snn, AIC_dnn, AIC_dnn_reg])
    eaic_values = np.array([EAIC_snn, EAIC_dnn, EAIC_dnn_reg])
    test_accuracies = np.array([accuracy_snn, accuracy_dnn, accuracy_dnn_reg])

    corr_aic = np.corrcoef(aic_values, test_accuracies)[0, 1]
    corr_eaic = np.corrcoef(eaic_values, test_accuracies)[0, 1]

    return corr_aic, corr_eaic

# Run the experiment 100 times and store the correlations
correlations_aic = []
correlations_eaic = []

num_iterations = 100
for _ in range(num_iterations):
    corr_aic, corr_eaic = run_experiment()
    correlations_aic.append(corr_aic)
    correlations_eaic.append(corr_eaic)

# Calculate average correlations
average_corr_aic = np.mean(correlations_aic)
average_corr_eaic = np.mean(correlations_eaic)

print(f'\n===== Final Results After {num_iterations} Iterations =====')
print(f'Average Correlation between AIC and Test Accuracy: {average_corr_aic*100:.2f}%')
print(f'Average Correlation between EAIC and Test Accuracy: {average_corr_eaic*100:.2f}%')

