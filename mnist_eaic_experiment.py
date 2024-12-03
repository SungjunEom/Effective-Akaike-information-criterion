import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
num_epochs = 10
batch_size = 1024  # Increased batch size
learning_rate = 0.001
threshold_r = 0.1  # Threshold scaling factor for EAIC
regularization_lambda = 0.001  # Regularization parameter for L2 regularization (weight decay)

# Number of runs
num_runs = 100

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

# Adjusted DataLoader with num_workers and pin_memory
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4,    # Number of subprocesses for data loading
                                           pin_memory=True)  # Speeds up the transfer to GPU

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=4,
                                          pin_memory=True)

# Model 1: Larger Simple Neural Network (SNN)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 10)
            
    def forward(self, x):
        x = x.view(-1, 28*28)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

# Model 2: Larger Deep Neural Network without Regularization (DNN)
class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 8192)
        self.fc2 = nn.Linear(8192, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 64)
        self.fc9 = nn.Linear(64, 10)
            
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x

# Model 3: Larger Deep Neural Network with L2 Regularization (DNN-Reg)
class DeepNNReg(nn.Module):
    def __init__(self):
        super(DeepNNReg, self).__init__()
        self.fc1 = nn.Linear(28*28, 8192)
        self.fc2 = nn.Linear(8192, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 64)
        self.fc9 = nn.Linear(64, 10)
            
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x

# Function to train the model
def train_model(model, train_loader, optimizer, criterion, num_epochs):
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    end_time = time.time()
    print(f'Training Time: {end_time - start_time:.2f} seconds')
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
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        avg_loss = total_loss / total
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
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
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

# Lists to store results
aic_snn_list = []
eaic_snn_list = []
accuracy_snn_list = []

aic_dnn_list = []
eaic_dnn_list = []
accuracy_dnn_list = []

aic_dnn_reg_list = []
eaic_dnn_reg_list = []
accuracy_dnn_reg_list = []

print(f'Running {num_runs} runs for each model...\n')

for run in range(num_runs):
    print(f'Run {run+1}/{num_runs}')
    
    # Initialize models
    model_snn = SimpleNN().to(device)
    model_dnn = DeepNN().to(device)
    model_dnn_reg = DeepNNReg().to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    optimizer_snn = optim.Adam(model_snn.parameters(), lr=learning_rate)
    optimizer_dnn = optim.Adam(model_dnn.parameters(), lr=learning_rate)
    optimizer_dnn_reg = optim.Adam(model_dnn_reg.parameters(), lr=learning_rate, weight_decay=regularization_lambda)
    
    # Train models
    model_snn = train_model(model_snn, train_loader, optimizer_snn, criterion, num_epochs)
    model_dnn = train_model(model_dnn, train_loader, optimizer_dnn, criterion, num_epochs)
    model_dnn_reg = train_model(model_dnn_reg, train_loader, optimizer_dnn_reg, criterion, num_epochs)
    
    # Evaluate models
    accuracy_snn, test_loss_snn = evaluate_model(model_snn, test_loader)
    accuracy_dnn, test_loss_dnn = evaluate_model(model_dnn, test_loader)
    accuracy_dnn_reg, test_loss_dnn_reg = evaluate_model(model_dnn_reg, test_loader)
    
    # Calculate AIC and EAIC
    AIC_snn, EAIC_snn, k_snn, k_eff_snn, nll_snn = calculate_aic_eaic(model_snn, train_loader, threshold_r)
    AIC_dnn, EAIC_dnn, k_dnn, k_eff_dnn, nll_dnn = calculate_aic_eaic(model_dnn, train_loader, threshold_r)
    AIC_dnn_reg, EAIC_dnn_reg, k_dnn_reg, k_eff_dnn_reg, nll_dnn_reg = calculate_aic_eaic(model_dnn_reg, train_loader, threshold_r)
    
    # Store results
    aic_snn_list.append(AIC_snn)
    eaic_snn_list.append(EAIC_snn)
    accuracy_snn_list.append(accuracy_snn)
    
    aic_dnn_list.append(AIC_dnn)
    eaic_dnn_list.append(EAIC_dnn)
    accuracy_dnn_list.append(accuracy_dnn)
    
    aic_dnn_reg_list.append(AIC_dnn_reg)
    eaic_dnn_reg_list.append(EAIC_dnn_reg)
    accuracy_dnn_reg_list.append(accuracy_dnn_reg)

# Convert lists to numpy arrays
aic_snn_array = np.array(aic_snn_list)
eaic_snn_array = np.array(eaic_snn_list)
accuracy_snn_array = np.array(accuracy_snn_list)

aic_dnn_array = np.array(aic_dnn_list)
eaic_dnn_array = np.array(eaic_dnn_list)
accuracy_dnn_array = np.array(accuracy_dnn_list)

aic_dnn_reg_array = np.array(aic_dnn_reg_list)
eaic_dnn_reg_array = np.array(eaic_dnn_reg_list)
accuracy_dnn_reg_array = np.array(accuracy_dnn_reg_list)

# Correlation Analysis for SNN
corr_aic_snn = np.corrcoef(aic_snn_array, accuracy_snn_array)[0,1]
corr_eaic_snn = np.corrcoef(eaic_snn_array, accuracy_snn_array)[0,1]

# Correlation Analysis for DNN
corr_aic_dnn = np.corrcoef(aic_dnn_array, accuracy_dnn_array)[0,1]
corr_eaic_dnn = np.corrcoef(eaic_dnn_array, accuracy_dnn_array)[0,1]

# Correlation Analysis for DNN-Reg
corr_aic_dnn_reg = np.corrcoef(aic_dnn_reg_array, accuracy_dnn_reg_array)[0,1]
corr_eaic_dnn_reg = np.corrcoef(eaic_dnn_reg_array, accuracy_dnn_reg_array)[0,1]

# Print Correlation Results
print('\n===== Correlation Analysis =====')
print('Simple Neural Network (SNN):')
print(f'Correlation between AIC and Test Accuracy: {corr_aic_snn*100:.2f}%')
print(f'Correlation between EAIC and Test Accuracy: {corr_eaic_snn*100:.2f}%')

print('\nDeep Neural Network (DNN):')
print(f'Correlation between AIC and Test Accuracy: {corr_aic_dnn*100:.2f}%')
print(f'Correlation between EAIC and Test Accuracy: {corr_eaic_dnn*100:.2f}%')

print('\nDeep Neural Network with L2 Regularization (DNN-Reg):')
print(f'Correlation between AIC and Test Accuracy: {corr_aic_dnn_reg*100:.2f}%')
print(f'Correlation between EAIC and Test Accuracy: {corr_eaic_dnn_reg*100:.2f}%')
