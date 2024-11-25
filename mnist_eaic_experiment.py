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
num_epochs = 50
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

# Initialize models
model_snn = SimpleNN().to(device)
model_dnn = DeepNN().to(device)
model_dnn_reg = DeepNNReg().to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()

# optimizer_snn = optim.Adam(model_snn.parameters(), lr=learning_rate)
# optimizer_dnn = optim.Adam(model_dnn.parameters(), lr=learning_rate)
# optimizer_dnn_reg = optim.Adam(model_dnn_reg.parameters(), lr=learning_rate, weight_decay=regularization_lambda)

optimizer_snn = optim.SGD(model_snn.parameters(), lr=learning_rate, momentum=0.9)
optimizer_dnn = optim.SGD(model_dnn.parameters(), lr=learning_rate, momentum=0.9)
optimizer_dnn_reg = optim.SGD(model_dnn_reg.parameters(), lr=learning_rate, momentum=0.9, weight_decay=regularization_lambda)

# Train models
print('Training Simple Neural Network (SNN)...')
model_snn = train_model(model_snn, train_loader, optimizer_snn, criterion, num_epochs)

print('\nTraining Deep Neural Network without Regularization (DNN)...')
model_dnn = train_model(model_dnn, train_loader, optimizer_dnn, criterion, num_epochs)

print('\nTraining Deep Neural Network with L2 Regularization (DNN-Reg)...')
model_dnn_reg = train_model(model_dnn_reg, train_loader, optimizer_dnn_reg, criterion, num_epochs)

# Evaluate models
print('\nEvaluating Simple Neural Network (SNN)...')
accuracy_snn, test_loss_snn = evaluate_model(model_snn, test_loader)
print(f'Test Accuracy: {accuracy_snn:.2f}%')

print('\nEvaluating Deep Neural Network without Regularization (DNN)...')
accuracy_dnn, test_loss_dnn = evaluate_model(model_dnn, test_loader)
print(f'Test Accuracy: {accuracy_dnn:.2f}%')

print('\nEvaluating Deep Neural Network with L2 Regularization (DNN-Reg)...')
accuracy_dnn_reg, test_loss_dnn_reg = evaluate_model(model_dnn_reg, test_loader)
print(f'Test Accuracy: {accuracy_dnn_reg:.2f}%')

# Calculate AIC and EAIC
print('\nCalculating AIC and EAIC for SNN...')
AIC_snn, EAIC_snn, k_snn, k_eff_snn, nll_snn = calculate_aic_eaic(model_snn, train_loader, threshold_r)

print('\nCalculating AIC and EAIC for DNN...')
AIC_dnn, EAIC_dnn, k_dnn, k_eff_dnn, nll_dnn = calculate_aic_eaic(model_dnn, train_loader, threshold_r)

print('\nCalculating AIC and EAIC for DNN-Reg...')
AIC_dnn_reg, EAIC_dnn_reg, k_dnn_reg, k_eff_dnn_reg, nll_dnn_reg = calculate_aic_eaic(model_dnn_reg, train_loader, threshold_r)

# Report results
print('\n===== Results =====')
print(f'SNN Test Accuracy: {accuracy_snn:.2f}%')
print(f'SNN AIC: {AIC_snn:.2f}, EAIC: {EAIC_snn:.2f}, Total Params: {k_snn}, Effective Params: {k_eff_snn}')

print(f'\nDNN Test Accuracy: {accuracy_dnn:.2f}%')
print(f'DNN AIC: {AIC_dnn:.2f}, EAIC: {EAIC_dnn:.2f}, Total Params: {k_dnn}, Effective Params: {k_eff_dnn}')

print(f'\nDNN-Reg Test Accuracy: {accuracy_dnn_reg:.2f}%')
print(f'DNN-Reg AIC: {AIC_dnn_reg:.2f}, EAIC: {EAIC_dnn_reg:.2f}, Total Params: {k_dnn_reg}, Effective Params: {k_eff_dnn_reg}')

# Correlation Analysis
aic_values = np.array([AIC_snn, AIC_dnn, AIC_dnn_reg])
eaic_values = np.array([EAIC_snn, EAIC_dnn, EAIC_dnn_reg])
test_accuracies = np.array([accuracy_snn, accuracy_dnn, accuracy_dnn_reg])

corr_aic = np.corrcoef(aic_values, test_accuracies)[0,1]
corr_eaic = np.corrcoef(eaic_values, test_accuracies)[0,1]

print('\n===== Correlation Analysis =====')
print(f'Correlation between AIC and Test Accuracy: {corr_aic*100:.2f}%')
print(f'Correlation between EAIC and Test Accuracy: {corr_eaic*100:.2f}%')
