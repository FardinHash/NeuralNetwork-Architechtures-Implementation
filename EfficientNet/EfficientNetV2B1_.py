# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet

# Load the EfficientNetV2B1 model
model = EfficientNet.from_pretrained('efficientnet-b1')

# Set the model to training mode
model.train()

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Train the model on dataset
for epoch in range(10):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Evaluate the model on test dataset
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

test_acc = 100 * correct / total

print('Test accuracy:', test_acc)
