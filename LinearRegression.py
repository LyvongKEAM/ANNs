# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

# # Load the CSV file
# data = pd.read_csv('flood.csv')

# # Separate features and targets
# X = data[['Water_Level']].values
# y = data['Flood_Occurrence'].values

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Convert to PyTorch tensors
# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.long)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.long)

# # Define the neural network model with four hidden layers and dropout
# class ImprovedANN(nn.Module):
#     def __init__(self):
#         super(ImprovedANN, self).__init__()
#         self.fc1 = nn.Linear(1, 20)
#         self.fc2 = nn.Linear(20, 20)
#         self.fc3 = nn.Linear(20, 20)
#         self.fc4 = nn.Linear(20, 20)
#         self.fc5 = nn.Linear(20, 2)
#         self.dropout = nn.Dropout(0.5)  # Dropout with probability 0.5

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = torch.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = torch.relu(self.fc3(x))
#         x = self.dropout(x)
#         x = torch.relu(self.fc4(x))
#         x = self.fc5(x)
#         return x

# model = ImprovedANN()

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Train the model
# num_epochs = 200
# best_accuracy = 0.0
# patience = 10
# trigger_times = 0

# train_losses = []
# test_accuracies = []

# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train)
#     loss = criterion(outputs, y_train)
#     loss.backward()
#     optimizer.step()
    
#     train_losses.append(loss.item())
    
#     if (epoch + 1) % 10 == 0:
#         model.eval()
#         with torch.no_grad():
#             outputs = model(X_test)
#             _, predicted = torch.max(outputs.data, 1)
#             accuracy = (predicted == y_test).sum().item() / y_test.size(0)
#             test_accuracies.append(accuracy)
#             print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {accuracy * 100:.2f}%')
            
#             # Early stopping
#             if accuracy > best_accuracy:
#                 best_accuracy = accuracy
#                 trigger_times = 0
#             else:
#                 trigger_times += 1

#             if trigger_times >= patience:
#                 print('Early stopping!')
#                 break

# # Final evaluation
# model.eval()
# with torch.no_grad():
#     outputs = model(X_test)
#     _, predicted = torch.max(outputs.data, 1)
#     accuracy = (predicted == y_test).sum().item() / y_test.size(0)
#     error = 1 - accuracy
#     print(f'Final Accuracy on test set: {accuracy * 100:.2f}%')
#     print(f'Final Error on test set: {error * 100:.2f}%')

# # Plot the training loss and test accuracy
# epochs = range(1, len(train_losses) + 1)
# plt.figure(figsize=(14, 5))

# plt.subplot(1, 2, 1)
# plt.plot(epochs, train_losses, label='Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss over Epochs')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(epochs[9::10], test_accuracies, label='Test Accuracy', marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Test Accuracy over Epochs')
# plt.legend()

# plt.tight_layout()
# plt.show()
# # Scatter plot of actual vs. predicted values
# plt.figure(figsize=(10, 5))
# plt.scatter(X_test, y_test, color='blue', label='Actual')
# plt.scatter(X_test, predicted, color='red', alpha=0.5, label='Predicted')
# plt.xlabel('Water Level (Standardized)')
# plt.ylabel('Flood Occurrence')
# plt.title('Scatter Plot of Actual vs. Predicted Values')
# plt.legend()
# plt.show()


import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('flood.csv')

# Separate features and targets
X = data[['Water_Level']].values
y = data['Flood_Occurrence'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Retain original X_test for plotting
X_test_original = X_test.copy()

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define the neural network model with four hidden layers and dropout
class ImprovedANN(nn.Module):
    def __init__(self):
        super(ImprovedANN, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 20)
        self.fc5 = nn.Linear(20, 2)
        self.dropout = nn.Dropout(0.5)  # Dropout with probability 0.5

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

model = ImprovedANN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 200
best_accuracy = 0.0
patience = 10
trigger_times = 0

train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
            test_accuracies.append(accuracy)
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {accuracy * 100:.2f}%')
            
            # Early stopping
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                trigger_times = 0
            else:
                trigger_times += 1

            if trigger_times >= patience:
                print('Early stopping!')
                break

# Final evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    error = 1 - accuracy
    print(f'Final Accuracy on test set: {accuracy * 100:.2f}%')
    print(f'Final Error on test set: {error * 100:.2f}%')

# Plot the training loss and test accuracy
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs[9::10], test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# Scatter plot of actual vs. predicted values using the original water level values
plt.figure(figsize=(10, 5))
plt.scatter(X_test_original, y_test, color='blue', label='Actual')
plt.scatter(X_test_original, predicted, color='red', alpha=0.5, label='Predicted')
plt.xlabel('Water Level')
plt.ylabel('Flood Occurrence')
plt.title('Scatter Plot of Actual vs. Predicted Values')
plt.legend()
plt.show()
