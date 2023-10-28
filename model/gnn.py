import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from layers import *

class GNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GNNModel, self).__init()

        # GCL Layer
        self.gcl_layer = GCLayer(input_size, hidden_size)

        # ReLU Activation
        self.relu = nn.ReLU()

        # Max Pooling Layer
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))

        # Fully Connected Layer (FC)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # GCL Layer
        x = self.gcl_layer(x)

        # ReLU Activation
        x = self.relu(x)

        # Max Pooling
        x = self.max_pool(x)

        # ReLU Activation
        x = self.relu(x)

        # Max Pooling
        x = self.max_pool(x)

        # Reshape for FC layer
        x = x.view(x.size(0), -1)

        # Fully Connected Layer (FC)
        x = self.fc(x)

        return x


def train(data, labels):
    # Define hyperparameters
    num_features = len(data[0])
    hidden_size = 512
    num_classes = 2
    learning_rate = 1.0e-3
    num_epochs = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model, optimizer, and loss function
    model = GNNModel(num_features, hidden_size, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        output = model(data.x, data.edge_index)

        # Compute loss
        loss = criterion(output, labels)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Compute accuracy
        predicted_labels = output.argmax(dim=1)
        accuracy = (predicted_labels == labels).sum().item() / len(labels)

        # Print epoch statistics
        print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

    return model


def predict(model, X, A):
    model.eval()

    with torch.no_grad():
        output = model(X, A)
        probabilities = F.softmax(output, dim=1)

    return probabilities
