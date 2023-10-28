from layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultiviewContrastiveGCLAutoencoder(nn.Module):
    def __init__(self, input_size, latent_size, view_size):
        super(MultiviewContrastiveGCLAutoencoder, self).__init()
        self.input_size = input_size
        self.latent_size = latent_size
        self.view_size = view_size

        self.gcl1 = GCLayer(input_size, 64)
        self.gcl2 = GCLayer(64, 32)
        self.gcl3 = GCLayer(32, 16)
        self.gcl4 = GCLayer(16, latent_size)

        # Multiview Contrastive Layer
        self.multiview_layer = MultiviewContrastiveLayer(input_size, view_size)

        # Edge Message Passing Layer
        self.edge_message_passing_layer = EdgeMessagePassingLayer(input_size, input_size)

        self.gcl5 = GCLayer(latent_size, 16)
        self.gcl6 = GCLayer(16, 32)
        self.gcl7 = GCLayer(32, 64)
        self.gcl8 = GCLayer(64, input_size)

    def forward(self, adjacency_matrix, node_features, edge_features):
        # Encoder
        x = F.relu(self.gcl1(adjacency_matrix, node_features))
        x = F.relu(self.gcl2(adjacency_matrix, x))
        x = F.relu(self.gcl3(adjacency_matrix, x))
        encoded = self.gcl4(adjacency_matrix, x)

        # Multiview Contrastive Layer
        subsequence_view, subspace_view = self.multiview_layer(node_features)

        # Edge Message Passing Layer
        updated_edge_features = self.edge_message_passing_layer(adjacency_matrix, edge_features)

        # Decoder
        x = F.relu(self.gcl5(adjacency_matrix, encoded))
        x = F.relu(self.gcl6(adjacency_matrix, x))
        x = F.relu(self.gcl7(adjacency_matrix, x))
        reconstructed_node_features = self.gcl8(adjacency_matrix, x)

        return encoded, subsequence_view, subspace_view, updated_edge_features, reconstructed_node_features


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
        super(Transformer, self).__init__()

        # Embedding layer for adjacency matrix
        self.adjacency_embedding = nn.Linear(input_size, hidden_size)

        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Feed-forward layer
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, adjacency_matrix, reconstructed_graph):
        # Embed the adjacency matrix
        adjacency_embedding = self.adjacency_embedding(adjacency_matrix)

        # Prepare input for multi-head attention
        mat_input = adjacency_embedding.permute(1, 0, 2)  # Reshape for multi-head attention

        # Apply multi-head self-attention
        for attention_layer in self.attention_layers:
            output, _ = attention_layer(mat_input, mat_input, mat_input)

        # Reshape output for feed-forward layer
        output = output.permute(1, 0, 2)

        # Apply feed-forward layer
        output = self.feed_forward(output)

        return output


class RepresentationLearningModel(nn.Module):
    def __init__(self, input_size, latent_size, view_size, transformer_input_size, hidden_size, num_layers, num_heads,
                 dropout, lambda0, lambda1, lambda2, lambda3):
        super(RepresentationLearningModel, self).__init()
        self.multiview_gcl_autoencoder = MultiviewContrastiveGCLAutoencoder(input_size, latent_size, view_size)
        self.transformer = Transformer(transformer_input_size, hidden_size, num_layers, num_heads, dropout)
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def forward(self, adjacency_matrix, node_features, edge_features, transformer_input):
        # Multiview GCL Autoencoder
        encoded, subsequence_view, subspace_view, updated_edge_features, reconstructed_node_features = \
            self.multiview_gcl_autoencoder(adjacency_matrix, node_features, edge_features)

        # Transformer
        transformer_output = self.transformer(adjacency_matrix, transformer_input)

        return encoded, subsequence_view, subspace_view, updated_edge_features, \
            reconstructed_node_features, transformer_output

    def calculate_loss(self, adjacency_matrix, node_features, edge_features, transformer_input,
                       target_graph_properties):

        encoded, subsequence_view, subspace_view, updated_edge_features, reconstructed_node_features, \
            transformer_output = self(adjacency_matrix, node_features, edge_features, transformer_input)

        # Node feature reconstruction loss (cross-entropy)
        loss_node_reconstruction = F.cross_entropy(node_features.view(-1, node_features.size(-1)),
                                                   reconstructed_node_features.view(-1,
                                                                                    reconstructed_node_features.size(
                                                                                        -1)))

        # KL Divergence
        mean_encoded = torch.mean(encoded, dim=1)
        kl_divergence = -0.5 * torch.sum(1 + encoded - mean_encoded - torch.exp(encoded), dim=1)

        # Mean Squared Error (MSE) loss for graph properties
        mse_loss = F.mse_loss(transformer_output, target_graph_properties)

        # Edge feature reconstruction loss (cross-entropy)
        loss_edge_reconstruction = F.cross_entropy(updated_edge_features.view(-1, updated_edge_features.size(-1)),
                                                   edge_features.view(-1, edge_features.size(-1)))

        # Total loss
        loss = self.lambda0 * loss_node_reconstruction + self.lambda1 * torch.mean(
            kl_divergence) + self.lambda2 * mse_loss + self.lambda3 * loss_edge_reconstruction

        return loss


def train(train_data_loader, num_epochs, learning_rate):
    # Example usage
    input_size = 30  # Input size for nodes and edges in GCL Autoencoder
    latent_size = 128  # Size of the latent space
    view_size = 64  # View size for the MultiviewContrastiveLayer
    transformer_input_size = 64  # Input size for the Transformer
    hidden_size = 128  # Hidden size for the Transformer
    num_layers = 4  # Number of Transformer layers
    num_heads = 8  # Number of attention heads
    dropout = 0.1  # Dropout rate
    lambda0 = 0.2  # Weight for the node reconstruction loss
    lambda1 = 0.3  # Weight for the KL divergence loss
    lambda2 = 0.3  # Weight for the MSE loss
    lambda3 = 0.2  # Weight for the edge reconstruction loss

    # Create the combined model
    model = RepresentationLearningModel(input_size, latent_size, view_size, transformer_input_size, hidden_size, num_layers,
                                   num_heads, dropout, lambda0, lambda1, lambda2, lambda3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_data in train_data_loader:
            adjacency_matrix, node_features, edge_features, transformer_input, target_graph_properties = batch_data

            # Move data to the GPU, if available
            adjacency_matrix = adjacency_matrix.to(device)
            node_features = node_features.to(device)
            edge_features = edge_features.to(device)
            transformer_input = transformer_input.to(device)
            target_graph_properties = target_graph_properties.to(device)

            # Forward pass
            encoded, _, _, _, transformer_output = model(adjacency_matrix, node_features, edge_features,
                                                         transformer_input)

            # Calculate the loss
            loss = model.calculate_loss(adjacency_matrix, node_features, edge_features, transformer_input,
                                        target_graph_properties)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print loss for the epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {total_loss / len(train_data_loader)}")
    return model


# Get representation for a single PPI
def get_interface_representation(model, node_features, adj_matrix, edge_features, transformer_input):
    model.eval()
    adj_matrix = torch.sparse_coo_tensor(adj_matrix.nonzero(), adj_matrix[adj_matrix.nonzero()].flatten(),
                                         size=adj_matrix.shape)

    # Convert input tensors to PyTorch tensors
    node_features_tensor = torch.tensor(node_features).float()
    adj_tensor = torch.tensor(adj_matrix).float()
    edge_features = torch.tensor(edge_features).float()
    transformer_input = torch.tensor(transformer_input).float()

    # Forward pass
    encoded, _, _, _, transformer_output = model(adj_tensor, node_features_tensor, edge_features,
                                                 transformer_input)

    return encoded, transformer_output




