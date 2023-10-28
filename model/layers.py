import torch
import torch.nn as nn
import random


class GCLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))

        # Initialize learnable parameters
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, adjacency_matrix, node_features):
        support = torch.matmul(node_features, self.weight)
        output = torch.matmul(adjacency_matrix, support) + self.bias
        return output


class MultiviewContrastiveLayer(nn.Module):
    def __init__(self, input_size, view_size):
        super(MultiviewContrastiveLayer, self).__init__()
        self.input_size = input_size
        self.view_size = view_size

    def subsequence_crop(self, graph, sequence_length):
        # Randomly sample a subsequence
        start_idx = random.randint(0, len(graph) - sequence_length)
        end_idx = start_idx + sequence_length
        subsequence = graph[start_idx:end_idx]
        return subsequence

    def subspace_crop(self, graph, center_node, distance_threshold):
        # Sample a subgraph around a center node
        subgraph = []
        for node in graph:
            if abs(node - center_node) <= distance_threshold:
                subgraph.append(node)
        return subgraph

    def random_edge_masking(self, graph, mask_ratio):
        # Randomly remove a fixed ratio of edges from the graph
        num_edges_to_mask = int(len(graph) * mask_ratio)
        if num_edges_to_mask > 0:
            masked_indices = random.sample(range(len(graph)), num_edges_to_mask)
            for index in masked_indices:
                graph[index] = 0
        return graph

    def forward(self, graph, sequence_length, center_node, distance_threshold, mask_ratio):
        # Apply subsequence cropping
        subsequence_view = self.subsequence_crop(graph, sequence_length)

        # Apply subspace cropping
        subspace_view = self.subspace_crop(graph, center_node, distance_threshold)

        # Apply random transformation (edge masking)
        if mask_ratio > 0:
            subsequence_view = self.random_edge_masking(subsequence_view, mask_ratio)
            subspace_view = self.random_edge_masking(subspace_view, mask_ratio)

        return subsequence_view, subspace_view


class EdgeMessagePassingLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(EdgeMessagePassingLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, adjacency_matrix, edge_features):
        # Calculate the line graph adjacency matrix
        line_graph_adjacency = self.calculate_line_graph_adjacency(adjacency_matrix)

        # Apply relational graph convolution
        messages = torch.matmul(line_graph_adjacency, edge_features)

        # Aggregate messages (e.g., sum or mean)
        aggregated_messages = torch.sum(messages, dim=1)

        # Apply a linear layer to update the edge features
        updated_edge_features = self.linear(aggregated_messages)

        return updated_edge_features

    def calculate_line_graph_adjacency(self, adjacency_matrix):
        num_nodes = adjacency_matrix.shape[0]
        line_graph_adjacency = torch.zeros(num_nodes, num_nodes)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    for k in range(num_nodes):
                        if k != i and k != j and adjacency_matrix[i][k] == 1 and adjacency_matrix[j][k] == 1:
                            line_graph_adjacency[i][j] = 1
                            line_graph_adjacency[j][i] = 1

        return line_graph_adjacency