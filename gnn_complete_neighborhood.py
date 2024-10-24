import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from image_processor_online import ImageGraphProcessor
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


# Define a simple GNN (Graph Neural Network) with dynamic adjacency matrix
class GNNCompleteNeighborhood(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNCompleteNeighborhood, self).__init__()
        # Define two fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def compute_adjacency_matrix(self, x, batch):
        """
        Computes dynamic adjacency matrices for a batch of graphs.
        Each graph's adjacency matrix is computed separately based on
        the cosine similarity between node feature vectors within that graph.
        """
        # Get the number of graphs in the batch
        num_graphs = batch.max().item() + 1

        # List to store adjacency matrices for each graph
        adjacency_matrices = []

        # Iterate over each graph in the batch
        for graph_id in range(num_graphs):
            # Extract the nodes belonging to the current graph
            node_mask = (batch == graph_id)
            x_graph = x[node_mask]

            # Normalize features for cosine similarity
            x_norm = F.normalize(x_graph, p=2, dim=1)

            # Compute the adjacency matrix for this graph
            adjacency_matrix = torch.mm(x_norm, x_norm.t())

            # Set diagonal to 0 (no self-loops)
            adjacency_matrix.fill_diagonal_(0)

            adjacency_matrices.append(adjacency_matrix)

        return adjacency_matrices  # Return a list of matrices

    def forward(self, data):
        x, batch = data.x, data.batch

        # Step 1: Compute dynamic adjacency matrices for the batch
        adjacency_matrices = self.compute_adjacency_matrix(x, batch)

        # Step 2: Apply the first linear transformation and ReLU activation
        x = F.relu(self.fc1(x))

        # Step 3: Use adjacency matrices to propagate features within each graph
        out = []
        for graph_id, adj_matrix in enumerate(adjacency_matrices):
            # Select the nodes for this graph
            node_mask = (batch == graph_id)
            x_graph = x[node_mask]

            # Multiply the node features with the adjacency matrix
            x_graph = torch.mm(adj_matrix, x_graph)

            out.append(x_graph)

        # Concatenate results from all graphs back into a single tensor
        x = torch.cat(out, dim=0)

        # Step 4: Apply the second linear layer
        x = self.fc2(x)

        # Step 5: Global mean pooling to get graph-level output
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)


# Training function
def train_gnn(model, data, optimizer):
    """Trains the GNN model."""
    model.train()
    optimizer.zero_grad()  # Clear previous gradients

    # Forward pass
    out = model(data)  # Get model output

    # Cross-Entropy Loss (use `nll_loss` if output is `log_softmax`)
    loss = F.nll_loss(out, data.y)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    _, predicted = torch.max(out, dim=1)  # Get the index of the max log-probability.
    correct = (predicted == data.y).sum().item()  # Count correct predictions
    total_size = len(data.y)  # Total number of samples

    return loss.item(), correct / total_size


def main():
    # Define the method name as a string
    method_name = "GNNCompleteNeighborhood"

    # Initialize the ImageGraphProcessor with original class names
    processor = ImageGraphProcessor(image_size=(64, 64), num_parts=64, num_clusters=3)

    # Initialize the GNN model
    input_dim = 3  # Node features: RGB colors (3 dimensions)
    hidden_dim = 16  # Hidden dimension size
    output_dim = 10  # Number of classes

    model = GNNCompleteNeighborhood(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Define number of epochs
    num_epochs = 5  # You can adjust this

    # Dataframe list to store each row of predictions for validation
    validation_results = []

    # Training loop with epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Train the model (skipping storing predictions for training phase)
        for i, graph_batch in enumerate(processor.process_batch(train=True)):
            loss, correct_ratio = train_gnn(model, graph_batch, optimizer)
            total_loss += loss
            total_correct += correct_ratio * len(graph_batch)  # Multiply by batch size to accumulate accuracy
            total_samples += len(graph_batch)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f} | Avg Accuracy: {avg_accuracy:.4f}")

        # Validation after each epoch
        correct = 0
        total = 0

        with torch.no_grad():
            for i, pyg_graph in enumerate(processor.process_batch(train=False)):
                out = model(pyg_graph)
                _, predicted = torch.max(out, dim=1)
                total += pyg_graph.y.size(0)
                correct += (predicted == pyg_graph.y).sum().item()

                # For each batch, record method, epoch, image (batch index), true, and predicted values
                for j in range(len(predicted)):
                    validation_results.append({
                        "method": method_name,
                        "epoch": epoch + 1,
                        "image": i * len(predicted) + j,  # Unique image index based on batch and item index
                        "true": pyg_graph.y[j].item(),
                        "pred": predicted[j].item()
                    })

        val_accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1} Validation Accuracy: {val_accuracy:.2f}%\n')

    # After all epochs, save the validation results to a CSV
    df = pd.DataFrame(validation_results)
    output_csv_path = f'validation_results.csv'
    df.to_csv(output_csv_path, index=False)
    print(f"Validation results saved to {output_csv_path}")


if __name__ == "__main__":
    main()
