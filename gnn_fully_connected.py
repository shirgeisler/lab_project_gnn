import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from image_processor_online import ImageGraphProcessor
import pandas as pd
import os


class GNNFullyConnected(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNFullyConnected, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def compute_adjacency_matrix(self, x, batch):
        """
           Compute dynamic adjacency matrices for each graph in the batch.
           Uses cosine similarity between node features within each graph to construct
           separate adjacency matrices, with self-connections (diagonal) set to zero.

           Args:
               x (Tensor): Node features of shape [total_nodes, feature_dim].
               batch (Tensor): Batch vector assigning each node to a specific graph, with
                               shape [total_nodes].

           Returns:
               List[Tensor]: A list of adjacency matrices, one for each graph in the batch,
                             where each matrix has shape [num_nodes_graph, num_nodes_graph].
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

            # Set diagonal to 0
            adjacency_matrix.fill_diagonal_(0)

            adjacency_matrices.append(adjacency_matrix)

        return adjacency_matrices

    def forward(self, data):
        """
        Perform a forward pass through the GNN model.

        This method computes dynamic adjacency matrices for each graph in the batch
        and applies feature transformations and propagation within each graph.
        Finally, it performs global mean pooling to get graph-level representations
        and returns log softmax values for classification.

        Args:
            data (Batch): A PyTorch Geometric Batch object containing:
                          - x (Tensor): Node features of shape [total_nodes, feature_dim].
                          - batch (Tensor): Batch vector assigning each node to a specific
                                            graph, with shape [total_nodes].

        Returns:
            Tensor: Log-softmax probabilities of shape [num_graphs, num_classes] for each
                    graph in the batch.
        """

        x, batch = data.x, data.batch

        # Compute dynamic adjacency matrices for the batch
        adjacency_matrices = self.compute_adjacency_matrix(x, batch)

        # Apply the first linear transformation and ReLU activation
        x = F.relu(self.fc1(x))

        # Use adjacency matrices to propagate features within each graph
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

        # Apply the second linear layer
        x = self.fc2(x)

        # Global mean pooling to get graph-level output
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)


def train_gnn(model, data, optimizer):
    """
    Train the GNN model on a single batch of data.

    Performs a forward pass, computes the cross-entropy loss, backpropagates
    the error, and updates the model parameters. It also calculates the
    batch accuracy.

    Args:
        model (torch.nn.Module): The GNN model to be trained.
        data (Batch): A PyTorch Geometric Batch containing:
                      - x (Tensor): Node features.
                      - edge_index (Tensor): Edge indices for the graph.
                      - y (Tensor): True labels for graph-level or node-level targets.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.

    Returns:
        tuple: Contains:
            - loss (float): The cross-entropy loss for the batch.
            - accuracy (float): The accuracy for the batch, calculated as the
              proportion of correct predictions.
    """

    model.train()
    optimizer.zero_grad()  # Clear previous gradients

    # Forward pass
    out = model(data)  # Get model output
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
    """
        Main function for training and validating the GNN model.

        This function initializes the data processor, GNN model, and optimizer, then
        trains and validates the model over multiple epochs. Validation predictions
        are saved to a CSV file, which appends new data on each run without overwriting
        existing results.

        Args:
            None

        Returns:
            None: Outputs are printed to the console, and validation results are saved
                  to 'validation_results.csv' after all epochs.
    """

    method_name = "complete_graph"
    processor = ImageGraphProcessor(method=method_name, image_size=(64, 64), num_parts=64, num_clusters=3)

    # Initialize the GNN model
    input_dim = 3  # Node features: RGB colors (3 dimensions)
    hidden_dim = 16
    output_dim = 10

    model = GNNFullyConnected(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 5

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

    if os.path.isfile(output_csv_path):
        df.to_csv(output_csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_csv_path, index=False)

    print(f"Validation results saved to {output_csv_path}")


if __name__ == "__main__":
    main()
