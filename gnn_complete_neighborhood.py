import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from image_processor_online import ImageGraphProcessor

# Define a simple GNN (Graph Neural Network) with dynamic adjacency matrix
class GNNCompleteNeighborhood(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNCompleteNeighborhood, self).__init__()
        # Define two fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def compute_adjacency_matrix(self, x):
        """
        Computes a fully connected dynamic adjacency matrix where the weights
        are determined by the cosine similarity between node feature vectors.
        """
        print()
        num_nodes = x.size(0)  # Number of nodes
        adjacency_matrix = torch.ones(num_nodes, num_nodes)  # Fully connected with ones initially

        # Compute dynamic weights using cosine similarity between nodes
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    similarity = F.cosine_similarity(x[i].unsqueeze(0), x[j].unsqueeze(0))
                    adjacency_matrix[i, j] = similarity.item()  # Set edge weight as cosine similarity

        return adjacency_matrix

    def forward(self, data):
        x, batch = data.x, data.batch

        # Step 1: Compute the dynamic adjacency matrix based on node features
        adjacency_matrix = self.compute_adjacency_matrix(x)

        # Step 2: First layer: apply linear transformation and ReLU activation
        x = F.relu(self.fc1(x))

        # Step 3: Multiply the transformed features by the dynamic adjacency matrix
        x = torch.mm(adjacency_matrix, x)

        # Step 4: Apply the second linear layer
        x = self.fc2(x)

        # Step 5: Pooling operation to get graph-level output
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)  # Apply log-softmax for classification


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
    # Assume ImageGraphProcessor and other parts are defined elsewhere
    processor = ImageGraphProcessor(image_size=(64, 64), num_parts=64, num_clusters=3)

    # Initialize the GNN model
    input_dim = 3  # Node features: RGB colors (3 dimensions)
    hidden_dim = 16  # Hidden dimension size
    output_dim = 10  # Number of classes

    model = GNNCompleteNeighborhood(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Define number of epochs
    num_epochs = 10  # You can adjust this

    # Training loop with epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for i, graph_batch in enumerate(processor.process_batch(train=True)):
            loss, correct_ratio = train_gnn(model, graph_batch, optimizer)
            total_loss += loss
            total_correct += correct_ratio * len(graph_batch)  # Multiply by batch size to accumulate accuracy
            total_samples += len(graph_batch)
            print(f"batch {i} finished")

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f} | Avg Accuracy: {avg_accuracy:.4f}")

        # Validation after each epoch
        correct = 0
        total = 0
        with torch.no_grad():
            for pyg_graph in processor.process_batch(train=False):
                out = model(pyg_graph)
                _, predicted = torch.max(out, dim=1)
                total += pyg_graph.y.size(0)
                correct += (predicted == pyg_graph.y).sum().item()

        print(f'Epoch {epoch + 1} Validation Accuracy: {100 * correct / total:.2f}%\n')


if __name__ == "__main__":
    main()
