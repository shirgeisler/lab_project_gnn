import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GlobalAttention, global_mean_pool


from image_processor import ImageGraphProcessor


# Define a simple GCN (Graph Convolutional Network)
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        # Graph Convolutional Layers
        self.conv1 = GCNConv(input_dim, hidden_dim)  # First GCN layer
        self.conv2 = GCNConv(hidden_dim, hidden_dim)  # Second GCN layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)  # Linear layer for output

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GCN layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second GCN layer
        x = self.conv2(x, edge_index)

        # Pooling to get graph-level output (global mean pooling)
        x = global_mean_pool(x, batch)  # Shape: [batch_size, hidden_dim]

        # Linear transformation to match the number of classes
        x = self.fc(x)  # Shape: [batch_size, output_dim]

        return F.log_softmax(x, dim=1)  # Log Softmax for classification



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
    _, predicted = torch.max(out, dim=1)  # Get the index of the max log-probability
    correct = (predicted == data.y).sum().item()  # Count correct predictions
    total_size = len(data.y)  # Total number of samples

    return loss.item(), correct / total_size


def main():
    # Assume ImageGraphProcessor and other parts are defined elsewhere
    processor = ImageGraphProcessor(image_size=(64, 64), num_parts=64, num_clusters=3)

    # Initialize the GNN model
    input_dim = 3  # Node features: RGB colors (3 dimensions)
    hidden_dim = 16  # Hidden dimension size
    output_dim = 200  # Number of classes (update as needed)

    model = GNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training Loop with tqdm for progress tracking
    batch_count = 0
    for graph_batch in processor.load_saved_graphs():
        batch_count += 1
        loss, correct_ratio = train_gnn(model, graph_batch, optimizer)
        print(
            f"Batch {batch_count} | Batch Size: {len(graph_batch)} | Loss: {loss:.4f} | Accuracy: {correct_ratio:.4f}")

    # Validation Loop
    correct = 0
    total = 0
    with torch.no_grad():
        for pyg_graph, labels in processor.process_batch(train=False):
            out = model(pyg_graph)
            _, predicted = torch.max(out, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {100 * correct / total:.2f}%')


if __name__ == "__main__":
    main()
