import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from image_processor_online import ImageGraphProcessor
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


# Define a simple GCN (Graph Convolutional Network)
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        # Graph Convolutional Layers
        self.conv1 = GCNConv(input_dim, hidden_dim)  # First GCN layer
        self.conv2 = GCNConv(hidden_dim, hidden_dim)  # Second GCN layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)  # Linear layer for output

    def forward(self, data):
        # Extract necessary components from the Batch
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GCN layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second GCN layer
        x = self.conv2(x, edge_index)

        # Global mean pooling to get graph-level output
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
    _, predicted = torch.max(out, dim=1)  # Get the index of the max log-probability.
    correct = (predicted == data.y).sum().item()  # Count correct predictions
    total_size = len(data.y)  # Total number of samples

    return loss.item(), correct / total_size


def main():
    # Define the method name as a string
    method_name = "knn"

    # Initialize the ImageGraphProcessor with original class names
    processor = ImageGraphProcessor(image_size=(64, 64), num_parts=64, num_clusters=3)

    # Initialize the GNN model
    input_dim = 3  # Node features: RGB colors (3 dimensions)
    hidden_dim = 16  # Hidden dimension size
    output_dim = 10  # Number of classes

    model = GNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Define number of epochs
    num_epochs = 10  # You can adjust this

    # Dataframe list to store each row of predictions for validation
    validation_results = []

    # Training loop with epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Train the model (skipping storing predictions for training phase)
        for graph_batch in processor.process_batch(train=True):
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
    output_csv_path = f'validation_results3.csv'
    df.to_csv(output_csv_path, index=False)
    print(f"Validation results saved to {output_csv_path}")


if __name__ == "__main__":
    main()
