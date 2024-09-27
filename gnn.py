import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from image_processor_obsolete import ImageGraphProcessor
from tqdm import tqdm


# Define a simple GCN (Graph Convolutional Network)
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        # Graph Convolutional Layers
        self.conv1 = GCNConv(input_dim, hidden_dim)  # First GCN layer
        self.conv2 = GCNConv(hidden_dim, output_dim)  # Second GCN layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First GCN layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second GCN layer
        x = self.conv2(x, edge_index)

        #adjust output dimension to be 1 with a linear layer
        #we want the output to be a tensor of (32,200) where 32 is the batch size and 200 is the number of classes

        x = global_mean_pool(x, data.batch)


        return F.log_softmax(x, dim=1)  # Log Softmax for classification


def train_gnn(model, data, labels, optimizer):
    """Trains the GNN model."""
    model.train()
    total_loss = 0

    for pyg_graph, label in zip(data, labels):
        optimizer.zero_grad()

        # Forward pass
        out = model(pyg_graph)
        #one hot encode the labels
        label = F.one_hot(label, num_classes=200)

        loss = F.nll_loss(out[0], label)

        loss.backward()  # Backward pass
        optimizer.step()  # Update the model weights

        total_loss += loss.item()

    return total_loss / len(data)


def main():
    # Initialize the ImageGraphProcessor
    processor = ImageGraphProcessor(image_size=(64, 64), num_parts=64, num_clusters=3)

    # Initialize the GNN model
    input_dim = 3  # Since node features are average RGB colors (3 dimensions)
    hidden_dim = 16  # Hidden dimension size
    output_dim = 200  # Output classes (e.g., binary classification)

    model = GNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the GNN model

    # Training Loop with tqdm for progress tracking
    batch_count = 0
    for graphs_batch, labels_batch in processor.process_batch(train=True, batch_size=32):
        # Now you have a batch of graphs and their corresponding labels
        batch_count += 1
        loss = train_gnn(model, graphs_batch, labels_batch, optimizer)  # Train on the batch of graphs
        print(f"number of batches processed: {batch_count} / {len(processor.train_loader)} | total loss: {loss}")

    # Validation Loop with tqdm for progress tracking
    correct = 0
    total = 0
    with torch.no_grad():
        for pyg_graph, labels in tqdm(processor.process_batch(train=False), desc="Validation"):
            out = model(pyg_graph)
            _, predicted = torch.max(out, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} validation images: {100 * correct / total}%')

if __name__ == '__main__':
    main()