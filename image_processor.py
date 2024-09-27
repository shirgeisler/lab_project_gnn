import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch_geometric.data import Data, Batch
import os
import random


class ImageGraphProcessor:
    def __init__(self, image_size=(64, 64), num_parts=64, num_clusters=3, batch_size=32,
                 data_path='data/tiny-imagenet-200'):
        """
        Initializes the ImageGraphProcessor with parameters for image size, number of parts, and clusters.
        """
        self.image_size = image_size
        self.num_parts = num_parts
        self.num_clusters = num_clusters
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_path = data_path + '/train'
        self.val_path = data_path + '/val'

    def split_image_into_parts(self, image):
        """Splits a PyTorch image tensor into num_parts parts."""
        C, H, W = image.shape
        num_rows = num_cols = int(self.num_parts ** 0.5)

        # Compute height and width of each part
        part_height = H // num_rows
        part_width = W // num_cols

        # Split the image into parts
        parts = []
        for i in range(num_rows):
            for j in range(num_cols):
                part = image[:, i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width]
                parts.append(part)

        return parts, num_rows, num_cols, part_height, part_width

    def calculate_average_color(self, part):
        """Calculates the average color of an image part (RGB channels)."""
        part_np = part.permute(1, 2, 0).numpy()
        avg_color = np.mean(part_np, axis=(0, 1))  # Average over the height and width
        return avg_color

    def create_pyg_graph(self, parts, num_rows, num_cols):
        """Directly creates a PyTorch Geometric graph without relying on NetworkX."""
        avg_colors = [self.calculate_average_color(part) for part in parts]

        # Clustering using KMeans
        kmeans = KMeans(n_clusters=self.num_clusters)
        cluster_labels = kmeans.fit_predict(avg_colors)

        # Initialize node features as a PyTorch tensor (avg color)
        x = torch.tensor(avg_colors, dtype=torch.float)

        # Initialize edge index list
        edge_index = []

        # Add edges based on adjacency
        for row in range(num_rows):
            for col in range(num_cols):
                idx = row * num_cols + col
                if col < num_cols - 1:  # Right neighbor
                    right_neighbor = idx + 1
                    edge_index.append([idx, right_neighbor])
                    edge_index.append([right_neighbor, idx])  # Add reverse direction
                if row < num_rows - 1:  # Bottom neighbor
                    bottom_neighbor = idx + num_cols
                    edge_index.append([idx, bottom_neighbor])
                    edge_index.append([bottom_neighbor, idx])  # Add reverse direction

        # Add edges based on clustering
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                if cluster_labels[i] == cluster_labels[j]:
                    edge_index.append([i, j])
                    edge_index.append([j, i])  # Add reverse direction

        # Convert edge list to a tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Create the PyTorch Geometric graph
        pyg_graph = Data(x=x, edge_index=edge_index)

        return pyg_graph, cluster_labels, edge_index

    def draw_grid_on_image_with_clusters(self, image, num_rows, num_cols, part_height, part_width, edge_index,
                                         cluster_labels):
        """Draws the original image and a grid with clusters and edges side by side,
        and adds edges between neighboring parts in black."""

        fig, axes = plt.subplots(1, 2, figsize=(
        16, 8))  # Create two subplots, one for the original image and one for the grid

        # Original Image
        ax1 = axes[0]
        image_np = image.permute(1, 2, 0).numpy()  # Convert to HWC format (Height, Width, Channels)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Rescale for display
        ax1.imshow(image_np)
        ax1.set_title("Original Image")
        ax1.axis("off")  # Turn off axis for the original image

        # Image with Grid and Clusters
        ax2 = axes[1]
        ax2.imshow(image_np)

        # Draw the grid
        for i in range(num_rows + 1):
            ax2.axhline(i * part_height, color='white', linewidth=1)
        for j in range(num_cols + 1):
            ax2.axvline(j * part_width, color='white', linewidth=1)

        # Compute the center of each grid cell (used for drawing edges)
        pos = {}
        for idx in range(num_rows * num_cols):
            row = idx // num_cols
            col = idx % num_cols
            x_center = (col + 0.5) * part_width
            y_center = (row + 0.5) * part_height
            pos[idx] = (x_center, y_center)

        # Draw the edges from edge_index
        for start, end in edge_index.t().tolist():
            x_start, y_start = pos[start]
            x_end, y_end = pos[end]
            if cluster_labels[start] == cluster_labels[end]:  # Clustering-based edges (color similarity)
                colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'purple', 5: 'orange'}
                color = colors[cluster_labels[start]]
                ax2.plot([x_start, x_end], [y_start, y_end], color=color, lw=2)
            else:  # Neighbor-based edges (adjacent parts)
                ax2.plot([x_start, x_end], [y_start, y_end], color='black', lw=2)  # Black for edges between neighbors

        ax2.set_title("Image with Grid, Clusters, and Neighbor Edges")
        ax2.axis("off")  # Turn off axis for the image with grid

        plt.tight_layout()
        plt.show()

    def process_and_save_graphs(self, train=True):
        """Processes images, constructs PyG graphs, and saves them individually to disk."""

        self.folder_name_to_label = {}
        with open(self.data_path + '/words.txt', 'r') as f:
            for line in f:
                parts = line.split('\t')
                self.folder_name_to_label[parts[0]] = parts[1].strip()

        # Set up transforms for image loading
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load the datasett

        if train:
            self.train_dataset = datasets.ImageFolder(root=self.self.train_path, transform=self.transform)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        else:
            self.val_dataset = datasets.ImageFolder(root=self.val_path, transform=self.transform)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        data_loader = self.train_loader if train else self.val_loader
        graph_count = 0  # Counter for naming saved graphs
        string_train_val = 'train' if train else 'val'

        for images, labels in data_loader:
            for i in range(images.size(0)):  # Loop through each image in the batch
                image = images[i]
                parts, num_rows, num_cols, part_height, part_width = self.split_image_into_parts(image)
                pyg_graph, cluster_labels, edge_index = self.create_pyg_graph(parts, num_rows, num_cols)
                pyg_graph.y = labels[i]  # Assign the label to the graph

                # Save each graph individually
                graph_filename = os.path.join(string_train_val + '_' + self.save_path, f"graph_{graph_count}.pt")
                torch.save(pyg_graph, graph_filename)
                print(f"Saved graph {graph_count} to {graph_filename}")
                graph_count += 1


    def load_saved_graphs(self, shuffle=True, train=True):
        """Loads pre-processed graphs from disk in batches."""
        train_or_val_string = 'train' if train else 'val'
        graphs_path = train_or_val_string + '_graphs'

        graph_files = sorted(os.listdir(graphs_path))  # List of saved graph files

        # Shuffle the graph files if shuffle=True
        if shuffle:
            random.shuffle(graph_files)

        # Load graphs in batches
        for i in range(0, len(graph_files), self.batch_size):
            graphs_batch = []
            for j in range(i, min(i + self.batch_size, len(graph_files))):
                graph_filename = os.path.join(graphs_path, graph_files[j])
                pyg_graph = torch.load(graph_filename)
                graphs_batch.append(pyg_graph)
            # Create a PyTorch Geometric batch from the list of individual graphs
            #print the names of the graphs

            yield graphs_batch

if __name__ == '__main__':




    # Initialize the processor
    processor = ImageGraphProcessor(image_size=(64, 64), num_parts=64, num_clusters=3)

    # Process and save batches of images and graphs
    # processor.process_and_save_graphs(train=True)
    #processor.process_and_save_graphs(train=False)  # if Train is False, then it will process the validation data

    # # Example: Load saved batches and use for training or evaluation
    for batched_graph in processor.load_saved_graphs(shuffle=True, train=True):
        print(f"Loaded batch with {len(batched_graph)} graphs")
