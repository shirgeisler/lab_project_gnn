import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch_geometric.data import Data
from baseline import filter_dataset
import random
import os
from torch_geometric.data import Batch


SEED = 30
random.seed(SEED)
train_path = 'data/tiny-imagenet-200/train'
test_path = 'data/tiny-imagenet-200/val'


class ImageGraphProcessor:
    def __init__(self, method, image_size=(64, 64), num_parts=64, num_clusters=3, data_path='data/tiny-imagenet-200'):
        """
        Initializes the ImageGraphProcessor with parameters for image size, number of parts, and clusters.
        """
        self.image_size = image_size
        self.num_parts = num_parts
        self.num_clusters = num_clusters
        self.num_nearest_neighbors = 5
        self.method = method
        self.folder_name_to_label = {}

        with open(data_path + '/words.txt', 'r') as f:
            for line in f:
                parts = line.split('\t')
                self.folder_name_to_label[parts[0]] = parts[1].strip()

        # Set up transforms for image loading
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        all_classes = os.listdir(train_path)  # Each sub-folder represents a class
        self.selected_classes = random.sample(all_classes, 10)
        self.original_class_names = [self.folder_name_to_label[c] for c in self.selected_classes]

        train_dataset = datasets.ImageFolder(root=train_path, transform=self.transform)
        test_dataset = datasets.ImageFolder(root=test_path, transform=self.transform)

        train_dataset = filter_dataset(train_dataset, self.selected_classes)
        test_dataset = filter_dataset(test_dataset, self.selected_classes)

        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    def get_class_names(self):
        """Returns the human-readable class names in the same order as the selected classes."""
        return self.original_class_names

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

        # Initialize node features as a PyTorch tensor (avg color)
        x = torch.tensor(avg_colors, dtype=torch.float)

        # Initialize edge index list
        edge_index = []
        edge_attr = []
        cluster_labels = None

        if self.method == 'complete_graph':
            num_nodes = len(parts)
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    edge_index.append([i, j])
                    edge_index.append([j, i])

                    # Compute weight as the negative Euclidean distance
                    weight = -torch.dist(x[i], x[j], p=2).item()
                    edge_attr.extend([weight, weight])  # Symmetric edge weights

            # Convert edge_index and edge_attr to tensors
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
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

        if self.method == 'kmeans':
            # Clustering using KMeans
            kmeans = KMeans(n_clusters=self.num_clusters)
            cluster_labels = kmeans.fit_predict(avg_colors)
            # Add edges based on clustering
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    if cluster_labels[i] == cluster_labels[j]:
                        edge_index.append([i, j])
                        edge_index.append([j, i])  # Add reverse direction

            # Convert edge list to a tensor
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        if self.method == 'knn':
            # create neighbor edges based on KNN
            from sklearn.neighbors import NearestNeighbors
            neigh = NearestNeighbors(n_neighbors=self.num_nearest_neighbors)
            neigh.fit(avg_colors)
            knn = neigh.kneighbors(avg_colors, return_distance=False)
            for i in range(len(parts)):
                for j in range(len(knn[i])):
                    edge_index.append([i, knn[i][j]])
                    edge_index.append([knn[i][j], i])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Create a PyTorch Geometric graph
        pyg_graph = Data(x=x, edge_index=edge_index)

        if self.method == 'complete_graph':
            pyg_graph.edge_attr = edge_attr  # Attach edge weights

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

    def process_batch(self, train=True, batch_size=32):
        """Processes a batch of images, constructs a batch of PyG graphs, and returns them."""
        graphs_batch = []  # To hold multiple graphs

        data_loader = self.train_loader if train else self.test_loader

        for images, labels in data_loader:
            for i in range(images.size(0)):  # Loop through each image in the batch
                image = images[i]  # Get each image from the batch

                # Split the image into parts
                parts, num_rows, num_cols, part_height, part_width = self.split_image_into_parts(image)

                # Create the graph, ensure node features are 64x3
                pyg_graph, cluster_labels, edge_index = self.create_pyg_graph(parts, num_rows, num_cols)

                assert pyg_graph.x.size() == (64, 3), "Node features should be 64x3."

                pyg_graph.y = labels[i]

                # Add the graph and corresponding label to the batch
                graphs_batch.append(pyg_graph)

                # Return a batch of graphs if it reaches the batch size
                if len(graphs_batch) == batch_size:
                    batch = Batch.from_data_list(graphs_batch)  # Create batch of graphs
                    assert batch.x.size(0) == batch_size * 64, "Batch node features should be (batch_size * 64) by 3."
                    yield batch  # Yield the batch for further processing
                    graphs_batch = []

        # Yield any remaining graphs if there are fewer than batch_size
        if graphs_batch:
            yield Batch.from_data_list(graphs_batch)


if __name__ == '__main__':
    processor = ImageGraphProcessor(image_size=(64, 64), num_parts=64, num_clusters=3)

    # Process a batch of images and create graphs
    for pyg_graph, cluster_labels in processor.process_batch():
        # The graph is processed and the image with clusters is displayed
        print(pyg_graph)