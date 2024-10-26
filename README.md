# Graph Construction Methods for Image Classification Using Graph Neural Networks

This project explores different graph construction methods for image classification using Graph Neural Networks (GNNs).

## Instructions

1. **Download the Dataset**  
   Download the Tiny ImageNet dataset from Kaggle: [Tiny ImageNet Dataset](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet?resource=download).

2. **Reformat Validation Data**  
   Run `reformat_validation_set.py` to reorganize the validation files, ensuring they match the structure of the training data.

3. **Baseline Results**  
   Execute `baseline.py` to obtain baseline performance metrics.

4. **GNN Fully Connected Model**  
   Use `gnn_fully_connected.py` to compute results for the Fully Connected GNN approach.

5. **KNN / K-Means Results**  
   Run `gnn.py` to test either the KNN or K-Means graph construction method. Modify the `method` parameter (line 111) to `"knn"` or `"kmeans"` as desired.
