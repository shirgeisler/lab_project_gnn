# Graph Construction Methods for Image Classification Using Graph Neural Networks

This project explores different graph construction methods for image classification using Graph Neural Networks (GNNs).

## Instructions

1. **Download the Dataset**  
   Download the Tiny ImageNet dataset from Kaggle: [Tiny ImageNet Dataset](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet?resource=download).
   Place the downloaded file in the `data` folder and unzip it. You can use `extract.py` to handle the extraction if preferred—just be sure to adjust the file paths to match your setup.
   
3. **Reformat Validation Data**  
   Run `reformat_validation_set.py` to reorganize the validation files, ensuring they match the structure of the training data.

4. **Baseline Results**  
   Execute `baseline.py` to obtain baseline performance metrics.

5. **GNN Fully Connected Model**  
   Use `gnn_fully_connected.py` to compute results for the Fully Connected GNN approach.

6. **KNN / K-Means Results**  
   Run `gnn.py` to test either the KNN or K-Means graph construction method. Modify the `method` parameter (line 111) to `"knn"` or `"kmeans"` as desired.
