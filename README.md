1. **Download the Dataset:** Get the data from Kaggle: https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet?resource=download..
2. **Reformat Validation Data:** Execute `reformat_validation_set.py` to reorganize the validation files to match the structure of the training data.
3. **Baseline Results:** Run `baseline.py` to obtain the baseline performance metrics.
4. **GNN Fully Connected Model:** Use `gnn_fully_connected.py` to compute results using the Fully Connected GNN approach.
5. **KNN / K-Means Results:** Run `gnn.py` to test either the KNN or K-Means method. Adjust the method parameter (line 111) to "knn" or "kmeans" as desired.
