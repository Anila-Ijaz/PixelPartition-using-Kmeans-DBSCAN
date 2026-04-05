# PixelPartition-using-Kmeans-DBSCAN
This project implements unsupervised clustering algorithms from scratch, including K-Means and DBSCAN, and applies them to image segmentation.  Each pixel of an image is treated as a data point in RGB color space and grouped into clusters based on similarity. The segmented output image is reconstructed using cluster-wise average colors.
Key Features:
- Implementation of K-Means and DBSCAN from scratch
- Supports multiple distance metrics: Euclidean, Manhattan, and Maximum
- Works with arbitrary dimensional feature vectors
- Command-line interface for experimentation
- Image segmentation based on clustering results

Technologies:
Python, NumPy, Matplotlib
