#!/usr/bin/env python3
import argparse
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors

#main function which is used for different distances selection, user can select Euclidean ,manhattan and maximum
def compute_distances(X, C, metric):
    if metric == "euclidean":
        return np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
    elif metric == "manhattan":
        return np.sum(np.abs(X[:, None, :] - C[None, :, :]), axis=2)
    elif metric == "maximum":
        return np.max(np.abs(X[:, None, :] - C[None, :, :]), axis=2)
    else:
        raise ValueError("Possible Metrics : euclidean, manhattan, maximum")


#Kmeans algorithm function
def kmeans(X, k, metric="euclidean", max_iter=200, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    centroids = X[rng.choice(n, size=k, replace=False)].copy() #random initial centroids
   #main iteration loop for kmeans algorithm
    for i in range(max_iter):
        distances = compute_distances(X, centroids, metric) #calculate distance between points and centroids
        labels = np.argmin(distances, axis=1) #then assign cluster according to their distances
        new_centroids = centroids.copy() #here updating centroids
        for j in range(k):
            points_in_cluster = X[labels == j]
            if len(points_in_cluster) > 0:
                new_centroids[j] = points_in_cluster.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]  #if cluster is empty then keep old random centroids
        if np.allclose(centroids, new_centroids, atol=1e-6):  #if center is not changing and stable then stop
            break

        centroids = new_centroids

    return labels, centroids


#function implemented for DBSCAN
def dbscan(X, eps, min_samples, metric="euclidean"):   #default euclidean distance
    metric_poss = {
        "euclidean": "euclidean",
        "manhattan": "manhattan",
        "maximum": "chebyshev"
    }
    if metric not in metric_poss:
        raise ValueError("Matrix only be: euclidean, manhattan, maximum")
    nn = NearestNeighbors(radius=eps, metric=metric_poss[metric])
    nn.fit(X)
    neighbors = nn.radius_neighbors(X, return_distance=False)
    n = X.shape[0]
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neigh = neighbors[i]
        if len(neigh) < min_samples: #if there are not enough neighbors then it means its noise
            labels[i] = -1
            continue
#starting new cluster
        labels[i] = cluster_id
        seeds = list(neigh[neigh != i])

        while seeds:
            j = seeds.pop()
            if not visited[j]:
                visited[j] = True
                j_neigh = neighbors[j]
                if len(j_neigh) >= min_samples:
                    for p in j_neigh:
                        p = int(p)
                        if not visited[p]:
                            seeds.append(p)
            if labels[j] == -1:
                labels[j] = cluster_id

        cluster_id += 1

    return labels, cluster_id

# support for image as input
def load_image_pixels(path, resize=None):
    img = Image.open(path).convert("RGB")
    if resize is not None:
        img = img.resize((resize, resize), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape
    X = arr.reshape(-1, 3)  # (n,3)
    return X, (h, w)


def recolor_image(X, labels, shape_hw, noise_strategy="original"):
    h, w = shape_hw
    out = X.copy()
    for c in np.unique(labels): #here recolor each cluster with mean value
        if c == -1:
            continue
        mask = labels == c
        out[mask] = out[mask].mean(axis=0)

    if np.any(labels == -1):  #here outliers will be handles
        if noise_strategy == "black":
            out[labels == -1] = 0
        elif noise_strategy == "mean":
            out[labels == -1] = out.mean(axis=0)
    out = np.clip(out, 0, 255).astype(np.uint8).reshape(h, w, 3)
    return out


def save_image(arr, path):   #image saved on path
    Image.fromarray(arr).save(path)


#CLI for algorithms
def parse_args():
    p = argparse.ArgumentParser(description="Segmented image using Kmeans and DBSCAN")

    p.add_argument("--algo", required=True, choices=["kmeans", "dbscan"], help="Select algorithm to use")
    p.add_argument("--input", required=True, help="Path to input image")
    p.add_argument("--output", required=True, help="Path to output image")
    p.add_argument("--metric", default="euclidean", choices=["euclidean", "manhattan", "maximum"])
    p.add_argument("--resize", type=int, default=None, help="Resize image for DBSCAN)")

    # Parameters for KMeans
    p.add_argument("--k", default=2, type=int, help="Number of clusters")
    p.add_argument("--max-iter", type=int, default=100, help="KMeans iterations")
    p.add_argument("--seed", type=int, default=42, help="KMeans seed")
    # Parameters for DBSCAN
    p.add_argument("--eps", type=float, default=7.0, help="eps for DBSCAN")
    p.add_argument("--min-samples", type=int, default=20, help="DBSCAN min samples")
    p.add_argument("--noise-strategy", choices=["original", "mean", "black"], default="original")
    return p.parse_args()

def main():
    args = parse_args()
    X, shape_hw = load_image_pixels(args.input, resize=args.resize)
    if args.algo == "kmeans":
        labels, centroids = kmeans(X, k=args.k, metric=args.metric, max_iter=args.max_iter, seed=args.seed)
        print(f"Kmeans ended: k={args.k}, metric={args.metric}")
    else:
        labels, n_clusters = dbscan(X, eps=args.eps, min_samples=args.min_samples, metric=args.metric)
        print(f"DBSCAN ended: clusters={n_clusters}, metric={args.metric}, noise={np.sum(labels==-1)}")

    out_img = recolor_image(X, labels, shape_hw, noise_strategy=args.noise_strategy)
    save_image(out_img, args.output)
    print(f"Segmented Image saved location: {args.output}")

if __name__ == "__main__":
    main()
