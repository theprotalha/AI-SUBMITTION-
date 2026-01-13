print("\n--- Q1: Write Python code to implement K-Means clustering. ---\n")
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Define Data Points
# -------------------------------
points = np.array([
    [1, 3],  # P1
    [2, 2],  # P2
    [5, 8],  # P3
    [8, 5],  # P4
    [3, 9],  # P5
    [10, 7], # P6
    [3, 3],  # P7
    [9, 4],  # P8
    [3, 7]   # P9
])

labels = ['P1','P2','P3','P4','P5','P6','P7','P8','P9']

# -------------------------------
# Step 2: Initialize Centroids
# -------------------------------
C1 = np.array([3,3])   # P7
C2 = np.array([3,7])   # P9
C3 = np.array([9,4])   # P8

centroids = np.array([C1, C2, C3])

# -------------------------------
# Helper Function: Assign Points
# -------------------------------
def assign_clusters(points, centroids):
    clusters = []
    for p in points:
        distances = np.linalg.norm(p - centroids, axis=1)
        clusters.append(np.argmin(distances))
    return np.array(clusters)

# -------------------------------
# Helper Function: Update Centroids
# -------------------------------
def update_centroids(points, clusters, k=3):
    new_centroids = []
    for i in range(k):
        new_centroids.append(points[clusters == i].mean(axis=0))
    return np.array(new_centroids)

# -------------------------------
# Step 3: Manual Iterations
# -------------------------------
print("Iteration 1")
clusters = assign_clusters(points, centroids)
print("Cluster assignments:", clusters)
centroids = update_centroids(points, clusters)
print("Updated centroids:\n", centroids)

print("\nIteration 2")
clusters = assign_clusters(points, centroids)
print("Cluster assignments:", clusters)
centroids = update_centroids(points, clusters)
print("Updated centroids:\n", centroids)

# -------------------------------
# Step 4 & 5: Plot Points + Labels + Centroids
# -------------------------------
colors = ['red', 'blue', 'green']
plt.figure(figsize=(8,6))

# Plot points
for i, p in enumerate(points):
    plt.scatter(p[0], p[1], color=colors[clusters[i]], s=60)
    plt.text(p[0]+0.1, p[1]+0.1, labels[i], fontsize=12)

# Plot centroids
for i, c in enumerate(centroids):
    plt.scatter(c[0], c[1], color='black', marker='X', s=200)
    plt.text(c[0]+0.1, c[1]+0.1, f"C{i+1}", fontsize=14, color='black')

plt.title("K-Means Clustering (2 Iterations)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

print("____________________________________________________________________________________________________________________________________________")

print("\n--- Q2: Use the scikit-learn KMeans() library to cluster the same points. ---\n")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Define points
points = np.array([
    [1, 3], [2, 2], [5, 8], [8, 5], [3, 9],
    [10, 7], [3, 3], [9, 4], [3, 7]
])

Ks = [2, 3, 4]

for K in Ks:
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(points)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Print cluster info
    print(f"\nK = {K}")
    for i in range(K):
        cluster_points = points[labels == i]
        print(f"Cluster {i+1}: {len(cluster_points)} points, Centroid: {centroids[i]}")

    # Plot clusters
    plt.figure(figsize=(6,5))
    for i in range(K):
        plt.scatter(points[labels == i, 0], points[labels == i, 1], label=f'Cluster {i+1}')
    plt.scatter(centroids[:,0], centroids[:,1], color='black', marker='x', s=100, label='Centroids')
    plt.title(f'KMeans Clustering with K={K}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

print("____________________________________________________________________________________________________________________________________________")

print("\n--- Q3: Add a New User Point and Re-Cluster. ---\n")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Original 9 points + new user P10(6,2)
points = np.array([
    [1, 3], [2, 2], [5, 8], [8, 5], [3, 9],
    [10, 7], [3, 3], [9, 4], [3, 7], [6, 2]  # P10 added
])

K = 3

# Run KMeans
kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(points)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Identify cluster of P10
p10_cluster = labels[-1]

# Print cluster info
print(f"\nK = {K} with new user P10(6,2):")
for i in range(K):
    cluster_points = points[labels == i]
    print(f"Cluster {i+1}: {len(cluster_points)} points, Centroid: {centroids[i]}")
print(f"P10(6,2) belongs to Cluster {p10_cluster+1}")

# Plot clusters
plt.figure(figsize=(6,5))
for i in range(K):
    plt.scatter(points[labels == i, 0], points[labels == i, 1], label=f'Cluster {i+1}')
plt.scatter(centroids[:,0], centroids[:,1], color='black', marker='x', s=100, label='Centroids')
plt.scatter(6, 2, color='red', marker='*', s=150, label='P10(6,2)')
plt.title('KMeans Clustering with K=3 (Including P10)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

print("____________________________________________________________________________________________________________________________________________")

print("\n--- Q4: Distance Table + First Iteration Manually ---\n")
import numpy as np
import matplotlib.pyplot as plt

# Original 9 points
points = np.array([
    [1, 3], [2, 2], [5, 8], [8, 5], [3, 9],
    [10, 7], [3, 3], [9, 4], [3, 7]
])

point_labels = ["P1","P2","P3","P4","P5","P6","P7","P8","P9"]

# Initial centroids
centroids = np.array([
    [3, 3],  # C1
    [3, 7],  # C2
    [9, 4]   # C3
])

# Step 1 & 2: Compute Euclidean distances and assign clusters
distance_table = []
assigned_clusters = []

for i, point in enumerate(points):
    dists = np.linalg.norm(point - centroids, axis=1)  # Euclidean distances
    cluster = np.argmin(dists)  # Assign to closest centroid
    distance_table.append([*dists, cluster+1])
    assigned_clusters.append(cluster)

# Print distance table
print("Distance Table (First Iteration):")
print("Point\tDist to C1\tDist to C2\tDist to C3\tAssigned Cluster")
for i, row in enumerate(distance_table):
    print(f"{point_labels[i]}\t{row[0]:.2f}\t\t{row[1]:.2f}\t\t{row[2]:.2f}\t\t{int(row[3])}")

# Step 4: Compute new centroids
new_centroids = np.zeros_like(centroids, dtype=float)
for k in range(3):
    cluster_points = points[np.array(assigned_clusters) == k]
    new_centroids[k] = cluster_points.mean(axis=0)

print("\nNew Centroids after First Iteration:")
for i, c in enumerate(new_centroids):
    print(f"C{i+1}: {c}")

# Step 5: Plot first iteration clusters
colors = ['r', 'g', 'b']
plt.figure(figsize=(7,6))
for k in range(3):
    cluster_points = points[np.array(assigned_clusters) == k]
    plt.scatter(cluster_points[:,0], cluster_points[:,1], color=colors[k], label=f'Cluster {k+1}')
    for idx, pt in enumerate(cluster_points):
        plt.text(pt[0]+0.1, pt[1]+0.1, point_labels[np.where(points==pt)[0][0]], fontsize=9)

# Plot new centroids
plt.scatter(new_centroids[:,0], new_centroids[:,1], color='black', marker='x', s=100, label='New Centroids')
for i, c in enumerate(new_centroids):
    plt.text(c[0]+0.1, c[1]+0.1, f"C{i+1}", fontsize=10, fontweight='bold')

plt.title("K-Means: First Iteration Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.show()

print("____________________________________________________________________________________________________________________________________________")
