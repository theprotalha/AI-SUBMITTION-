print("\n--- Q1: Apply Fuzzy C-Means clustering. ---\n")
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.cluster import KMeans

np.random.seed(42)
cluster1 = np.random.randn(100, 2) + [2, 2]
cluster2 = np.random.randn(100, 2) + [7, 7]
cluster3 = np.random.randn(100, 2) + [2, 7]
data = np.vstack((cluster1, cluster2, cluster3))

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.T, 3, 2, error=0.005, maxiter=1000)
labels_fcm = np.argmax(u, axis=0)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for i in range(3):
    plt.scatter(data[labels_fcm==i,0], data[labels_fcm==i,1])
plt.scatter(cntr[:,0], cntr[:,1], c='black', marker='x')
plt.title('Fuzzy C-Means Clustering')

kmeans = KMeans(n_clusters=3, random_state=42).fit(data)
labels_kmeans = kmeans.labels_

plt.subplot(1,2,2)
for i in range(3):
    plt.scatter(data[labels_kmeans==i,0], data[labels_kmeans==i,1])
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='black', marker='x')
plt.title('K-Means Clustering')
plt.show()

indices = np.random.choice(data.shape[0], 5, replace=False)
membership_values = u[:, indices]

print("Membership values for 5 random points:")
print(membership_values.T)
print("\nFuzzy Partition Coefficient (FPC):", fpc)

print("____________________________________________________________________________________________________________________________________________")
print("\n--- Q2: Fuzzy C-Means clustering on Iris dataset ---\n")
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

np.random.seed(42)
iris = load_iris()
X = iris.data
y = iris.target

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_norm.T, 3, 2, error=0.005, maxiter=1000)
labels_fcm = np.argmax(u, axis=0)

print("Predicted clusters for first 20 samples:")
print(labels_fcm[:20])
print("Actual labels for first 20 samples:")
print(y[:20])

mapping_fcm = {}
for c in range(3):
    idx = np.where(labels_fcm == c)[0]
    if idx.size > 0:
        maj = np.bincount(y[idx]).argmax()
        mapping_fcm[c] = maj
mapped_fcm = np.array([mapping_fcm[l] for l in labels_fcm])
acc_fcm = (mapped_fcm == y).mean()

print("\nAccuracy using majority mapping (FCM):", acc_fcm)
print("\nFuzzy Partition Coefficient (FPC):", fpc)

kmeans = KMeans(n_clusters=3, random_state=42).fit(X_norm)
labels_km = kmeans.labels_
mapping_km = {}
for c in range(3):
    idx = np.where(labels_km == c)[0]
    if idx.size > 0:
        maj = np.bincount(y[idx]).argmax()
        mapping_km[c] = maj
mapped_km = np.array([mapping_km[l] for l in labels_km])
acc_km = (mapped_km == y).mean()
print("\nAccuracy using majority mapping (K-Means):", acc_km)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for i in range(3):
    plt.scatter(X_norm[labels_fcm==i,0], X_norm[labels_fcm==i,1])
plt.scatter(cntr[:,0], cntr[:,1], c='black', marker='x')
plt.title('Fuzzy C-Means Clustering (Iris, first 2 features)')

plt.subplot(1,2,2)
for i in range(3):
    plt.scatter(X_norm[labels_km==i,0], X_norm[labels_km==i,1])
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='black', marker='x')
plt.title('K-Means Clustering (Iris, first 2 features)')
plt.show()

print("____________________________________________________________________________________________________________________________________________")
print("\n--- Q3: Image Segmentation using Fuzzy C-Means ---\n")
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from skimage import data, color

image = data.camera()
gray = color.rgb2gray(image) if image.ndim == 3 else image
pixels = gray.flatten()

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    pixels.reshape(1, -1), c=3, m=2, error=0.005, maxiter=1000
)
labels_fcm = np.argmax(u, axis=0)
segmented_fcm = labels_fcm.reshape(gray.shape)

pixels_reshaped = pixels.reshape(-1, 1)
kmeans = KMeans(n_clusters=3, random_state=42).fit(pixels_reshaped)
labels_kmeans = kmeans.labels_
segmented_kmeans = labels_kmeans.reshape(gray.shape)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(gray, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(segmented_fcm, cmap='gray')
plt.title("FCM Segmentation (3 clusters)")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(segmented_kmeans, cmap='gray')
plt.title("K-Means Segmentation (3 clusters)")
plt.axis('off')
plt.show()

print("Fuzzy Partition Coefficient (FPC):", fpc)

print("____________________________________________________________________________________________________________________________________________")
print("\n--- Q4: Market Segmentation using Fuzzy C-Means ---\n")
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pandas as pd

data = pd.DataFrame({
    'Age': [25,34,45,23,51,62,41,29,38,47,55,31,28,49,36],
    'Income': [15,40,65,20,80,90,55,25,45,70,85,35,30,75,50],
    'SpendingScore': [39,81,6,77,40,5,75,66,50,20,10,60,72,30,55]
})

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(data)

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_norm.T, 3, 2, error=0.005, maxiter=1000)
labels_fcm = np.argmax(u, axis=0)

print("Membership matrix (first 10 customers):")
print(u[:, :10].T)
print("\nCluster centers (normalized):")
print(cntr)

data['FCM_Cluster'] = labels_fcm
print("\nCustomer assignments (FCM):")
print(data[['Age','Income','SpendingScore','FCM_Cluster']])

kmeans = KMeans(n_clusters=3, random_state=42).fit(X_norm)
labels_kmeans = kmeans.labels_
data['KMeans_Cluster'] = labels_kmeans
print("\nCustomer assignments (K-Means):")
print(data[['Age','Income','SpendingScore','KMeans_Cluster']])

print("\nFuzzy Partition Coefficient (FPC):", fpc)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for i in range(3):
    plt.scatter(X_norm[labels_fcm==i,0], X_norm[labels_fcm==i,1])
plt.scatter(cntr[:,0], cntr[:,1], c='black', marker='x')
plt.title('Fuzzy C-Means Clustering')

plt.subplot(1,2,2)
for i in range(3):
    plt.scatter(X_norm[labels_kmeans==i,0], X_norm[labels_kmeans==i,1])
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='black', marker='x')
plt.title('K-Means Clustering')
plt.show()

print("____________________________________________________________________________________________________________________________________________")
print("\n--- Q5: COVID-19 Clustering using Fuzzy C-Means ---\n")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("DataFiles/owid-covid-data.csv")

# Selected countries
countries = ["Pakistan", "India", "China", "Iran", "Afghanistan"]

# Filter countries
df = data[data['location'].isin(countries)]

# Take latest available record for each country
df = df.sort_values('date').groupby('location').tail(1)

# Select required features
df = df[['location', 'total_cases', 'total_deaths', 'population']]

df.rename(columns={'location': 'Country'}, inplace=True)

# Handle missing values
df.fillna(0, inplace=True)

# Normalize features
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(df[['total_cases', 'total_deaths', 'population']])

# ---------- Fuzzy C-Means ----------
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_norm.T, 
    c=2, 
    m=2, 
    error=0.005, 
    maxiter=1000, 
    init=None
)

labels_fcm = np.argmax(u, axis=0)

print("Membership Matrix (FCM):")
print(pd.DataFrame(u.T, columns=['Cluster 0', 'Cluster 1'], index=df['Country']))

df['FCM_Cluster'] = labels_fcm

print("\nFinal Clusters (FCM):")
print(df[['Country', 'FCM_Cluster']])

print("\nFuzzy Partition Coefficient (FPC):", fpc)

# ---------- K-Means ----------
kmeans = KMeans(n_clusters=2, random_state=42)
labels_kmeans = kmeans.fit_predict(X_norm)

df['KMeans_Cluster'] = labels_kmeans

print("\nFinal Clusters (K-Means):")
print(df[['Country', 'KMeans_Cluster']])

# ---------- Visualization ----------
plt.figure(figsize=(12,5))

# FCM Plot
plt.subplot(1,2,1)
for i in range(2):
    plt.scatter(X_norm[labels_fcm==i, 0], X_norm[labels_fcm==i, 1], label=f'Cluster {i}')
plt.scatter(cntr[:,0], cntr[:,1], c='black', marker='x')
plt.title('Fuzzy C-Means Clustering')
plt.xlabel('Total Cases (Normalized)')
plt.ylabel('Total Deaths (Normalized)')
plt.legend()

# K-Means Plot
plt.subplot(1,2,2)
for i in range(2):
    plt.scatter(X_norm[labels_kmeans==i, 0], X_norm[labels_kmeans==i, 1], label=f'Cluster {i}')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='black', marker='x')
plt.title('K-Means Clustering')
plt.xlabel('Total Cases (Normalized)')
plt.ylabel('Total Deaths (Normalized)')
plt.legend()

plt.tight_layout()
plt.show()

print("____________________________________________________________________________________________________________________________________________")
