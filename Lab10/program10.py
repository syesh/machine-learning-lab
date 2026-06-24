import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,classification_report

data = load_breast_cancer()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters = 2 , random_state = 42)
y_kmeans = kmeans.fit_predict(X_scaled)

pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X_scaled)

df = pd.DataFrame(X_pca, columns = ['PC1', 'PC2'])
df['Cluster'] = y_kmeans
df['True Label'] = y

plt.figure(figsize=(8,6))
sns.scatterplot(data=df,x='PC1', y='PC2', hue='Cluster',palette='Set1' , s = 100 , edgecolor = 'black' , alpha=0.7)
plt.title('K-Means Clustering of breast cancer Dataset')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.legend(title = "cluster")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(data=df,x='PC1', y='PC2', hue='Cluster',palette='Set1' , s = 100 , edgecolor = 'black' , alpha=0.7)
centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0], centers[:,1], c='red', s=200,marker = 'X', label='Centroids')
plt.title('K-Means Clustering with centroids')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.legend(title = "cluster")
plt.grid(True)
plt.show()
