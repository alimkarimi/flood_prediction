from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np

from matplotlib import pyplot as plt

import os

from dataloading_scripts.feature_builder import df_pos_neg

obs_matrix_in_list = df_pos_neg['feature_vector'].tolist() # get feature vectors ordered into a list
obs_matrix_all = np.vstack(obs_matrix_in_list) # stack the feature vectors vertically.

labels = df_pos_neg['label'].to_list()
labels = np.vstack(labels)


scaler = StandardScaler()
obs_matrix_true_pos_scaled = scaler.fit_transform(obs_matrix_all)

pca = PCA(n_components=5)
pc_components = pca.fit_transform(obs_matrix_true_pos_scaled)

path = "./eda/"
fn = "PCA_true_pos_floods_1996.jpg"

plt.scatter(x = pc_components[10000:,0], y = pc_components[10000:, 1], label='neg')
plt.scatter(x = pc_components[0:10000,0], y = pc_components[0: 10000, 1], label='pos')

plt.title("PCA on ERA5 Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.savefig(path + fn,)
plt.clf()

# build scree plot:
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
fn = "PCA_scree_plot.jpg"
plt.savefig(path + fn)

plt.clf()