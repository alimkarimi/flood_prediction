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

pca = PCA(n_components=2)
pc_components = pca.fit_transform(obs_matrix_true_pos_scaled)

path = "./analysis/"
fn = "PCA_true_pos_floods_1996.jpg"

plt.scatter(x = pc_components[0: 24,0], y = pc_components[0: 24, 1], label='pos')
plt.scatter(x = pc_components[24:,0], y = pc_components[24:, 1], label='neg')
plt.title("PCA on ERA5 Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.savefig(path + fn)